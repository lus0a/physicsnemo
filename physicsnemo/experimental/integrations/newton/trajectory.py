# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn Newton rollouts into a PhysicsNeMo dataset for training.

A rollout from :meth:`NewtonEnv.rollout` is a ``(time, *feature)`` observation
trajectory. To train a next-step model on it you need sliding windows, optional
normalization, and a ``DataLoader`` -- all of which PhysicsNeMo already provides.
This module provides the adapter: ``TrajectoryWindowReader`` is a
``physicsnemo.datapipes.Reader`` of ``input -> next`` windows, and
``trajectory_dataset`` wraps it in a ``physicsnemo.datapipes.Dataset`` with a
``Normalize`` transform.

.. code-block:: python

    from physicsnemo.datapipes import DataLoader
    roll = env.rollout(steps=64)
    data = trajectory_dataset(roll.observations, window=8, predict_steps=1)
    for batch in DataLoader(data, batch_size=32):
        x, y = batch["input"], batch["target"]

Scope is the single autoregressive stream (the target is the frames right after
the input window), which covers the common next-state surrogate. A two-stream
setup (separate command/target) is out of scope and stays bespoke.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from physicsnemo.datapipes import Dataset, Reader
from physicsnemo.datapipes.transforms.normalize import Normalize
from physicsnemo.experimental.integrations.newton.data import field_to_torch


class TrajectoryWindowReader(Reader):
    """A ``physicsnemo.datapipes.Reader`` of sliding windows over trajectories.

    Each sample is ``{"input": (window, *feature), "target": (predict_steps, *feature)}``
    where ``target`` is the ``predict_steps`` frames immediately after ``input``. Windows
    never straddle two trajectories, so a list of separate rollouts is windowed
    independently.

    Parameters
    ----------
    trajectories : torch.Tensor or Sequence[torch.Tensor]
        One ``(time, *feature)`` tensor or a sequence of tensors, such as
        several rollouts or one rollout per parallel world.
    window : int
        Number of input frames per sample.
    predict_steps : int, optional
        Number of target frames immediately following the window.
    stride : int, optional
        Step between successive window starts.
    """

    def __init__(
        self,
        trajectories: torch.Tensor | Sequence[torch.Tensor],
        *,
        window: int,
        predict_steps: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.window, self.predict_steps, self.stride = (
            int(window),
            int(predict_steps),
            int(stride),
        )
        if self.window <= 0 or self.predict_steps <= 0 or self.stride <= 0:
            raise ValueError("window, predict_steps, and stride must be positive")
        items = (
            [trajectories]
            if isinstance(trajectories, torch.Tensor)
            else list(trajectories)
        )
        self._traj = [
            field_to_torch(t, dtype=torch.float32, clone=True).detach().cpu()
            for t in items
        ]
        if not self._traj:
            raise ValueError("trajectories must contain at least one trajectory")
        if any(traj.ndim < 2 for traj in self._traj):
            raise ValueError("each trajectory must have shape (time, *feature)")
        feature_shape = self._traj[0].shape[1:]
        if any(traj.shape[1:] != feature_shape for traj in self._traj[1:]):
            raise ValueError("all trajectories must have the same feature shape")
        span = self.window + self.predict_steps
        self._index = [
            (ti, start)
            for ti, traj in enumerate(self._traj)
            for start in range(0, traj.shape[0] - span + 1, self.stride)
        ]
        if not self._index:
            raise ValueError(
                f"no windows: need at least window + predict_steps = {span} frames per trajectory"
            )

    @property
    def trajectories(self) -> list[torch.Tensor]:
        """Prepared detached per-trajectory tensors (validated, float32 CPU)."""
        return self._traj

    def __len__(self) -> int:
        return len(self._index)

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        ti, start = self._index[index]
        traj = self._traj[ti]
        return {
            "input": traj[start : start + self.window],
            "target": traj[
                start + self.window : start + self.window + self.predict_steps
            ],
        }


def _fit_mean_std(
    trajectories: Sequence[torch.Tensor], eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-last-channel mean/std in one streamed pass (no window blow-up).

    Statistics are computed over the final feature axis and pooled across time
    and every other leading feature axis: for a ``(time, N, 3)`` trajectory the
    result is a ``(3,)`` mean/std pooled over all frames and all ``N`` entries,
    which :class:`~physicsnemo.datapipes.transforms.normalize.Normalize`
    broadcasts back over those axes at apply time.
    """
    feature = trajectories[0].shape[-1]
    count, total, total_sq = (
        0,
        torch.zeros(feature, dtype=torch.float64),
        torch.zeros(feature, dtype=torch.float64),
    )
    for traj in trajectories:
        flat = traj.reshape(-1, feature).double()
        count += flat.shape[0]
        total += flat.sum(0)
        total_sq += (flat * flat).sum(0)
    mean = total / max(count, 1)
    var = (total_sq / max(count, 1) - mean * mean).clamp_min(0.0)
    return mean.float(), var.sqrt().clamp_min(eps).float()


def trajectory_dataset(
    trajectories: torch.Tensor | Sequence[torch.Tensor],
    *,
    window: int,
    predict_steps: int = 1,
    stride: int = 1,
    normalize: bool = True,
    device: str | torch.device | None = None,
) -> Dataset:
    """Build a PhysicsNeMo ``Dataset`` of windowed rollouts.

    Wraps ``TrajectoryWindowReader`` and, when ``normalize=True``, fits
    mean/std statistics per last-axis channel -- pooled across time and all other
    leading feature axes (e.g. across all ``N`` particles of a ``(time, N, 3)``
    trajectory) -- and applies a shared
    :class:`~physicsnemo.datapipes.transforms.normalize.Normalize` to ``input`` and
    ``target`` (same feature axis, so one stat set is correct). Iterate the
    returned dataset with ``physicsnemo.datapipes.DataLoader`` (samples are
    TensorDicts, which plain ``torch.utils.data.DataLoader`` cannot collate).
    """
    reader = TrajectoryWindowReader(
        trajectories, window=window, predict_steps=predict_steps, stride=stride
    )
    transforms = None
    if normalize:
        mean, std = _fit_mean_std(reader.trajectories)
        # "input" and "target" share the same physical channels, so they share one
        # stat set; the same tensor is intentionally aliased under both keys.
        # Safe because Normalize only ever reads these stats (never updates one in
        # place); clone for the second key if a future Normalize mutates a stat.
        transforms = Normalize(
            ["input", "target"],
            method="mean_std",
            means={"input": mean, "target": mean},
            stds={"input": std, "target": std},
        )
    return Dataset(reader, transforms=transforms, device=device)
