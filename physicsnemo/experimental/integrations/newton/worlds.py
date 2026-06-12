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

"""Per-world reads for a batched (many-world) Newton model.

Newton scales by replicating a scene into many parallel "worlds" that advance in
a single ``solver.step``. The entities of all worlds are concatenated into one
flat array, and ``model.<entity>_world`` gives each row's world id (``-1`` for
globals such as a shared ground plane). Reading per-world results then turns into
host-copy bookkeeping::

    pw = model.particle_world.numpy()          # which rows are world w?
    for w in range(B):                         # a Python loop, on the host
        out[w] = positions[pw == w].mean(0)

:class:`WorldView` replaces that with readable on-device reductions, so a
researcher can run ``B`` designs as ``B`` worlds in one step and pull ``B``
results without a host round-trip or a Python loop.

It is the row-companion to :mod:`physicsnemo.experimental.integrations.newton.state`: the state
views answer "which columns are position?"; this view answers "which rows are
world ``w``?".

    from physicsnemo.experimental.integrations.newton.worlds import WorldView
    from physicsnemo.experimental.integrations.newton import particles

    worlds = WorldView(env.model)                       # particle entity by default
    z = particles(env.state).positions[:, 2]            # (num_particles,)
    mean_height = worlds.per_world_mean(z)              # (world_count,)
"""

from __future__ import annotations

from typing import Any

import torch

from physicsnemo.experimental.integrations.newton.data import field_to_torch


class WorldView:
    """Group a many-world model's flat per-entity arrays by world.

    Parameters
    ----------
    model : Any
        Finalized Newton ``Model`` built with multiple worlds.
    entity : {"particle", "body", "shape", "joint"}, optional
        Entity family to group. Select the family read by the observation, such
        as ``"body"`` for rigid-body or cable scenes.
    """

    _WORLD_ATTR = {
        "particle": "particle_world",
        "body": "body_world",
        "shape": "shape_world",
        "joint": "joint_world",
    }

    def __init__(self, model: Any, entity: str = "particle") -> None:
        if entity not in self._WORLD_ATTR:
            raise ValueError(
                f"entity must be one of {sorted(self._WORLD_ATTR)}, got {entity!r}"
            )
        self.entity = entity
        self.world_count = int(model.world_count)
        if self.world_count <= 0:
            raise ValueError("model.world_count must be positive")
        # `<entity>_world` is a wp.int32 array; bincount/scatter need int64 indices.
        world_ids = field_to_torch(getattr(model, self._WORLD_ATTR[entity])).long()
        invalid = (world_ids < -1) | (world_ids >= self.world_count)
        if bool(invalid.any()):
            bad_id = int(world_ids[invalid][0])
            raise ValueError(
                f"{self._WORLD_ATTR[entity]} contains invalid id {bad_id}; "
                f"expected -1 for globals or an id in [0, {self.world_count - 1}]"
            )
        self._world_ids = world_ids
        self._valid = (
            world_ids >= 0
        )  # drop -1 globals (shared ground, etc.); never clamp them to world 0

    @property
    def world_index(self) -> torch.Tensor:
        """The world id of each non-global entity row, shape ``(num_valid,)``."""
        return self._world_ids[self._valid]

    @property
    def counts(self) -> torch.Tensor:
        """Number of entities in each world, shape ``(world_count,)``."""
        return torch.bincount(self.world_index, minlength=self.world_count)

    def per_world_sum(
        self, values: torch.Tensor, *, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sum ``values`` within each world.

        Parameters
        ----------
        values : torch.Tensor
            Tensor shaped ``(num_entities, *feature)`` and aligned with the
            entity rows, such as ``particles(state).positions``.
        mask : torch.Tensor, optional
            Boolean tensor shaped ``(num_entities,)`` selecting rows to include.

        Returns
        -------
        torch.Tensor
            Per-world sums shaped ``(world_count, *feature)``.
        """
        values = self._validate_values(values)
        if mask is not None:
            keep = self._validate_mask(mask)[self._valid].reshape(
                values.shape[0], *([1] * (values.ndim - 1))
            )
            values = values * keep.to(values.dtype)
        out = values.new_zeros((self.world_count, *values.shape[1:]))
        index = self.world_index.reshape(
            values.shape[0], *([1] * (values.ndim - 1))
        ).expand_as(values)
        return out.scatter_add_(0, index, values)

    def per_world_count(self, mask: torch.Tensor) -> torch.Tensor:
        """Count the entities selected by ``mask`` in each world, shape ``(world_count,)``."""
        return torch.bincount(
            self.world_index[self._validate_mask(mask)[self._valid]],
            minlength=self.world_count,
        )

    def per_world_mean(
        self, values: torch.Tensor, *, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mean of ``values`` within each world (empty/masked-out worlds give 0)."""
        dtype = (
            values.dtype
            if values.is_floating_point() or values.is_complex()
            else torch.get_default_dtype()
        )
        total = self.per_world_sum(values.to(dtype), mask=mask)
        denom = (
            self.per_world_count(mask) if mask is not None else self.counts
        ).clamp_min(1)
        return total / denom.reshape(denom.shape[0], *([1] * (total.ndim - 1))).to(
            total.dtype
        )

    def _validate_values(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(values, torch.Tensor) or values.ndim == 0:
            raise ValueError("values must be a tensor shaped (num_entities, *feature)")
        if values.shape[0] != self._world_ids.shape[0]:
            raise ValueError(
                f"values has {values.shape[0]} rows, expected {self._world_ids.shape[0]}"
            )
        if values.device != self._world_ids.device:
            raise ValueError(
                "values and the model world-index array must share a device"
            )
        return values[self._valid]

    def _validate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.ndim != 1
            or mask.shape[0] != self._world_ids.shape[0]
        ):
            raise ValueError(
                f"mask must be a boolean tensor shaped ({self._world_ids.shape[0]},)"
            )
        if mask.device != self._world_ids.device:
            raise ValueError("mask and the model world-index array must share a device")
        return mask
