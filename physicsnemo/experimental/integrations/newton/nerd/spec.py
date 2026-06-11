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

"""Model dimension spec and training configuration for NeRD workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NeRDModelSpec:
    """Dimensions inferred by a NeRD workflow for a dynamics model.

    ``input_dim`` and ``prediction_dim`` are always the final feature widths.
    ``input_shape`` and ``prediction_shape`` include any entity dimensions. For
    example, a joint-space model has ``input_shape=(features,)`` while an
    entity-aware rigid-body model has ``input_shape=(bodies, features)``.
    """

    input_dim: int
    prediction_dim: int
    context_frames: int
    input_shape: tuple[int, ...] = ()
    prediction_shape: tuple[int, ...] = ()

    @property
    def entity_count(self) -> int | None:
        """Entity count for an entity-token state, otherwise ``None``."""
        return self.input_shape[-2] if len(self.input_shape) == 2 else None


@dataclass
class NeRDTrainingConfig:
    """Architecture-neutral NeRD optimization settings.

    Attributes
    ----------
    context_frames : int
        Number of past frames fed to the model.
    epochs : int
        Number of training epochs.
    steps_per_epoch : int
        Number of optimizer updates per epoch.
    batch_size : int
        Global batch size, split across distributed ranks.
    lr_start : float
        Learning rate at the first epoch. Together with ``lr_end``, this defines
        a per-epoch linear learning-rate decay.
    lr_end : float
        Learning rate at the final epoch.
    weight_decay : float
        AdamW weight decay.
    grad_clip : float
        Maximum global gradient norm passed to ``clip_grad_norm_``.
    normalization_floor : float
        Minimum per-channel standard deviation used to prevent division by zero
        for constant or nearly constant channels.
    active_delta_threshold : float
        Channels whose global absolute maximum delta does not exceed this value
        are excluded from the loss and zeroed in model output. A value that is
        too large can silently discard meaningful channels.
    """

    context_frames: int = 10
    epochs: int = 50
    steps_per_epoch: int = 500
    batch_size: int = 512
    lr_start: float = 1.0e-3
    lr_end: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    normalization_floor: float = 1.0e-6
    active_delta_threshold: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in ("context_frames", "epochs", "steps_per_epoch", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("lr_start", "lr_end", "grad_clip", "normalization_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "active_delta_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
