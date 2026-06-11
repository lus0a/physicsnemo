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

"""Normalization statistics and shared tensor utilities for NeRD models.

These helpers are deliberately low-level and codec-agnostic so both training
and deployment can share input assembly, normalization, and model-invocation
logic without creating an import cycle through the higher-level workflows.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from physicsnemo.experimental.integrations.newton.data import field_to_torch
from physicsnemo.experimental.integrations.newton.distributed import (
    _all_reduce,
    resolve_device,
)
from physicsnemo.utils.logging import PythonLogger

# Default progress-log callback for the public NeRD entry points. Uses the
# PhysicsNeMo logger (level- and format-aware) instead of the builtin print.
# Call sites already gate logging on rank 0, so this stays rank-safe.
_nerd_log = PythonLogger(name="nerd").info


@dataclass
class NeRDNormalizers:
    """Frozen model-input and relative-dynamics normalization statistics."""

    input_mean: torch.Tensor
    input_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor

    def to(self, device: str | torch.device) -> NeRDNormalizers:
        """Move all statistics to ``device``."""
        return NeRDNormalizers(
            self.input_mean.to(device),
            self.input_std.to(device),
            self.target_mean.to(device),
            self.target_std.to(device),
        )


def _append_inputs(state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if inputs.shape[:2] != state.shape[:2] or inputs.ndim < 3:
        raise ValueError(
            "inputs must have shape [batch, time, *external_input_shape] matching state"
        )
    entity_dims = state.ndim - 3
    input_entity_shape = tuple(inputs.shape[2:-1])
    state_entity_shape = tuple(state.shape[2:-1])
    if not input_entity_shape:
        expanded = inputs.reshape(
            *inputs.shape[:2], *([1] * entity_dims), inputs.shape[-1]
        ).expand(*state.shape[:-1], inputs.shape[-1])
    elif input_entity_shape == state_entity_shape:
        expanded = inputs
    else:
        raise ValueError(
            "inputs must be global per world or align with the encoded state "
            f"entities; got {input_entity_shape} for state entities "
            f"{state_entity_shape}"
        )
    return torch.cat((state, expanded), dim=-1)


def _input_tensor(
    frame_inputs: Any,
    batch_size: int,
    external_input_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if frame_inputs is None:
        if _input_width(external_input_shape) > 0:
            raise ValueError(
                "inputs are required because this model was trained with them"
            )
        value = torch.empty(
            (batch_size, *external_input_shape), dtype=torch.float32, device=device
        )
    else:
        try:
            value = field_to_torch(
                frame_inputs,
                dtype=torch.float32,
                device=device,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise TypeError(
                "NeRD step inputs must be tensor-like per-world features, not a "
                f"{type(frame_inputs).__name__}. Extract the application features "
                "in input_from_step; use NeRDControlInput for a Newton control field."
            ) from error
    expected = (batch_size, *external_input_shape)
    if tuple(value.shape) != expected:
        raise ValueError(f"expected inputs shape {expected}, got {tuple(value.shape)}")
    return value


def _input_width(external_input_shape: tuple[int, ...]) -> int:
    if not external_input_shape:
        raise ValueError("external_input_shape must include a final feature dimension")
    return external_input_shape[-1]


def _runtime_device(
    device: str | torch.device | None,
    value: torch.Tensor | np.ndarray,
    fallback: torch.Tensor,
) -> torch.device:
    """Resolve runtime placement from an explicit choice, input, or model state."""
    if device is not None:
        return resolve_device(device)
    if isinstance(value, torch.Tensor):
        return resolve_device(value.device)
    return resolve_device(fallback.device)


def _model_prediction(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    prediction_shape: tuple[int, ...],
) -> torch.Tensor:
    prediction = model(inputs)
    expected = (*inputs.shape[:2], *prediction_shape)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError(
            f"dynamics model must return a torch.Tensor shaped {expected}, "
            f"got {type(prediction).__name__}"
        )
    if tuple(prediction.shape) != expected:
        raise ValueError(
            "dynamics model must preserve batch/time and the inferred state "
            f"structure, returning {expected}; got {tuple(prediction.shape)}"
        )
    return prediction


def _predict_delta(
    model: torch.nn.Module,
    normalized_inputs: torch.Tensor,
    prediction_shape: tuple[int, ...],
    normalizers: NeRDNormalizers,
    active_delta_mask: torch.Tensor,
) -> torch.Tensor:
    prediction = _model_prediction(model, normalized_inputs, prediction_shape)[:, -1]
    # active_delta_mask is a bool tensor used as a multiplicative gate; cast to
    # the prediction dtype rather than relying on implicit bool->float promotion.
    return (
        prediction * normalizers.target_std + normalizers.target_mean
    ) * active_delta_mask.to(prediction.dtype)


def _global_mean_std(
    data: torch.Tensor, floor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed population mean/std over the leading batch/time dimensions.

    Sums are accumulated in float64 to limit cancellation error in
    ``E[x^2] - E[x]^2``. ``clamp_min(0.0)`` guards rounding-induced negative
    variance for near-constant channels, and ``floor``
    (``config.normalization_floor``) sets the minimum returned std.
    """
    flat = data.flatten(end_dim=1)
    value_sum = flat.sum(dim=0, dtype=torch.float64)
    square_sum = flat.square().sum(dim=0, dtype=torch.float64)
    count = torch.tensor(float(flat.shape[0]), dtype=torch.float64, device=data.device)
    _all_reduce(value_sum, square_sum, count)
    mean = value_sum / count
    variance = (square_sum / count - mean.square()).clamp_min(0.0)
    return mean.to(data.dtype), torch.sqrt(variance).clamp_min(floor).to(data.dtype)


def _global_abs_max(data: torch.Tensor) -> torch.Tensor:
    value = data.abs().flatten(end_dim=1).amax(dim=0)
    _all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return value


def _model_class_name(model: torch.nn.Module) -> str:
    cls = model.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _model_for_device(
    model: torch.nn.Module, device: str | torch.device
) -> torch.nn.Module:
    """Return an inference model on ``device`` without moving shared runtimes."""
    torch_device = torch.device(device)
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if all(tensor.device == torch_device for tensor in tensors):
        return model
    return deepcopy(model).to(torch_device)


def _state_rmse(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Raw-state RMSE per trajectory."""
    return torch.sqrt((prediction - truth).square().flatten(start_dim=1).mean(dim=1))
