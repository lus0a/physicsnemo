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

"""Dynamics-model construction, explicit adapters, and checkpoint reconstruction.

These helpers map a :class:`NeRDModelSpec` to a concrete dynamics model
(supported registry name, ready module, or builder) and provide the safe
allowlist used to self-describe and reconstruct the built-in NeRD models from a
checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from physicsnemo.core import ModelRegistry
from physicsnemo.experimental.integrations.newton.nerd.spec import NeRDModelSpec


def _resolve_nerd_model(
    dynamics_model: str | torch.nn.Module | Callable[[NeRDModelSpec], torch.nn.Module],
    spec: NeRDModelSpec,
    *,
    device: str | torch.device,
    model_kwargs: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Resolve an explicitly selected registry model, ready module, or builder."""
    kwargs = dict(model_kwargs or {})
    if dynamics_model == "auto":
        raise ValueError(
            'dynamics_model="auto" is not supported; choose "NeRDTransformer" '
            'for vector state, "NeRDEntityTransformer" for entity-token state, '
            'or pass "FullyConnected", a ready module, or a NeRDModelSpec builder'
        )
    if isinstance(dynamics_model, str):
        model_class = ModelRegistry().factory(dynamics_model)
        inferred = _registered_model_dimensions(dynamics_model, spec)
        inferred.update(kwargs)
        model = model_class(**inferred)
    elif isinstance(dynamics_model, torch.nn.Module):
        if kwargs:
            raise ValueError("model_kwargs cannot be used with a ready dynamics model")
        model = dynamics_model
    elif callable(dynamics_model):
        if kwargs:
            raise ValueError(
                "model_kwargs cannot be used with a dynamics model builder"
            )
        model = dynamics_model(spec)
    else:
        raise TypeError(
            "dynamics_model must be a ModelRegistry name, torch.nn.Module, "
            f"or NeRDModelSpec builder; got {type(dynamics_model).__name__}"
        )
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "a dynamics model builder must return a torch.nn.Module, "
            f"got {type(model).__name__}"
        )
    return model.to(device)


def _registered_model_dimensions(
    model_name: str, spec: NeRDModelSpec
) -> dict[str, Any]:
    """Return explicit NeRD adapters for supported PhysicsNeMo registry models."""
    if model_name == "NeRDTransformer":
        return {
            "input_dim": spec.input_dim,
            "prediction_dim": spec.prediction_dim,
            "context_frames": spec.context_frames,
        }
    if model_name == "FullyConnected":
        return {
            "in_features": spec.input_dim,
            "out_features": spec.prediction_dim,
        }
    if model_name == "NeRDEntityTransformer":
        if spec.entity_count is None:
            raise ValueError(
                "NeRDEntityTransformer requires entity-token input, but the "
                f"NeRD state shape is {spec.input_shape}"
            )
        return {
            "feature_dim": spec.input_dim,
            "prediction_dim": spec.prediction_dim,
            "num_entities": spec.entity_count,
            "context_frames": spec.context_frames,
        }
    raise ValueError(
        f"no built-in NeRD adapter is defined for registry model {model_name!r}; "
        "pass a ready module or a builder receiving NeRDModelSpec"
    )


class _CompiledNeRDModel(torch.nn.Module):
    """Mark CUDA-graph inference steps around one compiled NeRD model."""

    def __init__(self, model: torch.nn.Module, *, mode: str) -> None:
        super().__init__()
        self.model = model
        self.compiled = torch.compile(model, mode=mode, fullgraph=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        return self.compiled(inputs)


def _checkpoint_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the serializable model behind an inference-only wrapper."""
    while isinstance(model, _CompiledNeRDModel):
        model = model.model
    return model


def _model_descriptor(model: torch.nn.Module) -> dict[str, Any] | None:
    args = getattr(model, "_args", None)
    name = _checkpoint_model_name(model)
    if (
        name is None
        or not isinstance(args, dict)
        or not isinstance(args.get("__args__"), dict)
    ):
        return None
    return {"name": name, "args": args["__args__"]}


def _supported_checkpoint_models() -> dict[type, str]:
    """Single source of truth for checkpoint-reconstructable NeRD model classes.

    The allowlist is what keeps reconstruction safe under ``weights_only=True``
    loading, so both the name lookup and descriptor reconstruction derive from
    this one mapping. The imports are kept local to preserve the optional Newton
    dependency boundary.
    """
    from physicsnemo.experimental.models.nerd import (
        NeRDEntityTransformer,
        NeRDTransformer,
    )
    from physicsnemo.models.mlp import FullyConnected

    return {
        NeRDTransformer: "NeRDTransformer",
        NeRDEntityTransformer: "NeRDEntityTransformer",
        FullyConnected: "FullyConnected",
    }


def _model_from_descriptor(descriptor: dict[str, Any] | None) -> torch.nn.Module | None:
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        raise ValueError("saved NeRD model descriptor must be a dictionary")
    name, args = descriptor.get("name"), descriptor.get("args")
    by_name = {value: key for key, value in _supported_checkpoint_models().items()}
    if name not in by_name:
        raise ValueError(
            "checkpoint model cannot be reconstructed safely; pass model=... to load it"
        )
    if not isinstance(args, dict):
        raise ValueError(
            "saved NeRD model descriptor has invalid constructor arguments"
        )
    return by_name[name](**args)


def _checkpoint_model_name(model: torch.nn.Module) -> str | None:
    return _supported_checkpoint_models().get(type(model))
