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

"""Containers for the Newton objects a :class:`NewtonEnv` drives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class NewtonComponents:
    """The Newton objects that make up one simulation: model, solver, and
    optional collision pipeline, initial state, control, and contact buffers."""

    model: Any
    solver: Any
    pipeline: Any | None = None
    initial_state: Any | None = None
    control: Any | None = None
    contacts: Any | None = None


@dataclass
class NewtonRollout:
    """Trajectory returned by :meth:`NewtonEnv.rollout`.

    ``observations`` contains stable snapshots stacked as ``(time, *obs)``.
    ``final_state`` is always the last Newton state. ``states`` contains the
    complete per-substep state history only when ``keep_states=True`` was passed
    to :meth:`NewtonEnv.rollout`; otherwise it is ``None``.

    Keeping every state is substantially more expensive than the normal
    two-buffer rollout and is mainly intended for Warp adjoints and rendering.
    """

    observations: torch.Tensor
    final_state: Any
    states: list[Any] | None = None


@dataclass
class DifferentiableRollout:
    """Result of :func:`differentiable_rollout`.

    ``observations`` is the observation trajectory, ``adjoint`` is the Warp
    gradient of the loss w.r.t. the requested initial field, and ``loss`` is the
    scalar physics loss. All are detached Torch tensors (data products of the
    Warp tape, not part of any Torch autograd graph)."""

    observations: torch.Tensor
    adjoint: torch.Tensor
    loss: torch.Tensor


def _components_from_model(
    model: Any,
    *,
    solver: Any = None,
    pipeline: Any = None,
    control: Any = None,
    contacts: Any = None,
) -> NewtonComponents:
    """Wrap a Newton model, defaulting to ``SolverSemiImplicit`` when no solver is given."""

    if solver is None:
        from physicsnemo.experimental.integrations.newton.dependencies import (
            require_newton,
        )

        solver = require_newton().solvers.SolverSemiImplicit(model)
    return NewtonComponents(
        model=model,
        solver=solver,
        pipeline=pipeline,
        control=control,
        contacts=contacts,
    )


def _components_from_scene(scene: Any) -> NewtonComponents:
    """Read ``model``, ``solver``, collision pipeline, control, and contacts off a
    Newton example/scene object (e.g. ``newton.examples.diffsim.*.Example``)."""

    model = _require(scene, "model")
    contacts = _first(scene, "contacts")
    pipeline = _first(scene, "pipeline", "collision_pipeline")
    if (
        pipeline is None
        and contacts is not None
        and callable(getattr(model, "collide", None))
    ):
        pipeline = model
    return NewtonComponents(
        model=model,
        solver=_require(scene, "solver"),
        pipeline=pipeline,
        initial_state=_scene_initial_state(scene),
        control=_first(scene, "control"),
        contacts=contacts,
    )


def _require(scene: Any, name: str) -> Any:
    value = getattr(scene, name, None)
    if value is None:
        raise AttributeError(f"Newton scene must expose a non-None '{name}' attribute.")
    return value


def _first(scene: Any, *names: str) -> Any | None:
    for name in names:
        value = getattr(scene, name, None)
        if value is not None:
            return value
    return None


def _scene_initial_state(scene: Any) -> Any | None:
    state = getattr(scene, "state_0", None)
    if state is not None:
        return state
    states = getattr(scene, "states", None)
    return states[0] if states is not None and len(states) > 0 else None


def _scene_timing(scene: Any) -> dict[str, float | int]:
    """Infer timing and rollout horizon from a Newton example scene."""

    timing: dict[str, float | int] = {}
    substeps = getattr(scene, "sim_substeps", None)
    if substeps is not None:
        timing["substeps"] = int(substeps)
    dt = getattr(scene, "sim_dt", None)
    if dt is None:
        frame_dt = getattr(scene, "frame_dt", None)
        if frame_dt is not None and substeps:
            dt = frame_dt / substeps
        else:
            dt = frame_dt
    if dt is not None:
        timing["dt"] = float(dt)
    horizon = getattr(scene, "sim_steps", None)
    if horizon is not None:
        timing["horizon"] = int(horizon)
    return timing
