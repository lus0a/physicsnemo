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

"""Differentiable Newton rollouts via the Warp adjoint (the bridge to PyTorch).

Three entry points share the same Warp tape:

* :func:`differentiable_rollout` rolls out once and returns the trajectory, the
  adjoint, and the loss as detached *data products* (the teacher signal a
  surrogate learns from).
* :func:`optimize_field_in_newton` runs the simulator-in-the-loop optimizer:
  every step rolls the real solver out, backpropagates the physics loss to an
  initial-state field, and takes a Torch optimizer step on that field, feeding
  the Warp gradient buffer to the optimizer zero-copy.
* :func:`optimize_field_in_newton_multistart` fully refines several
  initializations and returns the best real-physics branch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypedDict, cast

import numpy as np
import torch
import warp as wp

from physicsnemo.experimental.integrations.newton.components import (
    DifferentiableRollout,
)
from physicsnemo.experimental.integrations.newton.data import (
    field_to_torch,
    torch_warp_stream,
)
from physicsnemo.experimental.integrations.newton.env import NewtonEnv


class FieldOptimizationRecord(TypedDict):
    """One real-physics evaluation during field optimization."""

    step: int
    real_loss: float
    wall_ms: float
    solver_evals: int


class FieldOptimizationResult(TypedDict):
    """Result returned by :func:`optimize_field_in_newton`."""

    best_params: np.ndarray
    best_loss: float
    initial_loss: float
    optimization_steps: int
    total_ms: float
    solver_evals: int
    history: list[FieldOptimizationRecord]


class MultiStartFieldOptimizationResult(FieldOptimizationResult):
    """Best result plus every branch from multi-start field optimization."""

    best_index: int
    runs: list[FieldOptimizationResult]
    starts: int
    per_start_solver_evals: int
    total_solver_evals: int


def differentiable_rollout(
    env: NewtonEnv,
    *,
    loss_fn: Callable[[list[Any]], Any],
    steps: int | None = None,
    field: str = "particle_qd",
) -> DifferentiableRollout:
    """Roll the environment out under a Warp tape and backpropagate a scalar loss.

    This is the bridge a static dataset cannot give you: it returns the state
    trajectory and the gradient of the loss w.r.t. an initial-state field,
    straight from the differentiable solver and zero-copy into Torch.

    Parameters
    ----------
    env : NewtonEnv
        Environment whose model was finalized with ``requires_grad=True``.
    loss_fn : Callable[[list[Any]], Any]
        Function that receives the kept per-substep states and returns a scalar
        Warp loss array, typically by launching a small Warp kernel on the final
        state.
    steps : int, optional
        Number of frames to roll out. Uses ``env.horizon`` when omitted.
    field : str, optional
        Name of the initial-state field to differentiate against, such as
        ``"particle_qd"`` for an initial velocity.

    Returns
    -------
    DifferentiableRollout
        Observation trajectory, ``d(loss)/d(initial field)`` adjoint, and scalar
        loss. All outputs are detached data products of the Warp tape rather
        than Torch autograd values.
    """

    with torch_warp_stream(env.model.device):
        tape = wp.Tape()
        with tape:
            rollout = env.rollout(steps, keep_states=True)
            states = cast(list[Any], rollout.states)
            loss = loss_fn(states)
        tape.backward(loss)

        initial_field = getattr(states[0], field)
        grad = getattr(initial_field, "grad", None)
        if grad is None:
            raise RuntimeError(
                f"Initial-state field '{field}' has no Warp gradient. Build the model "
                "with requires_grad=True and the env with requires_grad=True."
            )
        result = DifferentiableRollout(
            observations=rollout.observations.detach(),
            adjoint=field_to_torch(grad, clone=True).detach(),
            loss=field_to_torch(loss, clone=True).reshape(()).detach(),
        )
        tape.zero()
    return result


def optimize_field_in_newton(
    env: NewtonEnv,
    *,
    loss_fn: Callable[[list[Any]], Any],
    field: str = "particle_qd",
    initial: Any = None,
    optimization_steps: int = 24,
    steps: int | None = None,
    lr: float = 5.0e-2,
    betas: tuple[float, float] = (0.9, 0.999),
) -> FieldOptimizationResult:
    """Optimize a Newton initial-state field in place through the solver adjoint.

    This is the direct, simulator-in-the-loop counterpart to optimizing a design
    through a learned surrogate (:meth:`BPTTSurrogate.optimize`), and the honest
    baseline it should be compared against: every step runs a fresh
    :func:`differentiable_rollout` of the *real* solver from the current field
    value, takes the adjoint ``d(loss)/d(field)`` it returns, and applies one Torch
    optimizer step. ``differentiable_rollout`` owns its Warp tape and zeroes the
    gradient buffers per call, so the adjoint is clean every step; the loss it
    returns *is* the real Newton objective, so each step's quality is recorded
    without an extra forward pass.

    The optimized variable *is* the Newton field (e.g. an initial ``particle_qd``
    velocity). Use this when the design knob is the raw field itself; if a smaller
    design vector must be projected onto the field, map it outside this
    helper.

    Parameters
    ----------
    env : NewtonEnv
        Environment finalized with ``requires_grad=True``.
    loss_fn : Callable[[list[Any]], Any]
        Scalar Warp loss over the kept rollout states, matching the loss used to
        harvest teacher gradients.
    field : str, optional
        Name of the initial-state field to optimize, such as ``"particle_qd"``
        for an initial velocity.
    initial : Any, optional
        Starting value for ``field``, reshapeable to the field shape. Defaults
        to the environment's current value.
    optimization_steps : int, optional
        Number of optimizer updates. Each update runs one differentiable
        rollout.
    steps : int, optional
        Number of simulated frames per rollout. Uses ``env.horizon`` when
        omitted.
    lr : float, optional
        Adam learning rate.
    betas : tuple[float, float], optional
        Adam beta coefficients.

    Returns
    -------
    FieldOptimizationResult
        Result containing the best field value and loss, initial loss, timing,
        solver evaluation count, and per-step optimization history.
    """

    steps = _resolve_steps(env, steps)

    if optimization_steps < 0:
        raise ValueError("optimization_steps must be non-negative")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if env.state is None:
        env.reset()
    template = field_to_torch(getattr(env.state, field))
    if initial is not None:
        param = field_to_torch(
            initial, dtype=template.dtype, device=template.device
        ).reshape(template.shape)
    else:
        param = template.detach().clone()
    param = (
        param.to(device=template.device, dtype=template.dtype)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    opt = torch.optim.Adam([param], lr=lr, betas=betas)

    history: list[FieldOptimizationRecord] = []
    best_loss = float("inf")
    best_params = param.detach().cpu().numpy().copy()
    initial_loss = float("inf")
    start = time.perf_counter()

    for step in range(optimization_steps + 1):
        env.reset(**{field: param.detach()})
        rollout = differentiable_rollout(env, steps=steps, loss_fn=loss_fn, field=field)
        real_loss = float(rollout.loss)
        if step == 0:
            initial_loss = real_loss
        if real_loss < best_loss:
            best_loss = real_loss
            best_params = param.detach().cpu().numpy().copy()
        history.append(
            {
                "step": step,
                "real_loss": real_loss,
                "wall_ms": 1000.0 * (time.perf_counter() - start),
                "solver_evals": step + 1,
            }
        )
        if step == optimization_steps:
            break
        opt.zero_grad(set_to_none=True)
        param.grad = (
            rollout.adjoint.detach()
            .to(device=param.device, dtype=param.dtype)
            .reshape_as(param)
        )
        opt.step()

    return {
        "best_params": best_params,
        "best_loss": best_loss,
        "initial_loss": initial_loss,
        "optimization_steps": int(optimization_steps),
        "total_ms": 1000.0 * (time.perf_counter() - start),
        "solver_evals": optimization_steps + 1,
        "history": history,
    }


def optimize_field_in_newton_multistart(
    env: NewtonEnv,
    *,
    loss_fn: Callable[[list[Any]], Any],
    initials: Any,
    field: str = "particle_qd",
    optimization_steps: int = 24,
    steps: int | None = None,
    lr: float = 5.0e-2,
    betas: tuple[float, float] = (0.9, 0.999),
) -> MultiStartFieldOptimizationResult:
    """Refine several initial fields in Newton and return the best real result.

    This is the safeguarded counterpart to a learned or heuristic proposal. Add
    the nominal cold start to ``initials`` alongside any proposed starts; because
    every branch receives the same :func:`optimize_field_in_newton` refinement,
    the returned result cannot be worse than refining the cold start alone.

    The branches run serially and ``solver_evals`` reports their total solver
    work. Applications with replicated Newton worlds or separate processes may
    parallelize the independent branches while preserving the same result.

    Parameters
    ----------
    env : NewtonEnv
        Environment finalized with ``requires_grad=True``.
    loss_fn : Callable[[list[Any]], Any]
        Scalar Warp loss over the kept rollout states.
    initials : Any
        Sequence or array with one initial field value per leading row.
    field : str, optional
        Name of the initial-state field to optimize.
    optimization_steps : int, optional
        Number of optimizer updates per start.
    steps : int, optional
        Number of simulated frames per rollout. Uses ``env.horizon`` when
        omitted.
    lr : float, optional
        Adam learning rate shared by every branch.
    betas : tuple[float, float], optional
        Adam beta coefficients shared by every branch.

    Returns
    -------
    MultiStartFieldOptimizationResult
        Best branch result plus every branch, the selected branch index, start
        count, per-start evaluation budget, and aggregate solver work.
    """

    if isinstance(initials, torch.Tensor):
        starts = initials.detach().cpu().numpy()
    else:
        starts = np.asarray(initials)
    if starts.ndim == 0 or starts.shape[0] == 0:
        raise ValueError("initials must contain at least one start")
    if starts.ndim == 1:
        starts = starts[None, :]
    runs = [
        optimize_field_in_newton(
            env,
            loss_fn=loss_fn,
            field=field,
            initial=initial,
            optimization_steps=optimization_steps,
            steps=steps,
            lr=lr,
            betas=betas,
        )
        for initial in starts
    ]
    best_index = int(np.argmin([run["best_loss"] for run in runs]))
    best = dict(runs[best_index])
    best["best_index"] = best_index
    best["runs"] = runs
    best["starts"] = len(runs)
    best["per_start_solver_evals"] = int(runs[0]["solver_evals"])
    # ``total_solver_evals`` is the unambiguous cross-branch sum. The inherited
    # ``solver_evals`` is also overwritten with that total for backward
    # compatibility, so it means per-run budget inside each ``runs`` entry but
    # the serial aggregate at the top level; read ``total_solver_evals`` when an
    # unambiguous aggregate is wanted.
    total_solver_evals = int(sum(run["solver_evals"] for run in runs))
    best["total_solver_evals"] = total_solver_evals
    best["solver_evals"] = total_solver_evals
    return best


def _resolve_steps(env: NewtonEnv, steps: int | None) -> int:
    """Resolve a non-negative rollout length from the call or environment.

    Matches the accepted range of :meth:`NewtonEnv.rollout` (which rejects only
    negative step counts), so a zero-step rollout is permitted through both
    entry points.
    """

    value = env.horizon if steps is None else int(steps)
    if value is None:
        raise ValueError("steps is required when the Newton environment has no horizon")
    if value < 0:
        raise ValueError("steps must be non-negative")
    return value
