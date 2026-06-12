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

"""BPTT training and design optimization for learned Newton dynamics.

PhysicsNeMo owns the repetitive infrastructure here so a Newton example only has
to declare its physics. :class:`BPTTSurrogate` combines ordinary one-step
supervision with a free-running rollout loss. During that rollout, each predicted
state is fed back into the model, so later errors backpropagate through the full
chain of learned dynamics steps. This is the BPTT signal that teaches the model
to remain accurate under its own state distribution.

Once fitted, the same differentiable rollout can optimize a control or design
sequence without a simulator in the loop. Newton remains the source of reference
trajectories and the final authority for validating the optimized design; solver
adjoints are not required for surrogate training.

The surrogate network is not special to this module. The per-step model is any
PhysicsNeMo model (or ``torch.nn.Module``) wrapped as a residual step by
``ResidualDynamics``. The default reuses
``physicsnemo.models.mlp.FullyConnected`` rather than reimplementing an MLP,
and a caller can pass any other model with the same input/output width.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.experimental.integrations.newton.distributed import resolve_device

ToInputs = Callable[
    [torch.Tensor, "TeacherBatch"],
    torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
]
TaskLoss = Callable[[torch.Tensor, torch.Tensor, "TeacherBatch"], torch.Tensor]


@dataclass
class TeacherSample:
    """One reference rollout and the parameters or controls that generated it."""

    states: Any
    parameters: Any
    task_data: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TeacherBatch:
    """Stacked reference trajectories and their parameters.

    ``states`` has shape ``(sample, time, state_dim)`` and ``parameters`` has
    shape ``(sample, param_dim)``. ``task_data`` holds any aligned conditioning
    tensors needed by the input or task-loss callbacks.
    """

    states: torch.Tensor
    parameters: torch.Tensor
    task_data: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> TeacherBatch:
        """Move every task tensor in the batch to ``device``."""
        return TeacherBatch(
            self.states.to(device),
            self.parameters.to(device),
            {key: value.to(device) for key, value in self.task_data.items()},
            dict(self.metadata),
        )

    def head(self, n: int) -> TeacherBatch:
        """Return the first ``n`` samples.

        Requests larger than the batch are clamped to its length. ``n`` must be
        positive so a misspelled or zero sample count cannot silently optimize a
        different task.

        The returned tensors are slice views that share storage with this batch,
        so callers should reassign attributes (as the examples do) rather than
        mutate the returned tensors in place; ``.clone()`` the slices if
        independent ownership is required.
        """
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        n = min(n, self.states.shape[0])
        return TeacherBatch(
            self.states[:n],
            self.parameters[:n],
            {key: value[:n] for key, value in self.task_data.items()},
            dict(self.metadata),
        )

    def repeat(self, repeats: int) -> TeacherBatch:
        """Repeat each sample and its task data.

        This is useful when one conditioned task should be optimized from several
        candidate parameter values, as in :meth:`BPTTSurrogate.optimize_multistart`.
        """

        repeats = int(repeats)
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        return TeacherBatch(
            self.states.repeat_interleave(repeats, dim=0),
            self.parameters.repeat_interleave(repeats, dim=0),
            {
                key: value.repeat_interleave(repeats, dim=0)
                for key, value in self.task_data.items()
            },
            dict(self.metadata),
        )


def collect_teacher_batch(
    sample_count: int,
    sample_fn: Callable[[int], TeacherSample],
    *,
    metadata: Mapping[str, Any] | None = None,
    drop_nonfinite: bool = True,
) -> TeacherBatch:
    """Call ``sample_fn(i)`` ``sample_count`` times and stack the teacher samples."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    start = time.perf_counter()
    samples = [sample_fn(index) for index in range(sample_count)]
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / sample_count
    batch = TeacherBatch(
        states=_stack(s.states for s in samples),
        parameters=_stack(s.parameters for s in samples),
        task_data=_stack_task_data(samples),
        metadata={**dict(metadata or {}), "teacher_ms_per_sample": elapsed_ms},
    )
    return _filter_finite(batch) if drop_nonfinite else batch


def _filter_finite(batch: TeacherBatch) -> TeacherBatch:
    """Drop samples with non-finite states, parameters, or task data."""

    mask = torch.isfinite(batch.states).flatten(1).all(1) & torch.isfinite(
        batch.parameters
    ).flatten(1).all(1)
    for value in batch.task_data.values():
        mask &= torch.isfinite(value).reshape(value.shape[0], -1).all(1)
    metadata = dict(batch.metadata)
    metadata["discarded_nonfinite_samples"] = int((~mask).sum())
    if not bool(mask.any()):
        raise ValueError(
            "all teacher samples were non-finite; narrow the sampling distribution "
            "or inspect the Newton rollout"
        )
    return TeacherBatch(
        batch.states[mask],
        batch.parameters[mask],
        {key: value[mask] for key, value in batch.task_data.items()},
        metadata,
    )


@dataclass
class _ResidualMeta(ModelMetaData):
    # Capability flags inherit the conservative ``ModelMetaData`` False defaults.
    # ``ResidualDynamics`` copies them from the wrapped core's meta when that core
    # is itself a PhysicsNeMo Module, so the wrapper never claims more capability
    # (jit/cuda_graphs/amp/auto_grad) than the core it adapts.
    pass


def _residual_meta(core: nn.Module) -> _ResidualMeta:
    """Mirror the wrapped core's capability flags onto the residual wrapper.

    A core may be a graph network or transformer that deliberately disables jit,
    CUDA-graph capture, or AMP; copying its meta keeps StaticCapture/jit tooling
    accurate. Cores that are not PhysicsNeMo Modules expose no meta, so the
    conservative False defaults apply.
    """

    core_meta = getattr(core, "meta", None)
    if not isinstance(core_meta, ModelMetaData):
        return _ResidualMeta()
    return _ResidualMeta(
        jit=core_meta.jit,
        cuda_graphs=core_meta.cuda_graphs,
        amp=core_meta.amp,
        amp_cpu=core_meta.amp_cpu,
        amp_gpu=core_meta.amp_gpu,
        auto_grad=core_meta.auto_grad,
    )


class ResidualDynamics(Module):
    """Adapt any vector-to-vector model into a residual per-step dynamics map.

    The surrogate's per-step contract is ``f(state, inputs) -> next_state``. This
    wrapper supplies the integration glue (concatenate the inputs, add the
    residual) around a ``core`` network that does the actual regression:

        ``next_state = state + core([state, inputs])``

    ``core`` maps a width-``state_dim + input_dim`` input to a width-``state_dim``
    update, and can be any PhysicsNeMo model or ``torch.nn.Module`` with
    that signature (for example ``physicsnemo.models.mlp.FullyConnected``,
    a graph network, or a transformer). The surrogate is therefore not tied to a
    single architecture, and there is no bespoke MLP to maintain here. Use
    :meth:`mlp` for the default fully-connected core."""

    def __init__(self, core: nn.Module, *, state_dim: int, input_dim: int = 0) -> None:
        super().__init__(meta=_residual_meta(core))
        self.core = core
        self.state_dim, self.input_dim = int(state_dim), int(input_dim)
        if self.state_dim <= 0 or self.input_dim < 0:
            raise ValueError("state_dim must be positive and input_dim non-negative")

    @classmethod
    def mlp(
        cls,
        state_dim: int,
        input_dim: int = 0,
        *,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> ResidualDynamics:
        """Default surrogate: a residual step around a PhysicsNeMo
        ``physicsnemo.models.mlp.FullyConnected`` MLP."""
        from physicsnemo.models.mlp import FullyConnected

        if depth <= 0:
            raise ValueError("depth must be positive")
        core = FullyConnected(
            in_features=state_dim + input_dim,
            out_features=state_dim,
            layer_size=hidden_dim,
            num_layers=depth,
            activation_fn="silu",
        )
        return cls(core, state_dim=state_dim, input_dim=input_dim)

    def forward(
        self, state: torch.Tensor, inputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Data-dependent shape checks are skipped under torch.compile (MOD-005);
        # the input_dim/None config branches below are resolved at trace time.
        if not torch.compiler.is_compiling() and state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state must have final dimension {self.state_dim}, "
                f"got {state.shape[-1]}"
            )
        if self.input_dim == 0:
            if inputs is not None and inputs.shape[-1] != 0:
                raise ValueError("this model was created with input_dim=0")
            inputs = state.new_empty((*state.shape[:-1], 0))
        elif inputs is None:
            raise ValueError(
                f"inputs with final dimension {self.input_dim} is required"
            )
        elif not torch.compiler.is_compiling() and inputs.shape != (
            *state.shape[:-1],
            self.input_dim,
        ):
            raise ValueError(
                "inputs must match the state batch dimensions and have "
                f"final dimension {self.input_dim}; got {tuple(inputs.shape)}"
            )
        return state + self.core(torch.cat((state, inputs), dim=-1))


@dataclass(frozen=True)
class _RolloutStats:
    """Standardization statistics for surrogate state and input tensors."""

    state_mean: torch.Tensor
    state_std: torch.Tensor
    input_mean: torch.Tensor
    input_std: torch.Tensor

    def to(self, device: torch.device | str) -> _RolloutStats:
        """Move all standardization statistics to ``device``."""
        return _RolloutStats(
            self.state_mean.to(device),
            self.state_std.to(device),
            self.input_mean.to(device),
            self.input_std.to(device),
        )


def _standardize(
    data: torch.Tensor, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std over all but the feature axis, kept-dim for broadcasting."""

    dims = tuple(range(data.ndim - 1))
    return data.mean(dims, keepdim=True), data.std(
        dims, keepdim=True, unbiased=False
    ).clamp_min(eps)


def _rollout_model(
    model: nn.Module,
    initial: torch.Tensor,
    inputs: torch.Tensor,
    stats: _RolloutStats,
    horizon: int,
) -> torch.Tensor:
    """Roll a per-step dynamics model ``f(state, inputs) -> next_state`` forward in
    normalized space; return denormalized states."""

    state = (initial - stats.state_mean[:, 0, :]) / stats.state_std[:, 0, :]
    states = [state]
    fixed_inputs = inputs.shape[1] == 1
    for step in range(horizon):
        inputs_step = inputs[:, 0, :] if fixed_inputs else inputs[:, step, :]
        inputs_norm = (inputs_step - stats.input_mean[:, 0, :]) / stats.input_std[
            :, 0, :
        ]
        state = model(state, inputs_norm)
        states.append(state)
    return torch.stack(states, dim=1) * stats.state_std + stats.state_mean


class BPTTSurrogate:
    """Train learned dynamics with rollout BPTT, then optimize through them.

    Parameters
    ----------
    state_dim : int
        Surrogate state dimension.
    param_dim : int
        Dimension of the optimized parameters.
    to_inputs : ToInputs
        Function mapping ``(params, batch)`` to
        ``(initial_state[B, state_dim], inputs[B, t, input_dim])``.
    task_loss : TaskLoss
        Function mapping ``(predicted_states[B, T+1, state_dim], params, batch)``
        to a per-sample loss shaped ``[B]``. Required only by :meth:`optimize`
        and :meth:`optimize_multistart`; trajectory training and evaluation do
        not need a task objective.
    task_scale : float, optional
        Positive scale used to normalize ``task_loss`` during parameter
        optimization.
    input_dim : int, optional
        Per-step input feature dimension, or zero when there are no inputs.
    model : torch.nn.Module, optional
        Per-step dynamics model mapping state and inputs to the next state.
        Defaults to :meth:`ResidualDynamics.mlp`, a residual step around a
        PhysicsNeMo ``FullyConnected``. Wrap another PhysicsNeMo model in
        :class:`ResidualDynamics` to replace the architecture.
    hidden_dim : int, optional
        Hidden width of the default MLP. Ignored when ``model`` is supplied.
    depth : int, optional
        Depth of the default MLP. Ignored when ``model`` is supplied.
    device : torch.device or str, optional
        Device used for training and optimization.

    Notes
    -----
    Let ``F_theta`` be the learned one-step model and ``u_t`` the input at time
    ``t``. :meth:`fit` combines teacher-forced local supervision,

    ``F_theta(s_t, u_t) ~= s_{t+1}``,

    with a free-running rollout ``s_hat[t+1] = F_theta(s_hat[t], u_t)``. The
    rollout error at a later time differentiates through every earlier predicted
    state. Its parameter gradient therefore contains products of learned state
    Jacobians, which is precisely backpropagation through time. No task loss or
    Newton adjoint is minimized while fitting the dynamics.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        param_dim: int,
        to_inputs: ToInputs,
        task_loss: TaskLoss | None = None,
        task_scale: float = 1.0,
        input_dim: int = 0,
        model: nn.Module | None = None,
        hidden_dim: int = 128,
        depth: int = 3,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.to_inputs = to_inputs
        self.task_loss = task_loss
        self.state_dim = int(state_dim)
        self.param_dim = int(param_dim)
        self.input_dim = int(input_dim)
        self.task_scale = float(task_scale)
        if self.state_dim <= 0 or self.param_dim <= 0 or self.input_dim < 0:
            raise ValueError("state_dim and param_dim must be positive; input_dim >= 0")
        if not math.isfinite(self.task_scale) or self.task_scale <= 0.0:
            raise ValueError("task_scale must be finite and positive")
        if model is None:
            model = ResidualDynamics.mlp(
                state_dim, input_dim, hidden_dim=hidden_dim, depth=depth
            )
        self.model = model.to(self.device)
        self.stats: _RolloutStats | None = None
        self.param_mean: torch.Tensor | None = None
        self.param_std: torch.Tensor | None = None
        self.horizon = 0
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        batch: TeacherBatch,
        *,
        epochs: int,
        lr: float = 2.0e-3,
        weight_decay: float = 1.0e-5,
        rollout_weight: float = 0.05,
        rollout_warmup_epochs: int = 100,
        grad_clip: float = 10.0,
    ) -> dict[str, float]:
        """Train from reference trajectories with one-step and rollout losses.

        The one-step term evaluates every transition from the reference state.
        The rollout term starts from the reference initial state, feeds each
        prediction back into the next model step, and compares the resulting
        free-running trajectory with the reference. ``rollout_weight`` controls
        that BPTT term. Its weight ramps linearly over
        ``rollout_warmup_epochs`` so early optimization first learns a useful
        local transition map.
        """

        batch = batch.to(self.device)
        self._validate_batch(batch)
        states, params = batch.states.float(), batch.parameters.float()
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if (
            not math.isfinite(lr)
            or not math.isfinite(grad_clip)
            or lr <= 0.0
            or grad_clip <= 0.0
        ):
            raise ValueError("lr and grad_clip must be finite and positive")
        if (
            not math.isfinite(weight_decay)
            or not math.isfinite(rollout_weight)
            or weight_decay < 0.0
            or rollout_weight < 0.0
            or rollout_warmup_epochs < 0
        ):
            raise ValueError(
                "weight_decay, rollout_weight, and rollout_warmup_epochs "
                "must be finite and non-negative"
            )
        horizon = states.shape[1] - 1
        with torch.no_grad():
            initial, inputs = self._inputs(params, batch)
            stats = _RolloutStats(*_standardize(states), *_input_stats(inputs, horizon))
        # Commit fitted state only after every input/statistics computation has
        # succeeded, so a failed re-fit cannot leave a half-updated surrogate
        # (new horizon/param stats with stale rollout stats and weights).
        self.horizon = horizon
        self.param_mean = params.mean(0, keepdim=True)
        self.param_std = params.std(0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        self.stats = stats
        state_mean, state_std = self.stats.state_mean, self.stats.state_std
        normalized_states = (states - state_mean) / state_std
        normalized_inputs = _normalize_inputs(inputs, self.stats, self.horizon)

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.model.train()
        self.history = []
        for epoch in range(epochs):
            one_step_prediction = _one_step_model(
                self.model,
                normalized_states[:, :-1],
                normalized_inputs,
            )
            one_step_loss = F.smooth_l1_loss(
                one_step_prediction,
                normalized_states[:, 1:],
            )
            current_rollout_weight = (
                rollout_weight
                if rollout_warmup_epochs == 0
                else rollout_weight
                * min(1.0, float(epoch + 1) / float(rollout_warmup_epochs))
            )
            if current_rollout_weight > 0.0:
                prediction = _rollout_model(
                    self.model,
                    initial,
                    inputs,
                    self.stats,
                    self.horizon,
                )
                rollout_loss = F.smooth_l1_loss(
                    (prediction[:, 1:] - state_mean) / state_std,
                    normalized_states[:, 1:],
                )
            else:
                rollout_loss = one_step_loss.new_zeros(())
            loss = one_step_loss + current_rollout_weight * rollout_loss
            loss_value = float(loss.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            opt.step()
            self.history.append(
                {
                    "epoch": float(epoch),
                    "loss": loss_value,
                    "one_step_loss": float(one_step_loss.detach()),
                    "rollout_loss": float(rollout_loss.detach()),
                    "rollout_weight": current_rollout_weight,
                }
            )
        self.model.eval()
        return {
            "samples": int(states.shape[0]),
            "horizon": int(self.horizon),
            "task_scale": self.task_scale,
            "epochs": int(epochs),
            "rollout_weight": float(rollout_weight),
            "rollout_warmup_epochs": int(rollout_warmup_epochs),
            "teacher_ms_per_sample": float(
                batch.metadata.get("teacher_ms_per_sample", 0.0)
            ),
            "discarded_nonfinite_samples": int(
                batch.metadata.get("discarded_nonfinite_samples", 0)
            ),
            **{f"train_{k}": v for k, v in self._diagnostics(batch).items()},
        }

    def evaluate(self, batch: TeacherBatch) -> dict[str, float]:
        """Return held-out one-step, free-running, and terminal state RMSE."""

        self._require_fitted()
        batch = batch.to(self.device)
        self.model.eval()
        self._validate_batch(batch, fitted_horizon=True)
        return self._diagnostics(batch)

    def rollout(self, parameters: Any, batch: TeacherBatch) -> torch.Tensor:
        """Roll the fitted surrogate for one parameter row per task in ``batch``."""

        stats, _, _ = self._require_fitted()
        batch = batch.to(self.device)
        self._validate_batch(batch, fitted_horizon=True)
        parameters = torch.as_tensor(
            parameters, dtype=torch.float32, device=self.device
        ).reshape(-1, self.param_dim)
        if parameters.shape[0] != batch.states.shape[0]:
            raise ValueError(
                "parameters row count must match the number of batch samples"
            )
        self.model.eval()
        initial, inputs = self._inputs(parameters, batch)
        return _rollout_model(self.model, initial, inputs, stats, self.horizon)

    def optimize(
        self,
        batch: TeacherBatch,
        *,
        samples: int = 1,
        initial_params: Any | None = None,
        steps: int = 8,
        lr: float = 5.0e-2,
        reg: float = 1.0e-2,
        z_clip: float = 3.0,
    ) -> dict[str, Any]:
        """Optimize parameters through the frozen surrogate (no simulator in the loop).

        Parameters
        ----------
        batch : TeacherBatch
            Tasks and inputs used by ``to_inputs`` and ``task_loss``.
        samples : int, optional
            Number of leading tasks from ``batch`` to optimize, clamped to the
            batch size. Ignored when ``initial_params`` is provided.
        initial_params : Any, optional
            Initial parameters shaped ``(sample, param_dim)``. The row count
            must match ``batch``. Starts outside ``z_clip`` are projected onto
            the normalized optimization bounds before evaluation.
        steps : int, optional
            Number of surrogate-gradient optimizer steps.
        lr : float, optional
            Adam learning rate.
        reg : float, optional
            Weight of the normalized-parameter L2 regularizer.
        z_clip : float, optional
            Maximum absolute normalized parameter value.

        Returns
        -------
        dict[str, Any]
            Best parameters and task losses for every optimized task, their mean
            losses, timing, and optimization history.
        """

        stats, mean, std = self._require_fitted()
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if (
            not math.isfinite(lr)
            or not math.isfinite(z_clip)
            or lr <= 0.0
            or z_clip <= 0.0
        ):
            raise ValueError("lr and z_clip must be finite and positive")
        if not math.isfinite(reg) or reg < 0.0:
            raise ValueError("reg must be finite and non-negative")
        if initial_params is None:
            batch = batch.head(samples).to(self.device)
            start_params = batch.parameters.float()
        else:
            start_params = torch.as_tensor(initial_params, dtype=torch.float32).reshape(
                -1, self.param_dim
            )
            if start_params.shape[0] != batch.states.shape[0]:
                raise ValueError(
                    "initial_params row count must match the number of batch samples"
                )
            batch = batch.to(self.device)
            start_params = start_params.to(self.device)
        self._validate_batch(batch, fitted_horizon=True)
        if not bool(torch.isfinite(start_params).all()):
            raise ValueError("initial parameters must contain only finite values")
        self.model.eval()
        mean, std = mean.to(self.device), std.to(self.device)
        z = ((start_params - mean) / std).detach().clone()
        z.clamp_(-z_clip, z_clip).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)

        def objective() -> tuple[torch.Tensor, torch.Tensor]:
            p = mean + z * std
            initial, inputs = self._inputs(p, batch)
            pred = _rollout_model(self.model, initial, inputs, stats, self.horizon)
            tasks = self._task_losses(pred, p, batch)
            regularization = (z * z).reshape(z.shape[0], -1).mean(-1)
            return (
                (tasks / self.task_scale + reg * regularization).mean(),
                tasks,
            )  # reg keeps z in distribution

        def params_of() -> torch.Tensor:
            return (mean + z * std).detach().clone()

        self._sync()
        start = time.perf_counter()
        obj, task = objective()
        # Select the best parameters by the (surrogate-predicted) task loss; the
        # L2 term only steers the search and should not pick the reported point.
        initial_values = params_of()
        initial_tasks = task.detach().clone()
        best_tasks, best_params = initial_tasks.clone(), initial_values.clone()
        history = [
            {
                "step": 0,
                "objective": float(obj.detach()),
                "task_loss": float(initial_tasks.mean()),
                "min_task_loss": float(initial_tasks.min()),
            }
        ]
        for step in range(1, steps + 1):
            # ``obj`` here is the live-graph objective from before the loop on the
            # first step, then the post-step objective carried over from the
            # previous iteration -- one rollout per step rather than two.
            z.grad = torch.autograd.grad(obj, z)[0]
            opt.step()
            with torch.no_grad():
                z.clamp_(-z_clip, z_clip)
            obj, task = objective()
            task_values = task.detach()
            history.append(
                {
                    "step": step,
                    "objective": float(obj.detach()),
                    "task_loss": float(task_values.mean()),
                    "min_task_loss": float(task_values.min()),
                }
            )
            # Treat a NaN best as infinitely bad so a non-finite initial loss
            # cannot lock out every later (finite) improvement.
            improved = task_values < torch.nan_to_num(best_tasks, nan=torch.inf)
            if bool(improved.any()):
                best_tasks[improved] = task_values[improved]
                best_params[improved] = params_of()[improved]
        self._sync()
        opt_ms = 1000.0 * (time.perf_counter() - start)
        return {
            "initial_params": initial_values.detach(),
            "best_params": best_params.detach(),
            "initial_task_losses": initial_tasks.detach(),
            "best_task_losses": best_tasks.detach(),
            "initial_task_loss": float(initial_tasks.mean()),
            "best_task_loss": float(best_tasks.mean()),
            "steps": int(steps),
            "opt_ms": opt_ms,
            "samples": int(batch.states.shape[0]),
            "history": history,
        }

    def optimize_multistart(
        self,
        batch: TeacherBatch,
        *,
        starts: int = 32,
        initial_params: Any | None = None,
        steps: int = 8,
        lr: float = 5.0e-2,
        reg: float = 1.0e-2,
        z_clip: float = 3.0,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Optimize one conditioned task from many cheap surrogate starts.

        By default, a scrambled Sobol sequence draws diverse normalized
        parameters from the learned design distribution. Callers with a
        problem-specific feasible distribution may instead supply every start
        through ``initial_params``. All starts are optimized together in one
        batched surrogate call, and the lowest surrogate-predicted task loss is
        returned as ``best_params``. No Newton solver rollout is used.

        Parameters
        ----------
        batch : TeacherBatch
            Single task and its inputs.
        starts : int, optional
            Number of candidate starts. The task's supplied parameters are
            projected to ``z_clip`` and used as the first start when
            ``initial_params`` is omitted.
        initial_params : Any, optional
            Problem-specific candidate starts shaped ``(starts, param_dim)``.
            Supplying these bypasses Sobol sampling. Starts outside ``z_clip``
            are projected onto the normalized optimization bounds.
        steps : int, optional
            Number of surrogate-gradient optimizer steps per candidate.
        lr : float, optional
            Adam learning rate.
        reg : float, optional
            Weight of the normalized-parameter L2 regularizer.
        z_clip : float, optional
            Maximum absolute normalized parameter value.
        seed : int, optional
            Seed used to scramble the Sobol sequence.

        Returns
        -------
        dict[str, Any]
            Optimization plan compatible with :meth:`validate_in_newton`.
            ``candidate_best_params`` and ``candidate_best_task_losses`` retain
            every candidate for optional downstream refinement.
        """

        starts = int(starts)
        _, mean, std = self._require_fitted()
        if starts <= 0:
            raise ValueError("starts must be positive")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if (
            not math.isfinite(lr)
            or not math.isfinite(z_clip)
            or lr <= 0.0
            or z_clip <= 0.0
        ):
            raise ValueError("lr and z_clip must be finite and positive")
        if not math.isfinite(reg) or reg < 0.0:
            raise ValueError("reg must be finite and non-negative")
        if batch.states.shape[0] != 1:
            raise ValueError("optimize_multistart expects a batch containing one task")
        batch = batch.to(self.device)
        self._validate_batch(batch, fitted_horizon=True)

        mean, std = mean.to(self.device), std.to(self.device)
        if initial_params is None:
            engine = torch.quasirandom.SobolEngine(
                dimension=self.param_dim, scramble=True, seed=int(seed)
            )
            unit = engine.draw(starts).to(self.device).clamp_(1.0e-6, 1.0 - 1.0e-6)
            z = (torch.erfinv(2.0 * unit - 1.0) * math.sqrt(2.0)).clamp_(
                -z_clip, z_clip
            )
            z[0] = (batch.parameters[0] - mean[0]) / std[0]
            z.clamp_(-z_clip, z_clip)
            candidate_initial_params = mean + z * std
        else:
            candidate_initial_params = torch.as_tensor(
                initial_params, dtype=torch.float32, device=self.device
            ).reshape(-1, self.param_dim)
            if candidate_initial_params.shape[0] != starts:
                raise ValueError(
                    "initial_params must contain exactly "
                    f"{starts} starts, got {candidate_initial_params.shape[0]}"
                )
            if not bool(torch.isfinite(candidate_initial_params).all()):
                raise ValueError("initial_params must contain only finite values")

        candidates = self.optimize(
            batch.repeat(starts),
            initial_params=candidate_initial_params,
            steps=steps,
            lr=lr,
            reg=reg,
            z_clip=z_clip,
        )
        candidate_losses = candidates["best_task_losses"]
        finite = torch.isfinite(candidate_losses)
        if not bool(finite.any()):
            raise RuntimeError(
                "all multistart candidates produced non-finite task losses"
            )
        ranked_losses = torch.where(
            finite, candidate_losses, torch.full_like(candidate_losses, torch.inf)
        )
        best_index = int(torch.argmin(ranked_losses))
        history = [
            {
                **record,
                "mean_task_loss": record["task_loss"],
                "task_loss": record["min_task_loss"],
            }
            for record in candidates["history"]
        ]
        return {
            "initial_params": candidates["initial_params"][best_index : best_index + 1],
            "best_params": candidates["best_params"][best_index : best_index + 1],
            "initial_task_losses": candidates["initial_task_losses"][
                best_index : best_index + 1
            ],
            "best_task_losses": candidate_losses[best_index : best_index + 1],
            "initial_task_loss": float(candidates["initial_task_losses"][best_index]),
            "best_task_loss": float(candidate_losses[best_index]),
            "steps": int(steps),
            "opt_ms": candidates["opt_ms"],
            "samples": 1,
            "starts": starts,
            "best_index": best_index,
            "candidate_initial_params": candidates["initial_params"],
            "candidate_best_params": candidates["best_params"],
            "candidate_initial_task_losses": candidates["initial_task_losses"],
            "candidate_best_task_losses": candidate_losses,
            "history": history,
        }

    def validate_in_newton(
        self, plan: Mapping[str, Any], newton_loss: Callable[[torch.Tensor], float]
    ) -> dict[str, Any]:
        """Re-run the real Newton physics loss on the initial vs surrogate-chosen params."""

        initial_values = [
            float(newton_loss(row)) for row in _rows(plan["initial_params"])
        ]
        optimized_values = [
            float(newton_loss(row)) for row in _rows(plan["best_params"])
        ]
        initial = sum(initial_values) / len(initial_values)
        optimized = sum(optimized_values) / len(optimized_values)
        return {
            "newton_initial_loss": initial,
            "newton_optimized_loss": optimized,
            "newton_loss_delta": initial - optimized,
            "newton_improved": optimized < initial,
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize the fitted surrogate for checkpointing.

        The wrapped ``self.model`` is an ``nn.Module``, but the fitted
        normalization/scaling state (``stats``, ``param_mean``/``param_std``,
        ``task_scale``, ``horizon``) lives outside it, so saving only
        ``self.model.state_dict()`` would silently drop the statistics needed for
        correct rollouts and optimization. This bundles both. Restore with
        :meth:`load_state_dict` into a ``BPTTSurrogate`` constructed with the same
        dimensions, model architecture, and callbacks (the ``to_inputs`` and
        ``task_loss`` callbacks are not serialized)."""

        self._require_fitted()
        return {
            "model": self.model.state_dict(),
            "stats": {
                "state_mean": self.stats.state_mean,
                "state_std": self.stats.state_std,
                "input_mean": self.stats.input_mean,
                "input_std": self.stats.input_std,
            },
            "param_mean": self.param_mean,
            "param_std": self.param_std,
            "task_scale": float(self.task_scale),
            "horizon": int(self.horizon),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a surrogate saved by :meth:`state_dict`.

        Load into a ``BPTTSurrogate`` built with the same ``state_dim``,
        ``param_dim``, ``input_dim``, model architecture, and callbacks used when
        the checkpoint was produced."""

        self.model.load_state_dict(state["model"])
        self.model = self.model.to(self.device)
        stats = state["stats"]
        self.stats = _RolloutStats(
            stats["state_mean"],
            stats["state_std"],
            stats["input_mean"],
            stats["input_std"],
        ).to(self.device)
        self.param_mean = state["param_mean"].to(self.device)
        self.param_std = state["param_std"].to(self.device)
        self.task_scale = float(state["task_scale"])
        self.horizon = int(state["horizon"])

    def _diagnostics(self, batch: TeacherBatch) -> dict[str, float]:
        self.model.eval()
        states = batch.states.float()
        params = batch.parameters.float()
        horizon = states.shape[1] - 1
        with torch.no_grad():
            initial, inputs = self._inputs(params, batch)
            normalized_states = (states - self.stats.state_mean) / self.stats.state_std
            normalized_inputs = _normalize_inputs(inputs, self.stats, horizon)
            one_step_normalized = _one_step_model(
                self.model,
                normalized_states[:, :-1],
                normalized_inputs,
            )
            one_step = (
                one_step_normalized * self.stats.state_std + self.stats.state_mean
            )
            rollout = _rollout_model(self.model, initial, inputs, self.stats, horizon)
        return {
            "one_step_rmse": float(
                torch.sqrt(torch.mean((one_step - states[:, 1:]) ** 2))
            ),
            "rollout_rmse": float(
                torch.sqrt(torch.mean((rollout[:, 1:] - states[:, 1:]) ** 2))
            ),
            "terminal_rmse": float(
                torch.sqrt(torch.mean((rollout[:, -1] - states[:, -1]) ** 2))
            ),
        }

    def _validate_batch(
        self, batch: TeacherBatch, *, fitted_horizon: bool = False
    ) -> None:
        if batch.states.shape[0] == 0:
            raise ValueError("reference batch must contain at least one sample")
        if batch.states.ndim != 3 or batch.states.shape[-1] != self.state_dim:
            raise ValueError(
                "batch.states must have shape "
                f"[samples, time, {self.state_dim}], got {tuple(batch.states.shape)}"
            )
        if batch.states.shape[1] < 2:
            raise ValueError("batch.states must contain an initial and next state")
        expected_parameters = (batch.states.shape[0], self.param_dim)
        if tuple(batch.parameters.shape) != expected_parameters:
            raise ValueError(
                f"batch.parameters must have shape {expected_parameters}, "
                f"got {tuple(batch.parameters.shape)}"
            )
        if not bool(torch.isfinite(batch.states).all()) or not bool(
            torch.isfinite(batch.parameters).all()
        ):
            raise ValueError("reference states and parameters must be finite")
        for key, value in batch.task_data.items():
            if value.ndim == 0 or value.shape[0] != batch.states.shape[0]:
                raise ValueError(
                    f"batch.task_data[{key!r}] must have one row per sample"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"batch.task_data[{key!r}] must be finite")
        if fitted_horizon and batch.states.shape[1] - 1 != self.horizon:
            raise ValueError(
                f"batch horizon must match the fitted horizon ({self.horizon})"
            )

    def _inputs(
        self, params: torch.Tensor, batch: TeacherBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call ``to_inputs`` and normalize it to ``(initial_state, inputs)``.

        ``to_inputs`` may return just the initial state (no inputs) or a
        ``(initial, inputs)`` pair with ``inputs=None`` -- this fills in an empty
        per-step inputs tensor so the no-inputs case stays clean."""

        result = self.to_inputs(params, batch)
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("to_inputs must return state or (state, inputs)")
            initial, inputs = result
        else:
            initial, inputs = result, None
        if not isinstance(initial, torch.Tensor):
            raise TypeError("to_inputs must return Torch tensors")
        expected_initial = (params.shape[0], self.state_dim)
        if tuple(initial.shape) != expected_initial:
            raise ValueError(
                f"to_inputs state must have shape {expected_initial}, "
                f"got {tuple(initial.shape)}"
            )
        if inputs is None:
            inputs = initial.new_empty((initial.shape[0], 1, 0))
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError("to_inputs inputs must be a Torch tensor or None")
        if (
            inputs.ndim != 3
            or inputs.shape[0] != initial.shape[0]
            or inputs.shape[-1] != self.input_dim
        ):
            raise ValueError(
                "to_inputs inputs must have shape "
                f"[samples, time, {self.input_dim}], got {tuple(inputs.shape)}"
            )
        expected_time = batch.states.shape[1] - 1
        if inputs.shape[1] not in (1, expected_time):
            raise ValueError(
                "to_inputs inputs time dimension must be 1 or match the "
                f"rollout horizon ({expected_time}), got {inputs.shape[1]}"
            )
        return initial, inputs

    def _task_losses(
        self, states: torch.Tensor, params: torch.Tensor, batch: TeacherBatch
    ) -> torch.Tensor:
        if self.task_loss is None:
            raise RuntimeError(
                "task_loss is required for surrogate parameter optimization"
            )
        losses = self.task_loss(states, params, batch)
        expected = (params.shape[0],)
        if not isinstance(losses, torch.Tensor) or tuple(losses.shape) != expected:
            shape = tuple(losses.shape) if isinstance(losses, torch.Tensor) else None
            raise ValueError(
                f"task_loss must return one value per sample with shape {expected}; "
                f"got {shape}"
            )
        return losses

    def _require_fitted(
        self,
    ) -> tuple[_RolloutStats, torch.Tensor, torch.Tensor]:
        if self.stats is None or self.param_mean is None or self.param_std is None:
            raise RuntimeError("fit the surrogate before evaluation or optimization")
        return self.stats, self.param_mean, self.param_std

    def _sync(self) -> None:
        # Uses the device-agnostic torch.accelerator API on purpose so timing
        # brackets stay accurate on CUDA, XPU, and MPS alike (rather than the
        # CUDA-only torch.cuda.synchronize used elsewhere in the package).
        if self.device.type != "cpu":
            torch.accelerator.synchronize(self.device)


def _input_stats(
    inputs: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.shape[-1] == 0:
        empty = torch.zeros((1, 1, 0), dtype=inputs.dtype, device=inputs.device)
        return empty, empty + 1
    return _standardize(_expand_inputs(inputs, horizon))


def _expand_inputs(inputs: torch.Tensor, horizon: int) -> torch.Tensor:
    """Expand a time-invariant input to one input per rollout step."""

    return inputs.expand(-1, horizon, -1) if inputs.shape[1] == 1 else inputs


def _normalize_inputs(
    inputs: torch.Tensor, stats: _RolloutStats, horizon: int
) -> torch.Tensor:
    """Expand and standardize the inputs for vectorized one-step training."""

    expanded = _expand_inputs(inputs, horizon)
    return (expanded - stats.input_mean) / stats.input_std


def _one_step_model(
    model: nn.Module, states: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    """Evaluate every teacher-forced transition as one model batch."""

    batch, horizon, state_dim = states.shape
    input_dim = inputs.shape[-1]
    prediction = model(
        states.reshape(batch * horizon, state_dim),
        inputs.reshape(batch * horizon, input_dim),
    )
    return prediction.reshape(batch, horizon, state_dim)


def _stack(values) -> torch.Tensor:
    return torch.stack([_tensor(v) for v in values], dim=0)


def _stack_task_data(samples: list[TeacherSample]) -> dict[str, torch.Tensor]:
    keys = set(samples[0].task_data)
    if any(set(sample.task_data) != keys for sample in samples[1:]):
        raise ValueError("every TeacherSample must provide the same task_data keys")
    return {key: _stack(sample.task_data[key] for sample in samples) for key in keys}


def _tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.detach().to("cpu", torch.float32)


def _rows(params: Any) -> torch.Tensor:
    tensor = torch.as_tensor(params, dtype=torch.float32)
    return tensor if tensor.ndim > 1 else tensor.reshape(1, -1)
