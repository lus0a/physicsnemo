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

"""A small environment wrapper that owns the Newton simulation lifecycle."""

from __future__ import annotations

import argparse
import math
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import torch

from physicsnemo.experimental.integrations.newton.components import (
    NewtonComponents,
    NewtonRollout,
    _components_from_model,
    _components_from_scene,
    _scene_timing,
)
from physicsnemo.experimental.integrations.newton.data import (
    _assign_value,
    _copy_newton_object,
    torch_warp_stream,
)

if TYPE_CHECKING:
    from physicsnemo.experimental.integrations.newton.step_model import NewtonStepModel

# An observation is just a function of the current Newton state. Returning a
# plain Torch tensor keeps the spec next to the physics instead of in a DSL.
Observation = Callable[[Any], torch.Tensor]
BeforeSubstep = Callable[[Any, Any, Any, float, int], None]
StepMode = Literal["replace", "correct"]


class NewtonEnv:
    """Drives reset / step / rollout for a Newton simulation and reads
    observations from each state.

    The environment owns its own state buffers (allocated from ``model.state``),
    so the underlying scene's own trajectory list is left untouched. Build one
    with :meth:`from_example` (the shortest path; wraps a ``newton.examples``
    Example class), :meth:`from_scene` (an existing Newton scene object), or
    :meth:`from_model` (a bare Newton model)."""

    def __init__(
        self,
        components: NewtonComponents,
        *,
        observe: Observation | None = None,
        dt: float = 1.0 / 60.0,
        substeps: int = 1,
        horizon: int | None = None,
        requires_grad: bool = False,
        clear_forces: bool = True,
        collide_on_reset: bool = False,
        collide_each_substep: bool = False,
        step_model: NewtonStepModel | None = None,
        step_mode: StepMode = "replace",
    ) -> None:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        if horizon is not None and horizon <= 0:
            raise ValueError("horizon must be positive when provided")
        if step_mode not in ("replace", "correct"):
            raise ValueError(
                "step_mode must be 'replace' (model instead of solver) or 'correct' (model after solver)"
            )
        self.components = components
        self._observe = observe
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.horizon = int(horizon) if horizon is not None else None
        self.requires_grad = bool(requires_grad)
        self.clear_forces = bool(clear_forces)
        self.collide_on_reset = bool(collide_on_reset)
        self.collide_each_substep = bool(collide_each_substep)
        self.step_model = step_model
        self.step_mode = step_mode
        self.state: Any | None = None
        self.next_state: Any | None = None
        self.control: Any | None = None
        self.contacts: Any | None = None

    @classmethod
    def from_scene(
        cls,
        scene: Any,
        *,
        observe: Observation | None = None,
        dt: float | None = None,
        substeps: int | None = None,
        horizon: int | None = None,
        requires_grad: bool = False,
        clear_forces: bool = True,
        collide_on_reset: bool = False,
        collide_each_substep: bool | None = None,
        step_model: NewtonStepModel | None = None,
        step_mode: StepMode = "replace",
    ) -> NewtonEnv:
        """Wrap an existing Newton scene.

        The scene supplies the model, solver, initial state, controls, contacts,
        and timing. Passing ``substeps`` preserves the scene's frame duration by
        adjusting the substep ``dt`` unless ``dt`` is also provided explicitly.
        Overriding ``substeps`` without ``dt`` requires the scene to expose a
        frame duration to rescale; otherwise pass ``dt`` explicitly.
        When a collision pipeline is available, contacts are refreshed before
        every solver substep by default. Pass ``collide_each_substep=False`` only
        for scenes whose authored contacts are intentionally valid for the whole
        rollout, such as collisions against a fixed plane.

        Parameters
        ----------
        scene : Any
            Newton example or scene exposing ``model`` and ``solver``.
        observe : Observation, optional
            Function mapping a Newton state to a Torch observation.
        dt : float, optional
            Duration of one solver substep, in seconds.
        substeps : int, optional
            Number of solver substeps per environment step.
        horizon : int, optional
            Default number of environment steps for :meth:`rollout`.
        requires_grad : bool, optional
            Whether to allocate state buffers with Warp gradients.
        clear_forces : bool, optional
            Whether to clear transient state forces before every substep.
        collide_on_reset : bool, optional
            Whether to refresh contacts after every reset.
        collide_each_substep : bool, optional
            Whether to refresh contacts before every solver substep. Defaults
            to true when the scene exposes a collision pipeline.
        step_model : NewtonStepModel, optional
            Learned replacement or corrector for the solver.
        step_mode : {"replace", "correct"}, optional
            Use ``"replace"`` to skip the solver or ``"correct"`` to run the
            learned model after it.
        """

        timing = _scene_timing(scene)
        had_dt = "dt" in timing
        original_frame_dt = float(timing.get("dt", 0.0)) * int(
            timing.get("substeps", 1)
        )
        if dt is not None:
            timing["dt"] = dt
        if substeps is not None:
            if substeps <= 0:
                raise ValueError("substeps must be positive")
            timing["substeps"] = substeps
            if dt is None:
                if not had_dt:
                    raise ValueError(
                        "cannot override substeps: scene exposes no frame timing "
                        "to rescale; pass dt explicitly"
                    )
                timing["dt"] = original_frame_dt / substeps
        if horizon is not None:
            timing["horizon"] = horizon
        components = _components_from_scene(scene)
        if collide_each_substep is None:
            collide_each_substep = components.pipeline is not None
        return cls(
            components,
            observe=observe,
            requires_grad=requires_grad,
            clear_forces=clear_forces,
            collide_on_reset=collide_on_reset,
            collide_each_substep=collide_each_substep,
            step_model=step_model,
            step_mode=step_mode,
            **timing,
        )

    @classmethod
    def from_example(
        cls,
        example_cls: type,
        *,
        observe: Observation | None = None,
        device: str | torch.device | None = None,
        substeps: int | None = None,
        horizon: int | None = None,
        verbose: bool = False,
        args: argparse.Namespace | None = None,
        arg_overrides: Mapping[str, Any] | None = None,
        requires_grad: bool = False,
        clear_forces: bool = True,
        collide_on_reset: bool = False,
        collide_each_substep: bool | None = None,
        step_model: NewtonStepModel | None = None,
        step_mode: StepMode = "replace",
    ) -> NewtonEnv:
        """Build a headless Newton example and wrap it in one call.

        This is the shortest path for using a class from ``newton.examples``.
        The example's timestep, solver, initial state, controls, and contacts are
        preserved. ``substeps`` overrides the example while keeping its frame
        duration unchanged.

        Parameters
        ----------
        example_cls : type
            Newton ``Example`` class to construct.
        observe : Observation, optional
            Function mapping a Newton state to a Torch observation.
        device : str or torch.device, optional
            Device on which Newton builds the example. An active distributed
            run uses its rank-local device when omitted.
        substeps : int, optional
            Number of solver substeps per environment step.
        horizon : int, optional
            Default number of environment steps for :meth:`rollout`.
        verbose : bool, optional
            Value forwarded to the Newton example arguments.
        args : argparse.Namespace, optional
            Pre-built Newton example argument namespace.
        arg_overrides : Mapping[str, Any], optional
            Argument values applied before construction.
        requires_grad : bool, optional
            Whether to allocate environment states with Warp gradients.
        clear_forces : bool, optional
            Whether to clear transient state forces before each substep.
        collide_on_reset : bool, optional
            Whether to refresh contacts immediately after reset.
        collide_each_substep : bool, optional
            Whether to refresh contacts before every solver substep. Defaults
            to true when the example exposes a collision pipeline.
        step_model : NewtonStepModel, optional
            Learned step called instead of or after the Newton solver.
        step_mode : {"replace", "correct"}, optional
            Use ``"replace"`` to skip the solver or ``"correct"`` to run the
            learned step after it.
        """
        from physicsnemo.experimental.integrations.newton.scene import (
            load_example_scene,
        )

        scene = load_example_scene(
            example_cls,
            device=device,
            substeps=substeps,
            verbose=verbose,
            args=args,
            arg_overrides=arg_overrides,
        )
        return cls.from_scene(
            scene,
            observe=observe,
            horizon=horizon,
            requires_grad=requires_grad,
            clear_forces=clear_forces,
            collide_on_reset=collide_on_reset,
            collide_each_substep=collide_each_substep,
            step_model=step_model,
            step_mode=step_mode,
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        observe: Observation | None = None,
        solver: Any = None,
        pipeline: Any = None,
        control: Any = None,
        contacts: Any = None,
        collisions: bool = True,
        collision_options: dict[str, Any] | None = None,
        dt: float = 1.0 / 60.0,
        substeps: int = 1,
        horizon: int | None = None,
        requires_grad: bool = False,
        clear_forces: bool = True,
        collide_on_reset: bool = False,
        collide_each_substep: bool | None = None,
        step_model: NewtonStepModel | None = None,
        step_mode: StepMode = "replace",
    ) -> NewtonEnv:
        """Wrap a bare Newton model (defaults to ``SolverSemiImplicit``).

        With ``collisions=True``, a Newton collision pipeline is constructed
        when the model has shapes. Pass ``collisions=False`` for a contact-free
        model, a solver such as MuJoCo that owns collision detection, or a
        learned replacement that does not consume contacts.

        Parameters
        ----------
        model : Any
            Finalized Newton ``Model`` to simulate.
        observe : Observation, optional
            Function mapping a Newton state to a Torch observation.
        solver : Any, optional
            Newton solver used to step the model. Defaults to
            ``SolverSemiImplicit``.
        pipeline : Any, optional
            Explicit Newton collision pipeline. Mutually exclusive with
            ``collision_options`` and invalid when ``collisions=False``.
        control : Any, optional
            Control template copied into each episode's control.
        contacts : Any, optional
            Authored contact buffer used when no pipeline refreshes contacts.
        collisions : bool, optional
            Whether to build an automatic collision pipeline when the model has
            shapes. Set to ``False`` for contact-free or solver-owned collision.
        collision_options : dict[str, Any], optional
            Keyword arguments for the automatic ``CollisionPipeline``. Must not
            include ``requires_grad``; configure that on the environment.
        dt : float, optional
            Duration of one solver substep, in seconds.
        substeps : int, optional
            Number of solver substeps per environment step.
        horizon : int, optional
            Default number of environment steps for :meth:`rollout`.
        requires_grad : bool, optional
            Whether to allocate state buffers with Warp gradients.
        clear_forces : bool, optional
            Whether to clear transient state forces before every substep.
        collide_on_reset : bool, optional
            Whether to refresh contacts after every reset.
        collide_each_substep : bool, optional
            Whether to refresh contacts before every solver substep. Defaults
            to true when a collision pipeline is present.
        step_model : NewtonStepModel, optional
            Learned replacement or corrector for the solver.
        step_mode : {"replace", "correct"}, optional
            Use ``"replace"`` to skip the solver or ``"correct"`` to run the
            learned model after it.
        """

        if not isinstance(collisions, bool):
            raise TypeError("collisions must be a bool")
        if not collisions and (pipeline is not None or collision_options):
            raise ValueError(
                "pipeline and collision_options cannot be provided when collisions=False"
            )
        if pipeline is not None and collision_options:
            raise ValueError(
                "collision_options configure an automatic pipeline and cannot be "
                "combined with an explicit pipeline"
            )
        if collision_options and "requires_grad" in collision_options:
            raise ValueError(
                "requires_grad is env-level config and must not be set inside "
                "collision_options; pass requires_grad= to the factory instead"
            )
        if (
            collisions
            and pipeline is None
            and int(getattr(model, "shape_count", 0)) > 0
        ):
            from physicsnemo.experimental.integrations.newton.dependencies import (
                require_newton,
            )

            options = dict(collision_options or {})
            options.setdefault("requires_grad", requires_grad)
            pipeline = require_newton().CollisionPipeline(model, **options)
        components = _components_from_model(
            model,
            solver=solver,
            pipeline=pipeline,
            control=control,
            contacts=contacts,
        )
        if collide_each_substep is None:
            collide_each_substep = components.pipeline is not None
        return cls(
            components,
            observe=observe,
            dt=dt,
            substeps=substeps,
            horizon=horizon,
            requires_grad=requires_grad,
            clear_forces=clear_forces,
            collide_on_reset=collide_on_reset,
            collide_each_substep=collide_each_substep,
            step_model=step_model,
            step_mode=step_mode,
        )

    @property
    def model(self) -> Any:
        """The Newton model being simulated."""
        return self.components.model

    @property
    def solver(self) -> Any:
        """The Newton solver that steps the model."""
        return self.components.solver

    @property
    def pipeline(self) -> Any | None:
        """The optional Newton collision pipeline (``None`` if unused)."""
        return self.components.pipeline

    @property
    def frame_dt(self) -> float:
        """Simulated time advanced by one :meth:`step`, in seconds."""
        return self.dt * self.substeps

    def reset(self, **fields: Any) -> torch.Tensor:
        """Allocate fresh state/control buffers, assign any ``field=value`` initial
        conditions (e.g. ``reset(particle_qd=v0)``), and return the observation."""

        # Optional per-episode hook: a step model may define ``reset()`` to flush
        # any internal causal history/state (e.g. the NeRD context window). It is
        # duck-typed because it is not part of the NewtonStepModel protocol.
        reset_step_model = getattr(self.step_model, "reset", None)
        if callable(reset_step_model):
            reset_step_model()
        with torch_warp_stream(self.model.device):
            self.state = self.model.state(requires_grad=self.requires_grad)
            self.next_state = self.model.state(requires_grad=self.requires_grad)
            if self.components.initial_state is not None:
                _copy_newton_object(self.state, self.components.initial_state)
                _copy_newton_object(self.next_state, self.components.initial_state)
            self.control = self._new_control()
            self.contacts = (
                self.components.pipeline.contacts()
                if self.components.pipeline is not None
                and (self.collide_on_reset or self.collide_each_substep)
                else self.components.contacts
            )
            for name, value in fields.items():
                _assign_value(getattr(self.state, name), value)
            if self.collide_on_reset:
                self._collide(self.state)
            elif (
                self.components.pipeline is not None
                and not self.collide_each_substep
                and self.contacts is None
            ):
                # Collision-capable scene with neither authored contacts nor any
                # contact refresh: the solver would silently receive
                # ``contacts=None``. Warn rather than collide implicitly so the
                # caller decides between ``collide_on_reset``/``collide_each_substep``
                # and intentional contact-free dynamics.
                warnings.warn(
                    "NewtonEnv has a collision pipeline but no contacts: "
                    "collide_on_reset and collide_each_substep are both False and "
                    "the scene authored no contacts, so the solver will receive "
                    "contacts=None. Enable collide_on_reset/collide_each_substep, "
                    "call collide(), or author contacts if contact-free dynamics "
                    "are not intended.",
                    stacklevel=2,
                )
        return self.observe()

    def observe(self, state: Any | None = None) -> torch.Tensor:
        """Read the observation from ``state`` (defaults to the current state)."""

        if self._observe is None:
            return torch.empty(0, device=str(self.model.device))
        if state is None and self.state is None:
            self.reset()
        return self._observe(self.state if state is None else state)

    def collide(self, state: Any | None = None) -> None:
        """Refresh contacts for ``state`` (defaults to the current state);
        a no-op when the environment has no collision pipeline."""
        if state is None:
            if self.state is None:
                self.reset()
            state = self.state
        with torch_warp_stream(self.model.device):
            self._collide(state)

    def rollout(
        self,
        steps: int | None = None,
        *,
        keep_states: bool = False,
        include_initial: bool = True,
        before_substep: BeforeSubstep | None = None,
    ) -> NewtonRollout:
        """Advance steps and stack observations into ``(time, *observation_shape)``.

        When ``steps`` is omitted, the environment's ``horizon`` is used.
        With ``keep_states=True`` a distinct Newton state is allocated for every
        solver substep so the whole trajectory stays on a Warp tape for adjoints;
        otherwise two buffers are ping-ponged. Returned observations are stable
        snapshots; :meth:`observe` and :meth:`step` still return live views. In
        the two-buffer (``keep_states=False``) path the returned ``final_state``
        aliases an env buffer that the next :meth:`step`/:meth:`rollout`
        overwrites; clone or read its fields before continuing to step."""

        if steps is None:
            if self.horizon is None:
                raise ValueError(
                    "steps is required when the environment has no horizon"
                )
            steps = self.horizon
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if self.state is None:
            self.reset()
        if keep_states:
            return self._rollout_keep_states(steps, include_initial, before_substep)
        # Observations commonly alias the two ping-pong state buffers. Snapshot
        # each step before either buffer is reused.
        observations = [self.observe().clone()] if include_initial else []
        observations.extend(
            self.step(before_substep=before_substep).clone() for _ in range(steps)
        )
        return NewtonRollout(
            observations=_stack(observations, device=str(self.model.device)),
            final_state=self.state,
        )

    def step(self, *, before_substep: BeforeSubstep | None = None) -> torch.Tensor:
        """Advance one step and return its observation.

        ``before_substep`` runs after force clearing and before collision/solver
        execution on every substep. It is the application hook for controls,
        targets, or state forces that must be reapplied after
        ``newton.State.clear_forces``.
        """
        if self.state is None:
            self.reset()
        for substep in range(self.substeps):
            self._step_once(
                self.state,
                self.next_state,
                before_substep=before_substep,
                substep=substep,
            )
            self.state, self.next_state = self.next_state, self.state
        return self.observe()

    def _rollout_keep_states(
        self,
        steps: int,
        include_initial: bool,
        before_substep: BeforeSubstep | None,
    ) -> NewtonRollout:
        history = [
            self.model.state(requires_grad=self.requires_grad)
            for _ in range(steps * self.substeps + 1)
        ]
        _copy_newton_object(history[0], self.state)
        observations = [self.observe(history[0])] if include_initial else []
        index = 0
        for _step in range(steps):
            for substep in range(self.substeps):
                self._step_once(
                    history[index],
                    history[index + 1],
                    before_substep=before_substep,
                    substep=substep,
                )
                index += 1
            observations.append(self.observe(history[index]))
        self.state = history[-1]
        return NewtonRollout(
            observations=_stack(observations, device=str(self.model.device)),
            final_state=history[-1],
            states=history,
        )

    def _step_once(
        self,
        state_in: Any,
        state_out: Any,
        *,
        before_substep: BeforeSubstep | None = None,
        substep: int = 0,
    ) -> None:
        with torch_warp_stream(self.model.device):
            if self.clear_forces and hasattr(state_in, "clear_forces"):
                state_in.clear_forces()
            if before_substep is not None:
                before_substep(state_in, self.control, self.contacts, self.dt, substep)
            if self.collide_each_substep:
                self._collide(state_in)
            # No model -> just the solver. "replace" -> the model is the step.
            # "correct" -> the solver steps, then the model adjusts state_out.
            if self.step_model is None or self.step_mode == "correct":
                self.solver.step(
                    state_in, state_out, self.control, self.contacts, self.dt
                )
            if self.step_model is not None:
                self.step_model(
                    state_in, state_out, self.control, self.contacts, self.dt
                )

    def _collide(self, state: Any) -> None:
        pipeline = self.components.pipeline
        if pipeline is None:
            return
        if self.contacts is None:
            self.contacts = pipeline.contacts()
        pipeline.collide(state, self.contacts)

    def _new_control(self) -> Any:
        """Allocate an episode-local control and copy the configured template."""

        control = self.model.control()
        if self.components.control is not None:
            _copy_newton_object(control, self.components.control)
        return control


def _stack(
    observations: list[torch.Tensor], *, device: str | torch.device
) -> torch.Tensor:
    if not observations:
        return torch.empty(0, device=device)
    return torch.stack(observations, dim=0)
