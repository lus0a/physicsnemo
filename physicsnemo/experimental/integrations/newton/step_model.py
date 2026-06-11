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

"""Run a learned model in place of (or as a corrector to) the Newton solver step.

The deployment goal of a surrogate is to stand in for the solver at inference.
A :class:`NewtonStepModel` is a callable with the same shape as ``solver.step``:
read ``state_in``, write the predicted next state into ``state_out``. Hand one to
:class:`NewtonEnv` and it is called every substep instead of the solver
(``step_mode="replace"``) or right after it as a residual corrector
(``step_mode="correct"``)::

    env = NewtonEnv.from_model(
        model, observe=observe_fn, step_model=learned_step, step_mode="replace"
    )
    rollout = env.rollout(steps=120)   # the learned step drives the simulation

:func:`write_state_fields` is the one-liner a learned step uses to land its
prediction in the live state through the Warp/Torch bridge.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from physicsnemo.experimental.integrations.newton.data import _assign_value


class NewtonStepModel(Protocol):
    """A learned stand-in for one Newton solver step.

    Implement ``__call__`` with the same signature as ``solver.step``: read
    ``state_in`` (and ``control``/``contacts`` if needed) and write the next state
    into ``state_out`` (typically with :func:`write_state_fields`). It is a structural
    protocol, so any matching callable works -- no base class to inherit."""

    def __call__(
        self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float
    ) -> None: ...


def write_state_fields(state: Any, **fields: torch.Tensor) -> None:
    """Write fields into a Newton state (or control) object in place.

    This is the generic, stream-guarded counterpart to :func:`field_to_torch`:
    it accepts any Newton object, so it also writes control inputs, for example
    ``write_state_fields(env.control, joint_f=tau)`` -- prefer it over a raw
    ``.copy_()`` so writes go through the Warp/Torch stream guard. ``state`` names
    the target only for the common case.

    ``write_state_fields(state_out, body_q=q, body_qd=qd)`` copies each tensor into the
    matching Warp array. A device or dtype conversion may allocate a temporary
    tensor, but never requires a host round trip. Writes mutate simulation state
    under ``no_grad``; this deployment helper is not a functional Torch-autograd
    operator. Remember Newton's layouts: ``body_q`` is
    ``[position, quaternion]`` and ``body_qd`` is ``[linear, angular]``.

    Targeting a physics family this simulation does not have raises a
    field-named error: an unknown attribute raises :class:`AttributeError`, and a
    present-but-unallocated family (a ``None`` field, such as ``body_q`` in a
    particle-only sim) raises :class:`ValueError`."""
    for name, value in fields.items():
        if not hasattr(state, name):
            raise AttributeError(f"state has no field {name!r}")
        target = getattr(state, name)
        if target is None:
            raise ValueError(
                f"state field {name!r} is not allocated; this physics family is "
                "not present in the simulation"
            )
        _assign_value(target, value)
