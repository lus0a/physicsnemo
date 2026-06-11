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

"""Readable, zero-copy views over Newton state and model fields.

Newton stores simulation state as Warp arrays with names like ``particle_q``,
``body_qd``, or ``joint_q`` and packs poses and spatial velocities into compact
7- and 6-vectors. That is exactly right for the solver, but it makes user code
read like array bookkeeping::

    q = wp.to_torch(state.particle_q)          # which columns are position?
    v = wp.to_torch(state.body_qd)[:, :3]      # ...and which are linear velocity?

These views give the same data obvious names, as zero-copy Torch tensors, with
setters that write straight back into the live Warp buffers:

    particles(state).positions          # (N, 3) Torch view, no copy
    bodies(state).linear_velocities     # (N, 3), the first half of body_qd

The goal is clarity over generality: one obvious accessor per state family
(particles, rigid bodies, and articulation joints). Every property is a live
view. Reading does not copy data, and setters write into the live simulation
buffers.

The velocity/coordinate vocabulary differs across families on purpose: joints
expose generalized ``coordinates``/``velocities`` because ``joint_q`` mixes
linear and angular degrees of freedom, while bodies deliberately expose split
``linear_velocities``/``angular_velocities``/``spatial_velocities`` (and no plain
``velocities``) to force the caller to be explicit about linear vs. spatial.
"""

from __future__ import annotations

from typing import Any

import torch

from physicsnemo.experimental.integrations.newton.data import (
    _assign_value,
    field_to_torch,
)


class ParticleView:
    """Positions and velocities of a particle/continuum state.

    Covers particle systems, soft bodies, MPM, and fluids. Wraps
    ``state.particle_q`` / ``state.particle_qd``.

    Setters mutate simulation state in place under ``no_grad`` and are not
    connected to Torch autograd or the Warp tape; for gradient-based use go
    through ``differentiable_rollout`` / ``env.reset(field=...)`` instead.
    """

    def __init__(self, state: Any) -> None:
        self._state = state

    @property
    def positions(self) -> torch.Tensor:
        """Particle positions ``[m]``, shape ``(num_particles, 3)`` (zero-copy)."""
        return field_to_torch(self._state.particle_q)

    @positions.setter
    def positions(self, value: Any) -> None:
        """Write particle positions back into ``state.particle_q``."""
        _assign_value(self._state.particle_q, value)

    @property
    def velocities(self) -> torch.Tensor:
        """Particle velocities ``[m/s]``, shape ``(num_particles, 3)`` (zero-copy)."""
        return field_to_torch(self._state.particle_qd)

    @velocities.setter
    def velocities(self, value: Any) -> None:
        """Write particle velocities back into ``state.particle_qd``."""
        _assign_value(self._state.particle_qd, value)

    def __len__(self) -> int:
        return int(self.positions.shape[0])


class BodyView:
    """Pose and spatial velocity of rigid bodies (and rod/cable segments, which
    Newton models as rigid bodies). Wraps ``state.body_q`` (a 7-vector transform
    ``[px, py, pz, qx, qy, qz, qw]``) and ``state.body_qd`` (a 6-vector spatial
    velocity ``[vx, vy, vz, wx, wy, wz]``), splitting both into named parts.

    Setters mutate simulation state in place under ``no_grad`` and are not
    connected to Torch autograd or the Warp tape; for gradient-based use go
    through ``differentiable_rollout`` / ``env.reset(field=...)`` instead."""

    def __init__(self, state: Any) -> None:
        self._state = state

    @property
    def transforms(self) -> torch.Tensor:
        """Full body transforms, shape ``(num_bodies, 7)`` = position + quaternion (zero-copy)."""
        transforms = field_to_torch(self._state.body_q)
        if transforms.ndim != 2 or transforms.shape[1] != 7:
            raise ValueError("state.body_q must have shape [bodies, 7]")
        return transforms

    @transforms.setter
    def transforms(self, value: Any) -> None:
        """Write full body transforms back into ``state.body_q``."""
        _assign_value(self._state.body_q, value)

    @property
    def positions(self) -> torch.Tensor:
        """Body positions ``[m]``, shape ``(num_bodies, 3)`` (the ``q`` translation)."""
        return self.transforms[:, :3]

    @positions.setter
    def positions(self, value: Any) -> None:
        """Write body positions while preserving orientations."""
        _assign_value(self.positions, value)

    @property
    def orientations(self) -> torch.Tensor:
        """Body orientation quaternions ``[qx, qy, qz, qw]``, shape ``(num_bodies, 4)``."""
        return self.transforms[:, 3:7]

    @orientations.setter
    def orientations(self, value: Any) -> None:
        """Write body quaternions while preserving positions."""
        _assign_value(self.orientations, value)

    @property
    def spatial_velocities(self) -> torch.Tensor:
        """Full spatial velocities, shape ``(num_bodies, 6)`` = linear + angular (zero-copy)."""
        spatial_velocities = field_to_torch(self._state.body_qd)
        if spatial_velocities.ndim != 2 or spatial_velocities.shape[1] != 6:
            raise ValueError("state.body_qd must have shape [bodies, 6]")
        return spatial_velocities

    @spatial_velocities.setter
    def spatial_velocities(self, value: Any) -> None:
        """Write full spatial velocities back into ``state.body_qd``."""
        _assign_value(self._state.body_qd, value)

    @property
    def angular_velocities(self) -> torch.Tensor:
        """Body angular velocities ``[rad/s]``, shape ``(num_bodies, 3)``."""
        return self.spatial_velocities[:, 3:6]

    @angular_velocities.setter
    def angular_velocities(self, value: Any) -> None:
        """Write angular velocities while preserving linear velocities."""
        _assign_value(self.angular_velocities, value)

    @property
    def linear_velocities(self) -> torch.Tensor:
        """Body linear velocities ``[m/s]``, shape ``(num_bodies, 3)``."""
        return self.spatial_velocities[:, :3]

    @linear_velocities.setter
    def linear_velocities(self, value: Any) -> None:
        """Write linear velocities while preserving angular velocities."""
        _assign_value(self.linear_velocities, value)

    def __len__(self) -> int:
        return int(self.transforms.shape[0])


class JointView:
    """Generalized (joint-space) coordinates of an articulation. Wraps
    ``state.joint_q`` (joint coordinates) and ``state.joint_qd`` (joint speeds).

    Setters mutate simulation state in place under ``no_grad`` and are not
    connected to Torch autograd or the Warp tape; for gradient-based use go
    through ``differentiable_rollout`` / ``env.reset(field=...)`` instead."""

    def __init__(self, state: Any) -> None:
        self._state = state

    @property
    def coordinates(self) -> torch.Tensor:
        """Joint coordinates ``[m or rad]``, shape ``(num_joint_coords,)`` (zero-copy)."""
        return field_to_torch(self._state.joint_q)

    @coordinates.setter
    def coordinates(self, value: Any) -> None:
        """Write joint coordinates back into ``state.joint_q``."""
        _assign_value(self._state.joint_q, value)

    @property
    def velocities(self) -> torch.Tensor:
        """Joint speeds ``[m/s or rad/s]``, shape ``(num_joint_dofs,)`` (zero-copy)."""
        return field_to_torch(self._state.joint_qd)

    @velocities.setter
    def velocities(self, value: Any) -> None:
        """Write joint speeds back into ``state.joint_qd``."""
        _assign_value(self._state.joint_qd, value)

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])


def particles(state: Any) -> ParticleView:
    """Return a ``ParticleView`` over a particle/continuum Newton state."""
    return ParticleView(state)


def bodies(state: Any) -> BodyView:
    """Return a ``BodyView`` over a rigid-body (or rod/cable) Newton state."""
    return BodyView(state)


def joints(state: Any) -> JointView:
    """Return a ``JointView`` over an articulation's joint-space state."""
    return JointView(state)
