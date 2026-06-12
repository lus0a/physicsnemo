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

"""Rotation, quaternion, and anchor-frame helpers shared by NeRD codecs."""

from __future__ import annotations

import math

import torch

_TWO_PI = 2.0 * math.pi


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Map angles into ``[-pi, pi)``."""
    return angles - torch.floor((angles + math.pi) / _TWO_PI) * _TWO_PI


def _quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., :3] = -out[..., :3]
    return out


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def _quat_rotate_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = quat[..., :3]
    q_scalar = quat[..., 3:4]
    cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return vector - q_scalar * cross + torch.linalg.cross(q_vector, cross, dim=-1)


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vector = quat[..., :3]
    q_scalar = quat[..., 3:4]
    cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return vector + q_scalar * cross + torch.linalg.cross(q_vector, cross, dim=-1)


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _canonicalize_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat(quat)
    # At 180 degrees w is exactly zero, so the usual w >= 0 convention does
    # not distinguish q from -q. Use the first nonzero x/y/z component as a
    # deterministic tie-breaker.
    sign_source = quat[..., 3]
    unresolved = sign_source == 0.0
    for index in range(3):
        component = quat[..., index]
        sign_source = torch.where(
            unresolved & (component != 0.0),
            component,
            sign_source,
        )
        unresolved &= component == 0.0
    sign = torch.where(sign_source < 0.0, -1.0, 1.0).unsqueeze(-1)
    return quat * sign


def _quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    quat = _canonicalize_quat(quat)
    vector = quat[..., :3]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quat[..., 3:4].clamp_min(0.0))
    scale = torch.where(
        vector_norm > 1.0e-7,
        angle / vector_norm.clamp_min(1.0e-8),
        torch.full_like(vector_norm, 2.0),
    )
    return vector * scale


def _rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-7,
        torch.sin(half) / angle.clamp_min(1.0e-8),
        0.5 - angle.square() / 48.0,
    )
    return _normalize_quat(torch.cat((rotvec * scale, torch.cos(half)), dim=-1))


def _heading_quat(quat: torch.Tensor, up_axis: int) -> torch.Tensor:
    """Return the rotation about ``up_axis`` encoded by ``quat``.

    The heading frame removes global translation and yaw while preserving body
    tilt relative to gravity. This is the spatial invariance used by an explicit
    body-heading reference frame.
    """
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    quat = _canonicalize_quat(quat)
    # Swing-twist decomposition: projecting the quaternion vector part onto
    # gravity isolates the twist around the up axis without an Euler-angle
    # singularity when the body's forward direction approaches vertical.
    result = torch.zeros((*quat.shape[:-1], 4), dtype=quat.dtype, device=quat.device)
    result[..., up_axis] = quat[..., up_axis]
    result[..., 3] = quat[..., 3]
    degenerate = torch.linalg.vector_norm(result, dim=-1, keepdim=True) < 1.0e-8
    identity = torch.zeros_like(result)
    identity[..., 3] = 1.0
    result = torch.where(degenerate, identity, result)
    return _canonicalize_quat(result)


def _base_frame_transform(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    frame_position: torch.Tensor,
    frame_quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express a free-body pose and velocity in an anchor frame.

    Inverse of :func:`_world_frame_transform`. Used by maximal-coordinate
    reference-frame codecs and covered by a round-trip test against the world
    transform.
    """
    position_local = _quat_rotate_inverse(frame_quaternion, position - frame_position)
    quaternion_local = _quat_mul(_quat_inverse(frame_quaternion), quaternion)
    linear_local = _quat_rotate_inverse(frame_quaternion, linear_velocity)
    angular_local = _quat_rotate_inverse(frame_quaternion, angular_velocity)
    return position_local, quaternion_local, linear_local, angular_local


def _world_frame_transform(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    frame_position: torch.Tensor,
    frame_quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform a free-body pose and velocity from an anchor frame to world.

    Inverse of :func:`_base_frame_transform`; covered by a round-trip test
    against the base transform.
    """

    position_world = _quat_rotate(frame_quaternion, position) + frame_position
    quaternion_world = _quat_mul(frame_quaternion, quaternion)
    angular_world = _quat_rotate(frame_quaternion, angular_velocity)
    linear_world = _quat_rotate(frame_quaternion, linear_velocity)
    return position_world, quaternion_world, linear_world, angular_world
