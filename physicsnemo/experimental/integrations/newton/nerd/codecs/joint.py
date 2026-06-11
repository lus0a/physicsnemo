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

"""Generalized-coordinate (joint) NeRD codec and its layout inference.

This module covers the uniform per-world generalized-coordinate layout
(:class:`JointLayout`), its inference from a finalized Newton model, the
robot-centric encoding and wrapped-delta integration of generalized state, and
the :class:`NeRDJointStateCodec` that ties them to live Newton joint fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from physicsnemo.experimental.integrations.newton.data import (
    field_to_torch,
    torch_warp_stream,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.base import (
    NeRDStateCodec,
    _body_label,
    _model_labels,
    _normalized_label,
    _optional_numpy_field,
    _relative_index_signature,
    _uniform_world_indices,
    _validate_state_value,
)
from physicsnemo.experimental.integrations.newton.nerd.rotation import (
    _TWO_PI,
    _canonicalize_quat,
    _quat_inverse,
    _quat_mul,
    _quat_to_rotvec,
    _rotvec_to_quat,
    _wrap_to_pi,
)


@dataclass(frozen=True)
class JointLayout:
    """Uniform per-world generalized-coordinate layout from a Newton model."""

    world_count: int
    dof_q: int
    dof_qd: int
    continuous_q_mask: torch.Tensor
    base_translation_mask: torch.Tensor
    root_is_free: bool
    up_axis_index: int
    quaternion_q_starts: tuple[int, ...]
    q_indices: torch.Tensor | None = None
    qd_indices: torch.Tensor | None = None
    semantic_signature: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        starts = tuple(int(index) for index in self.quaternion_q_starts)
        if starts != tuple(sorted(set(starts))):
            raise ValueError("quaternion_q_starts must be unique and ordered")
        if any(start < 0 or start + 4 > self.dof_q for start in starts):
            raise ValueError("each quaternion must occupy four joint_q coordinates")
        if any(right < left + 4 for left, right in zip(starts, starts[1:])):
            raise ValueError("joint quaternion coordinate ranges must not overlap")
        object.__setattr__(self, "quaternion_q_starts", starts)

    @property
    def state_dim(self) -> int:
        """Width of the concatenated ``[joint_q, joint_qd]`` state."""
        return self.dof_q + self.dof_qd

    @property
    def delta_q_dim(self) -> int:
        """Joint-position target width after quaternion-to-rotvec conversion."""
        return self.dof_q - len(self.quaternion_q_starts)

    @property
    def prediction_dim(self) -> int:
        """Width of a generalized-coordinate relative-dynamics target."""
        return self.delta_q_dim + self.dof_qd


def _joint_layout(model: Any) -> JointLayout:
    """Infer the uniform per-world generalized-coordinate layout."""
    if int(model.articulation_count) <= 0:
        raise ValueError("joint state requires at least one articulation")
    from physicsnemo.experimental.integrations.newton.dependencies import (
        require_newton,
    )

    joint_types = require_newton().JointType
    prismatic_type = int(joint_types.PRISMATIC)
    revolute_type = int(joint_types.REVOLUTE)
    free_type = int(joint_types.FREE)
    # Newton stores BALL entirely as a quaternion and FREE/DISTANCE with a
    # trailing quaternion; all coordinate widths still come from its metadata.
    quaternion_joint_types = {
        int(joint_types.BALL),
        free_type,
        int(joint_types.DISTANCE),
    }
    quaternion_coordinate_count = int(joint_types.BALL.dof_count(0)[1])
    joint_groups = _uniform_world_indices(model, "joint").cpu().numpy()
    q_start = np.asarray(model.joint_q_start.numpy())
    qd_start = np.asarray(model.joint_qd_start.numpy())
    joint_type = np.asarray(model.joint_type.numpy())
    limit_lower = np.asarray(model.joint_limit_lower.numpy())
    limit_upper = np.asarray(model.joint_limit_upper.numpy())
    q_widths = np.diff(q_start)
    qd_widths = np.diff(qd_start)
    q_indices = _joint_coordinate_indices(joint_groups, q_start, "position")
    qd_indices = _joint_coordinate_indices(joint_groups, qd_start, "velocity")
    dof_q = int(q_indices.shape[1])
    dof_qd = int(qd_indices.shape[1])
    world_count = int(joint_groups.shape[0])
    template_joint_types = joint_type[joint_groups[0]]
    template_q_widths = q_widths[joint_groups[0]]
    template_qd_widths = qd_widths[joint_groups[0]]
    for group in joint_groups[1:]:
        if not (
            np.array_equal(joint_type[group], template_joint_types)
            and np.array_equal(q_widths[group], template_q_widths)
            and np.array_equal(qd_widths[group], template_qd_widths)
        ):
            raise ValueError(
                "joint state requires the same joint topology in every world"
            )

    continuous = np.zeros(dof_q, dtype=bool)
    base_translation = np.zeros(dof_q, dtype=bool)
    quaternion_q_starts: list[int] = []
    q_local = {int(index): local for local, index in enumerate(q_indices[0])}
    articulation_start = getattr(model, "articulation_start", None)
    root_joints = (
        set(np.asarray(articulation_start.numpy())[:-1].tolist())
        if articulation_start is not None
        else {int(joint_groups[0, 0])}
    )
    root_is_free = False
    for joint in joint_groups[0]:
        q0, q1 = int(q_start[joint]), int(q_start[joint + 1])
        qd0 = int(qd_start[joint])
        joint_kind = int(joint_type[joint])
        local_q = [q_local[index] for index in range(q0, q1)]
        joint_kind_enum = joint_types(joint_kind)
        actual_qd_width = int(qd_widths[joint])
        expected_qd_width, expected_q_width = joint_kind_enum.dof_count(actual_qd_width)
        if (len(local_q), actual_qd_width) != (
            int(expected_q_width),
            int(expected_qd_width),
        ):
            raise ValueError(
                f"Newton {joint_kind_enum.name} joint metadata requires "
                f"{expected_q_width} position and {expected_qd_width} velocity "
                f"coordinates; got {len(local_q)} and {actual_qd_width}"
            )
        if joint_kind in quaternion_joint_types:
            quaternion_coordinates = local_q[-quaternion_coordinate_count:]
            start = quaternion_coordinates[0]
            if quaternion_coordinates != list(
                range(start, start + quaternion_coordinate_count)
            ):
                raise ValueError(
                    "Newton quaternion joints must use contiguous joint_q coordinates"
                )
            quaternion_q_starts.append(start)
        if (
            joint_kind == revolute_type
            and float(limit_upper[qd0] - limit_lower[qd0]) > _TWO_PI
        ):
            continuous[local_q] = True
        if int(joint) in root_joints:
            root_is_free |= joint_kind == free_type
            if joint_kind == free_type:
                base_translation[local_q[:3]] = True
            elif joint_kind == prismatic_type:
                base_translation[local_q] = True
    up_axis = getattr(model, "up_axis", 2)
    joint_labels = _model_labels(model, "joint")
    body_labels = _model_labels(model, "body")
    joint_parent = _optional_numpy_field(model, "joint_parent")
    joint_child = _optional_numpy_field(model, "joint_child")
    semantic_signatures = tuple(
        tuple(
            (
                _normalized_label(joint_labels[int(joint)])
                if int(joint) < len(joint_labels)
                else "",
                int(joint_type[joint]),
                int(q_widths[int(joint)]),
                int(qd_widths[int(joint)]),
                _body_label(body_labels, joint_parent, int(joint)),
                _body_label(body_labels, joint_child, int(joint)),
            )
            for joint in group
        )
        for group in joint_groups
    )
    if any(
        signature != semantic_signatures[0] for signature in semantic_signatures[1:]
    ):
        raise ValueError(
            "NeRD requires the same ordered joint semantics in every world"
        )
    semantic_signature = semantic_signatures[0]
    return JointLayout(
        world_count=world_count,
        dof_q=dof_q,
        dof_qd=dof_qd,
        continuous_q_mask=torch.as_tensor(continuous),
        base_translation_mask=torch.as_tensor(base_translation),
        root_is_free=root_is_free,
        up_axis_index=int(getattr(up_axis, "value", up_axis)),
        quaternion_q_starts=tuple(quaternion_q_starts),
        q_indices=q_indices,
        qd_indices=qd_indices,
        semantic_signature=semantic_signature,
    )


def _joint_coordinate_indices(
    joint_groups: np.ndarray, starts: np.ndarray, name: str
) -> torch.Tensor:
    groups = [
        np.concatenate(
            tuple(
                np.arange(starts[joint], starts[joint + 1], dtype=np.int64)
                for joint in joints
            )
        )
        for joints in joint_groups
    ]
    counts = {len(group) for group in groups}
    if len(counts) != 1 or not counts or next(iter(counts)) == 0:
        raise ValueError(
            f"joint state requires equal non-zero {name} coordinate counts per world; "
            f"got {[len(group) for group in groups]}"
        )
    return torch.as_tensor(np.stack(groups), dtype=torch.long)


def _anchor_base(
    q: torch.Tensor, qd: torch.Tensor, layout: JointLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove translating-root position from generalized state."""
    if layout.root_is_free:
        raise NotImplementedError(
            "joint-space robot-centric framing does not support free roots; "
            "use NeRDBodyStateCodec for free-floating systems"
        )
    mask = layout.base_translation_mask.to(q.device)
    if not bool(mask.any()):
        return q, qd
    q = q.clone()
    q[..., mask] = 0.0
    return q, qd


def _state_to_input(
    q: torch.Tensor,
    qd: torch.Tensor,
    *,
    layout: JointLayout,
    robot_centric: bool = True,
) -> torch.Tensor:
    """Encode generalized state for a NeRD model."""
    q, qd = _anchor_base(q, qd, layout) if robot_centric else (q, qd)
    q = _canonicalize_joint_quaternions(q, layout)
    return torch.cat((q, qd), dim=-1)


def _canonicalize_joint_quaternions(
    q: torch.Tensor, layout: JointLayout
) -> torch.Tensor:
    """Normalize joint quaternions and choose a deterministic sign."""
    if not layout.quaternion_q_starts:
        return q
    q = q.clone()
    for start in layout.quaternion_q_starts:
        q[..., start : start + 4] = _canonicalize_quat(q[..., start : start + 4])
    return q


def _wrap_continuous_joint_q(q: torch.Tensor, layout: JointLayout) -> torch.Tensor:
    mask = layout.continuous_q_mask.to(q.device)
    if not bool(mask.any()):
        return q
    q = q.clone()
    q[..., mask] = _wrap_to_pi(q[..., mask])
    return q


def _next_state_to_delta(
    current: torch.Tensor, next_state: torch.Tensor, layout: JointLayout
) -> torch.Tensor:
    """Encode one generalized-coordinate transition as a manifold-aware delta."""
    if current.shape != next_state.shape or current.shape[-1] != layout.state_dim:
        raise ValueError(
            "current and next_state must have the same joint state shape ending "
            f"in {layout.state_dim}"
        )
    current_q = _canonicalize_joint_quaternions(current[..., : layout.dof_q], layout)
    next_q = _canonicalize_joint_quaternions(next_state[..., : layout.dof_q], layout)
    additive_q_delta = _wrap_continuous_joint_q(next_q - current_q, layout)
    q_parts: list[torch.Tensor] = []
    q_cursor = 0
    for start in layout.quaternion_q_starts:
        q_parts.append(additive_q_delta[..., q_cursor:start])
        relative_quat = _quat_mul(
            next_q[..., start : start + 4],
            _quat_inverse(current_q[..., start : start + 4]),
        )
        q_parts.append(_quat_to_rotvec(relative_quat))
        q_cursor = start + 4
    q_parts.append(additive_q_delta[..., q_cursor:])
    q_delta = torch.cat(q_parts, dim=-1)
    qd_delta = next_state[..., layout.dof_q :] - current[..., layout.dof_q :]
    return torch.cat((q_delta, qd_delta), dim=-1)


def _delta_to_next_state(
    current: torch.Tensor, delta: torch.Tensor, layout: JointLayout
) -> torch.Tensor:
    """Integrate a generalized-coordinate delta on scalar and quaternion manifolds."""
    if current.shape[-1] != layout.state_dim:
        raise ValueError(f"current must end in joint state width {layout.state_dim}")
    expected = (*current.shape[:-1], layout.prediction_dim)
    if delta.shape != expected:
        raise ValueError(f"delta must have shape {expected}, got {tuple(delta.shape)}")
    current_q = _canonicalize_joint_quaternions(current[..., : layout.dof_q], layout)
    delta_q = delta[..., : layout.delta_q_dim]
    q_parts: list[torch.Tensor] = []
    q_cursor = 0
    delta_cursor = 0
    for start in layout.quaternion_q_starts:
        width = start - q_cursor
        q_parts.append(
            current_q[..., q_cursor:start]
            + delta_q[..., delta_cursor : delta_cursor + width]
        )
        delta_cursor += width
        next_quat = _quat_mul(
            _rotvec_to_quat(delta_q[..., delta_cursor : delta_cursor + 3]),
            current_q[..., start : start + 4],
        )
        q_parts.append(_canonicalize_quat(next_quat))
        delta_cursor += 3
        q_cursor = start + 4
    q_parts.append(
        current_q[..., q_cursor:] + delta_q[..., delta_cursor : layout.delta_q_dim]
    )
    next_q = _wrap_continuous_joint_q(torch.cat(q_parts, dim=-1), layout)
    next_qd = current[..., layout.dof_q :] + delta[..., layout.delta_q_dim :]
    return torch.cat((next_q, next_qd), dim=-1)


class NeRDJointStateCodec(NeRDStateCodec):
    """Generalized-coordinate codec for one replicated articulation per world."""

    state_fields = frozenset(("joint_q", "joint_qd", "body_q", "body_qd"))

    def __init__(
        self,
        model: Any | None = None,
        *,
        layout: JointLayout | None = None,
        robot_centric: bool = True,
    ) -> None:
        if layout is None:
            if model is None:
                raise ValueError("model or layout is required")
            layout = _joint_layout(model)
        self.name = "joint"
        self.model = model
        self.layout = layout
        self.robot_centric = bool(robot_centric)
        self.batch_size = int(layout.world_count)
        self.state_shape = (int(layout.state_dim),)
        self.prediction_shape = (int(layout.prediction_dim),)

    def read(self, state: Any) -> torch.Tensor:
        """Read packed joint coordinates and velocities by world."""
        q = field_to_torch(state.joint_q, dtype=torch.float32)
        qd = field_to_torch(state.joint_qd, dtype=torch.float32)
        if self.layout.q_indices is not None and self.layout.qd_indices is not None:
            q = q.index_select(0, self.layout.q_indices.to(q.device).reshape(-1))
            qd = qd.index_select(0, self.layout.qd_indices.to(qd.device).reshape(-1))
        return torch.cat(
            (
                q.reshape(self.batch_size, self.layout.dof_q),
                qd.reshape(self.batch_size, self.layout.dof_qd),
            ),
            dim=-1,
        ).clone()

    def write(self, state: Any, value: torch.Tensor) -> None:
        """Write packed joint coordinates and velocities into Newton state."""
        value = _validate_state_value(value, self.batch_size, self.state_shape)
        q = value[..., : self.layout.dof_q].reshape(-1)
        qd = value[..., self.layout.dof_q :].reshape(-1)
        target_q = field_to_torch(state.joint_q)
        target_qd = field_to_torch(state.joint_qd)
        with torch.no_grad():
            if self.layout.q_indices is None or self.layout.qd_indices is None:
                target_q.copy_(q)
                target_qd.copy_(qd)
            else:
                target_q.index_copy_(
                    0, self.layout.q_indices.to(target_q.device).reshape(-1), q
                )
                target_qd.index_copy_(
                    0, self.layout.qd_indices.to(target_qd.device).reshape(-1), qd
                )

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode joint state with wrapped angles and optional root centering."""
        return _state_to_input(
            state[..., : self.layout.dof_q],
            state[..., self.layout.dof_q :],
            layout=self.layout,
            robot_centric=self.robot_centric,
        )

    def state_to_delta(
        self, current: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """Encode a joint transition with wrapped coordinate deltas."""
        return _next_state_to_delta(current, next_state, self.layout)

    def delta_to_state(
        self, current: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Integrate wrapped joint deltas onto the current state."""
        return _delta_to_next_state(current, delta, self.layout)

    def finalize_state(self, state: Any) -> None:
        """Refresh dependent body state through Newton forward kinematics."""
        if not hasattr(state, "body_q"):
            # No live Newton body state to refresh (e.g. pure-tensor rollout).
            return
        if self.model is None:
            raise ValueError(
                "this joint-state codec was reconstructed without a Newton model "
                "and cannot run forward kinematics to refresh body poses; obtain a "
                "live codec for deployment (e.g. via "
                "TrainedNeRDModel.as_step_model(newton_model=...))"
            )
        from physicsnemo.experimental.integrations.newton.dependencies import (
            require_newton,
        )

        with torch_warp_stream(field_to_torch(state.joint_q).device):
            require_newton().eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def compatibility_signature(self) -> tuple[Any, ...]:
        """Return joint layout, ordering, and semantic compatibility data."""
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.state_shape,
            self.prediction_shape,
            self.robot_centric,
            self.layout.root_is_free,
            self.layout.up_axis_index,
            tuple(self.layout.continuous_q_mask.tolist()),
            tuple(self.layout.base_translation_mask.tolist()),
            self.layout.quaternion_q_starts,
            _relative_index_signature(self.layout.q_indices),
            _relative_index_signature(self.layout.qd_indices),
            self.layout.semantic_signature,
        )
