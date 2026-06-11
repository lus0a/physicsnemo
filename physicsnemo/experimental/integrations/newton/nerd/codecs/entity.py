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

"""Maximal-coordinate (body and particle) NeRD codecs.

This module covers the per-world entity indexing shared by maximal-coordinate
codecs, the canonical rigid-body packing and relative-dynamics deltas (position,
rotation vector, and velocity), and the concrete :class:`NeRDBodyStateCodec` and
:class:`NeRDParticleStateCodec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from physicsnemo.experimental.integrations.newton.data import (
    field_to_torch,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.base import (
    NeRDStateCodec,
    _entity_semantic_ids,
    _model_labels,
    _normalized_label,
    _uniform_world_indices,
    _validate_state_value,
)
from physicsnemo.experimental.integrations.newton.nerd.rotation import (
    _base_frame_transform,
    _canonicalize_quat,
    _heading_quat,
    _quat_inverse,
    _quat_mul,
    _quat_rotate_inverse,
    _quat_to_rotvec,
    _rotvec_to_quat,
    _world_frame_transform,
)

_BODY_STATE_DIM = 13
_BODY_DELTA_DIM = 12
# Raw Newton body field widths: body_q is [position(3), quaternion(4)] and
# body_qd is [linear(3), angular(3)].
_BODY_Q_DIM = 7
_BODY_QD_DIM = 6
# Particle state/delta widths: position(3) + velocity(3).
_PARTICLE_STATE_DIM = 6
_PARTICLE_DELTA_DIM = 6


def _fixed_width_tuple(value: Any, width: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {width} finite values")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class NeRDFixedFrame:
    """A static environment frame used by a body-state codec.

    The pose maps frame coordinates into Newton world coordinates. A fixed frame
    is appropriate when dynamics depend on immovable geometry such as a socket,
    fixture, terrain, or room: it removes arbitrary placement of the complete
    environment without discarding body position relative to that geometry.

    Parameters
    ----------
    position : tuple[float, float, float], optional
        Frame origin in Newton world coordinates.
    quaternion : tuple[float, float, float, float], optional
        Frame orientation in Newton ``(x, y, z, w)`` order.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        position = _fixed_width_tuple(self.position, 3, "position")
        quaternion = np.asarray(
            _fixed_width_tuple(self.quaternion, 4, "quaternion"),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-8:
            raise ValueError("quaternion must have non-zero norm")
        quaternion /= norm
        if quaternion[3] < 0.0:
            quaternion *= -1.0
        object.__setattr__(self, "position", position)
        object.__setattr__(
            self,
            "quaternion",
            tuple(float(item) for item in quaternion),
        )


@dataclass(frozen=True)
class NeRDBodyHeadingFrame:
    """A moving heading frame attached to one body.

    Use this only when translating the body and changing its heading together
    with the complete environment leaves the dynamics unchanged. Fixed world
    geometry breaks that symmetry and instead requires :class:`NeRDFixedFrame`
    or the ordinary world frame.

    Parameters
    ----------
    body : int or str, optional
        Local body index or normalized Newton body label.
    up_axis : int, optional
        Gravity-axis index. ``None`` uses ``model.up_axis``.
    """

    body: int | str = 0
    up_axis: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.body, bool) or not isinstance(self.body, (int, str)):
            raise TypeError("body must be an integer local index or string label")
        if isinstance(self.body, str) and not self.body:
            raise ValueError("body label must not be empty")
        if self.up_axis is not None and (
            isinstance(self.up_axis, bool)
            or not isinstance(self.up_axis, int)
            or self.up_axis not in (0, 1, 2)
        ):
            raise ValueError("up_axis must be 0, 1, 2, or None")


def _canonicalize_body_state(state: torch.Tensor) -> torch.Tensor:
    """Normalize body quaternions and choose a deterministic sign."""
    if state.ndim == 0 or state.shape[-1] != _BODY_STATE_DIM:
        raise ValueError("body state must have shape [..., 13]")
    state = state.clone()
    state[..., 3:7] = _canonicalize_quat(state[..., 3:7])
    return state


def _body_state_tokens(
    state: Any, *, device: str | torch.device | None = None
) -> torch.Tensor:
    """Pack live Newton ``body_q`` and ``body_qd`` into canonical tokens."""
    body_q = field_to_torch(state.body_q, device=device, dtype=torch.float32)
    body_qd = field_to_torch(state.body_qd, device=device, dtype=torch.float32)
    if body_q.ndim != 2 or body_q.shape[-1] != _BODY_Q_DIM:
        raise ValueError(f"state.body_q must have shape [bodies, {_BODY_Q_DIM}]")
    if body_qd.shape != (body_q.shape[0], _BODY_QD_DIM):
        raise ValueError(f"state.body_qd must have shape [bodies, {_BODY_QD_DIM}]")
    return _canonicalize_body_state(torch.cat((body_q, body_qd), dim=-1))


def _body_state_to_delta(
    current: torch.Tensor, next_state: torch.Tensor
) -> torch.Tensor:
    """Encode body motion as position, rotation-vector, and velocity deltas."""
    if current.shape != next_state.shape:
        raise ValueError("current and next_state must have the same body state shape")
    current = _canonicalize_body_state(current)
    next_state = _canonicalize_body_state(next_state)
    relative_quat = _quat_mul(next_state[..., 3:7], _quat_inverse(current[..., 3:7]))
    return torch.cat(
        (
            next_state[..., :3] - current[..., :3],
            _quat_to_rotvec(relative_quat),
            next_state[..., 7:] - current[..., 7:],
        ),
        dim=-1,
    )


def _delta_to_body_state(current: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Integrate a body-state relative-dynamics target."""
    current = _canonicalize_body_state(current)
    expected = (*current.shape[:-1], _BODY_DELTA_DIM)
    if delta.shape != expected:
        raise ValueError(f"delta must have shape {expected}, got {tuple(delta.shape)}")
    next_quat = _canonicalize_quat(
        _quat_mul(_rotvec_to_quat(delta[..., 3:6]), current[..., 3:7])
    )
    return torch.cat(
        (
            current[..., :3] + delta[..., :3],
            next_quat,
            current[..., 7:] + delta[..., 6:],
        ),
        dim=-1,
    )


class _NeRDEntityStateCodec(NeRDStateCodec):
    """Shared fixed-topology world indexing for body and particle codecs."""

    entity: str
    indices: torch.Tensor

    def __init__(
        self,
        model: Any,
        *,
        entity: str,
        state_width: int,
        prediction_width: int,
        indices: torch.Tensor | np.ndarray | None = None,
        semantic_ids: tuple[Any, ...] | None = None,
    ) -> None:
        self.name = entity
        self.entity = entity
        if indices is None:
            indices = _uniform_world_indices(model, entity)
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        if self.indices.ndim != 2 or self.indices.shape[1] == 0:
            raise ValueError(
                f"{entity} indices must have shape [worlds, entities_per_world]"
            )
        self.batch_size = int(self.indices.shape[0])
        self.state_shape = (int(self.indices.shape[1]), state_width)
        self.prediction_shape = (int(self.indices.shape[1]), prediction_width)
        self.semantic_ids = (
            _entity_semantic_ids(model, entity, self.indices)
            if semantic_ids is None
            else tuple(semantic_ids)
        )

    def _indices_on(self, device: torch.device) -> torch.Tensor:
        return self.indices.to(device=device)

    def compatibility_signature(self) -> tuple[Any, ...]:
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.state_shape,
            self.prediction_shape,
            self.semantic_ids,
        )


class NeRDBodyStateCodec(_NeRDEntityStateCodec):
    """Maximal-coordinate rigid-body state codec.

    Body state is ``[position, quaternion, spatial velocity]`` with width 13.
    Relative orientation is represented by a three-component rotation vector,
    so the prediction width is 12. Model inputs and transition targets may be
    expressed in an explicit fixed-environment or moving-body heading frame.
    Raw dataset and Newton state remain in world coordinates, so rollouts and
    live deployment preserve the ordinary codec contract.

    Parameters
    ----------
    model : Any
        Finalized Newton model used to infer per-world body indices.
    indices : torch.Tensor or numpy.ndarray, optional
        Explicit ``[worlds, bodies]`` body index table.
    semantic_ids : tuple[Any, ...], optional
        Batch-independent body-order signature.
    reference_frame : NeRDFixedFrame or NeRDBodyHeadingFrame, optional
        Use ``None`` for Newton world coordinates, :class:`NeRDFixedFrame` for
        coordinates relative to static environment geometry, or
        :class:`NeRDBodyHeadingFrame` when the complete problem is translation-
        and heading-equivariant.
    """

    state_fields = frozenset(("body_q", "body_qd"))

    def __init__(
        self,
        model: Any,
        *,
        indices: torch.Tensor | np.ndarray | None = None,
        semantic_ids: tuple[Any, ...] | None = None,
        reference_frame: NeRDFixedFrame | NeRDBodyHeadingFrame | None = None,
    ) -> None:
        super().__init__(
            model,
            entity="body",
            state_width=_BODY_STATE_DIM,
            prediction_width=_BODY_DELTA_DIM,
            indices=indices,
            semantic_ids=semantic_ids,
        )
        if reference_frame is not None and not isinstance(
            reference_frame, (NeRDFixedFrame, NeRDBodyHeadingFrame)
        ):
            raise TypeError(
                "reference_frame must be NeRDFixedFrame, NeRDBodyHeadingFrame, or None"
            )
        if isinstance(reference_frame, NeRDBodyHeadingFrame):
            root_body_index = self._resolve_reference_body(model, reference_frame.body)
            up_axis = reference_frame.up_axis
            if up_axis is None:
                model_up_axis = getattr(model, "up_axis", 2) if model is not None else 2
                up_axis = int(model_up_axis)
            reference_frame = NeRDBodyHeadingFrame(
                body=root_body_index,
                up_axis=int(up_axis),
            )
        self.reference_frame = reference_frame

    def read(self, state: Any) -> torch.Tensor:
        """Read canonical body poses and spatial velocities by world."""
        q = field_to_torch(state.body_q, dtype=torch.float32)
        qd = field_to_torch(state.body_qd, dtype=torch.float32)
        index = self._indices_on(q.device).reshape(-1)
        value = torch.cat((q.index_select(0, index), qd.index_select(0, index)), dim=-1)
        return _canonicalize_body_state(
            value.reshape(self.batch_size, *self.state_shape)
        )

    def write(self, state: Any, value: torch.Tensor) -> None:
        """Write canonical body poses and spatial velocities into Newton state."""
        value = _canonicalize_body_state(
            _validate_state_value(value, self.batch_size, self.state_shape)
        ).reshape(-1, self.state_shape[-1])
        q = field_to_torch(state.body_q)
        qd = field_to_torch(state.body_qd)
        index = self._indices_on(q.device).reshape(-1)
        with torch.no_grad():
            q.index_copy_(0, index, value[:, :_BODY_Q_DIM])
            qd.index_copy_(0, index, value[:, _BODY_Q_DIM:])

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Canonicalize state and express it in the configured reference frame."""
        state = _canonicalize_body_state(state)
        if self.reference_frame is None:
            return state
        frame_position, frame_quaternion = self._model_frame(state)
        return self._to_frame(state, frame_position, frame_quaternion)

    def state_to_delta(
        self, current: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """Encode body translation, rotation-vector, and velocity deltas."""
        if self.reference_frame is not None:
            current = _canonicalize_body_state(current)
            next_state = _canonicalize_body_state(next_state)
            frame_position, frame_quaternion = self._model_frame(current)
            current = self._to_frame(current, frame_position, frame_quaternion)
            next_state = self._to_frame(next_state, frame_position, frame_quaternion)
        return _body_state_to_delta(current, next_state)

    def delta_to_state(
        self, current: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Integrate body deltas and normalize resulting orientations."""
        if self.reference_frame is not None:
            current = _canonicalize_body_state(current)
            frame_position, frame_quaternion = self._model_frame(current)
            local_current = self._to_frame(current, frame_position, frame_quaternion)
            local_next = _delta_to_body_state(local_current, delta)
            return self._from_frame(local_next, frame_position, frame_quaternion)
        return _delta_to_body_state(current, delta)

    def world_vectors_to_model_frame(
        self, state: torch.Tensor, vectors: torch.Tensor
    ) -> torch.Tensor:
        """Rotate world-frame vectors into the codec's model-input frame.

        Parameters
        ----------
        state : torch.Tensor
            Raw body state shaped ``[..., bodies, 13]``.
        vectors : torch.Tensor
            World-frame vectors shaped either ``[..., 3]`` per world or
            ``[..., bodies, 3]`` aligned with ``state``.

        Returns
        -------
        torch.Tensor
            Vectors in the configured model frame, or ``vectors`` unchanged
            when world coordinates are used.
        """
        frame_quaternion, global_shape = self._vector_frame(state, vectors)
        if self.reference_frame is None:
            return vectors
        if vectors.shape == global_shape:
            frame_quaternion = frame_quaternion.squeeze(-2)
        return _quat_rotate_inverse(frame_quaternion, vectors)

    def world_points_to_model_frame(
        self, state: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """Transform world-space points into the codec's model-input frame.

        ``points`` may contain one point per world or one point per encoded body.
        This is useful for geometry observations such as targets, landmarks, or
        nearest-surface points that must use the same coordinates as body state.
        """
        _, global_shape = self._vector_frame(state, points)
        if self.reference_frame is None:
            return points
        frame_position, frame_quaternion = self._model_frame(
            _canonicalize_body_state(state)
        )
        if points.shape == global_shape:
            frame_position = frame_position.squeeze(-2)
            frame_quaternion = frame_quaternion.squeeze(-2)
        return _quat_rotate_inverse(frame_quaternion, points - frame_position)

    def compatibility_signature(self) -> tuple[Any, ...]:
        signature = super().compatibility_signature()
        if self.reference_frame is None:
            return signature
        if isinstance(self.reference_frame, NeRDFixedFrame):
            return (
                *signature,
                (
                    "fixed_frame",
                    self.reference_frame.position,
                    self.reference_frame.quaternion,
                ),
            )
        return (
            *signature,
            (
                "body_heading_frame",
                int(self.reference_frame.body),
                int(self.reference_frame.up_axis),
            ),
        )

    def _resolve_reference_body(self, model: Any, body: int | str) -> int:
        if isinstance(body, str):
            if model is None:
                raise ValueError(
                    "a heading-frame body must be an integer when no Newton "
                    "model is provided"
                )
            labels = _model_labels(model, "body")
            row = self.indices[0].detach().cpu().tolist()
            if not labels or any(index >= len(labels) for index in row):
                raise ValueError(
                    "heading-frame body labels require a Newton model with one "
                    "label per body"
                )
            normalized = [_normalized_label(labels[index]) for index in row]
            matches = [i for i, label in enumerate(normalized) if label == body]
            if len(matches) != 1:
                raise ValueError(
                    f"heading-frame body label {body!r} must identify exactly one "
                    f"body per world; available labels are {normalized}"
                )
            return matches[0]
        index = int(body)
        if index < 0 or index >= self.state_shape[0]:
            raise ValueError(
                f"heading-frame body index must be in [0, {self.state_shape[0] - 1}]"
            )
        return index

    def _model_frame(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference_frame = self.reference_frame
        if isinstance(reference_frame, NeRDBodyHeadingFrame):
            root = state[..., int(reference_frame.body), :]
            position = root[..., :3].unsqueeze(-2)
            heading = _heading_quat(
                root[..., 3:7], int(reference_frame.up_axis)
            ).unsqueeze(-2)
            return position, heading
        if isinstance(reference_frame, NeRDFixedFrame):
            leading_dimensions = len(state.shape) - 2
            prefix = (1,) * leading_dimensions
            position = state.new_tensor(reference_frame.position).reshape(*prefix, 1, 3)
            quaternion = state.new_tensor(reference_frame.quaternion).reshape(
                *prefix, 1, 4
            )
            return position, quaternion
        raise RuntimeError("world coordinates do not define a separate model frame")

    def _vector_frame(
        self, state: torch.Tensor, vectors: torch.Tensor
    ) -> tuple[torch.Tensor | None, tuple[int, ...]]:
        body_aligned_shape = (*state.shape[:-1], 3)
        global_shape = (*state.shape[:-2], 3)
        if vectors.shape not in (global_shape, body_aligned_shape):
            raise ValueError(
                "values must be global per-world or body-aligned with final width 3"
            )
        if self.reference_frame is None:
            return None, global_shape
        _, frame_quaternion = self._model_frame(_canonicalize_body_state(state))
        return frame_quaternion, global_shape

    @staticmethod
    def _to_frame(
        state: torch.Tensor,
        frame_position: torch.Tensor,
        frame_quaternion: torch.Tensor,
    ) -> torch.Tensor:
        position, quaternion, linear, angular = _base_frame_transform(
            state[..., :3],
            state[..., 3:7],
            state[..., 7:10],
            state[..., 10:13],
            frame_position,
            frame_quaternion,
        )
        return _canonicalize_body_state(
            torch.cat((position, quaternion, linear, angular), dim=-1)
        )

    @staticmethod
    def _from_frame(
        state: torch.Tensor,
        frame_position: torch.Tensor,
        frame_quaternion: torch.Tensor,
    ) -> torch.Tensor:
        position, quaternion, linear, angular = _world_frame_transform(
            state[..., :3],
            state[..., 3:7],
            state[..., 7:10],
            state[..., 10:13],
            frame_position,
            frame_quaternion,
        )
        return _canonicalize_body_state(
            torch.cat((position, quaternion, linear, angular), dim=-1)
        )


class NeRDParticleStateCodec(_NeRDEntityStateCodec):
    """Particle/continuum codec using per-particle position and velocity."""

    state_fields = frozenset(("particle_q", "particle_qd"))

    def __init__(
        self,
        model: Any,
        *,
        indices: torch.Tensor | np.ndarray | None = None,
        semantic_ids: tuple[Any, ...] | None = None,
    ) -> None:
        super().__init__(
            model,
            entity="particle",
            state_width=_PARTICLE_STATE_DIM,
            prediction_width=_PARTICLE_DELTA_DIM,
            indices=indices,
            semantic_ids=semantic_ids,
        )

    def read(self, state: Any) -> torch.Tensor:
        """Read particle positions and velocities by world."""
        q = field_to_torch(state.particle_q, dtype=torch.float32)
        qd = field_to_torch(state.particle_qd, dtype=torch.float32)
        index = self._indices_on(q.device).reshape(-1)
        value = torch.cat((q.index_select(0, index), qd.index_select(0, index)), dim=-1)
        return value.reshape(self.batch_size, *self.state_shape).clone()

    def write(self, state: Any, value: torch.Tensor) -> None:
        """Write particle positions and velocities into Newton state."""
        value = _validate_state_value(value, self.batch_size, self.state_shape).reshape(
            -1, self.state_shape[-1]
        )
        q = field_to_torch(state.particle_q)
        qd = field_to_torch(state.particle_qd)
        index = self._indices_on(q.device).reshape(-1)
        with torch.no_grad():
            q.index_copy_(0, index, value[:, :3])
            qd.index_copy_(0, index, value[:, 3:])
