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

"""Reusable adapters for NeRD controls, contacts, and input composition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from physicsnemo.experimental.integrations.newton.data import (
    _assign_value,
    field_to_torch,
    torch_warp_stream,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.entity import (
    NeRDBodyStateCodec,
)


@dataclass(frozen=True)
class NeRDControlInput:
    """Use one Newton control field as the input to a NeRD model.

    The adapter provides both callbacks needed by the standard workflow:
    :meth:`apply` writes sampled training inputs into ``env.control``, and
    :meth:`from_step` reads the same field during learned deployment. Newton
    commonly stores all replicated worlds in one flat array, so
    ``per_world_shape`` declares how that array is grouped for the model.

    Dotted field names are supported for namespaced controls, for example
    ``"mujoco.ctrl"``.
    """

    field: str
    per_world_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.field or any(not part for part in self.field.split(".")):
            raise ValueError("field must be a non-empty dotted attribute name")
        shape = tuple(int(size) for size in self.per_world_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("per_world_shape must contain positive dimensions")
        object.__setattr__(self, "per_world_shape", shape)

    def read(self, control: Any) -> torch.Tensor:
        """Return a zero-copy per-world Torch view of the configured field."""
        target = self._field(control)
        tensor = field_to_torch(target)
        width = math.prod(self.per_world_shape)
        if tensor.numel() % width:
            raise ValueError(
                f"control field {self.field!r} has {tensor.numel()} values, which "
                f"cannot be grouped as per_world_shape={self.per_world_shape}"
            )
        return tensor.reshape(-1, *self.per_world_shape)

    def write(self, control: Any, inputs: Any) -> None:
        """Write per-world input features into the configured control field."""
        target = self._field(control)
        target_tensor = field_to_torch(target)
        values = field_to_torch(
            inputs,
            dtype=target_tensor.dtype,
            device=target_tensor.device,
        )
        width = math.prod(self.per_world_shape)
        if target_tensor.numel() % width:
            raise ValueError(
                f"control field {self.field!r} has {target_tensor.numel()} values, "
                f"which cannot be grouped as per_world_shape={self.per_world_shape}"
            )
        expected = (target_tensor.numel() // width, *self.per_world_shape)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"inputs for control field {self.field!r} must have shape "
                f"{expected}, got {tuple(values.shape)}"
            )
        _assign_value(target, values.reshape(target_tensor.shape))

    def apply(
        self,
        env: Any,
        inputs: torch.Tensor,
        frame: int,
        substep: int,
    ) -> None:
        """``NeRDProblem.from_env(..., apply_inputs=...)`` callback."""
        del frame, substep
        if env.control is None:
            raise ValueError("NewtonEnv has no allocated control; call reset() first")
        self.write(env.control, inputs)

    def from_step(
        self,
        state: Any,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> torch.Tensor:
        """``TrainedNeRDModel.as_step_model(..., input_from_step=...)`` callback."""
        del state, contacts, dt
        return self.read(control)

    def _field(self, control: Any) -> Any:
        if control is None:
            raise ValueError(
                f"cannot read NeRD input field {self.field!r}: control is None"
            )
        value = control
        for part in self.field.split("."):
            if not hasattr(value, part):
                raise AttributeError(
                    f"control has no field {self.field!r}; missing component {part!r}"
                )
            value = getattr(value, part)
        if value is None:
            raise ValueError(f"control field {self.field!r} is not allocated")
        return value


def concatenate_nerd_inputs(
    *features: Any,
    entity_shape: tuple[int, ...],
) -> torch.Tensor:
    """Concatenate global and entity-aligned NeRD input features.

    Each input must be shaped ``[worlds, features]`` or
    ``[worlds, *entity_shape, features]``. Global inputs are broadcast over the
    entity dimensions before concatenation. This keeps application code simple
    when a model consumes both commands and per-body/per-particle observations.

    Parameters
    ----------
    *features : Any
        Tensor-like global or entity-aligned feature arrays.
    entity_shape : tuple[int, ...]
        Entity dimensions shared by the encoded NeRD state.

    Returns
    -------
    torch.Tensor
        Entity-aligned features with the final feature dimensions concatenated.
    """
    if not features:
        raise ValueError("at least one input feature tensor is required")
    entity_shape = tuple(int(size) for size in entity_shape)
    if not entity_shape or any(size <= 0 for size in entity_shape):
        raise ValueError("entity_shape must contain positive dimensions")
    tensors = tuple(field_to_torch(feature) for feature in features)
    batch_size = int(tensors[0].shape[0]) if tensors[0].ndim >= 2 else -1
    if batch_size <= 0:
        raise ValueError("input features must have shape [worlds, ..., features]")
    output: list[torch.Tensor] = []
    for tensor in tensors:
        if tensor.ndim < 2 or tensor.shape[0] != batch_size:
            raise ValueError(
                "all input features must have the same positive world dimension"
            )
        if tensor.device != tensors[0].device:
            raise ValueError("all input features must share a device")
        if tensor.dtype != tensors[0].dtype:
            raise ValueError("all input features must share a dtype")
        if tensor.ndim == 2:
            tensor = tensor.reshape(
                batch_size, *([1] * len(entity_shape)), tensor.shape[-1]
            ).expand(batch_size, *entity_shape, tensor.shape[-1])
        elif tuple(tensor.shape[1:-1]) != entity_shape:
            raise ValueError(
                "entity-aligned input shape must be "
                f"{entity_shape}, got {tuple(tensor.shape[1:-1])}"
            )
        output.append(tensor)
    return torch.cat(tuple(output), dim=-1)


class NeRDRigidContactInput:
    """Encode variable-length Newton rigid contacts as fixed per-body features.

    The output aligns with a :class:`NeRDBodyStateCodec` and has seven features
    per body: ``log1p(contact_count)``, the mean oriented contact normal, and the
    mean contact point in that body's local frame. A contact normal points from
    the encoded body toward its counterpart; normals are transformed into the
    codec's configured reference frame. Body-local points require no additional
    frame conversion.

    Aggregation is invariant to contact-buffer ordering and retains a fixed
    shape suitable for training, checkpointing, and live deployment. Contacts
    omitted by an overflowing Newton contact buffer cannot be represented.
    Autoregressive deployments must recompute contacts from predicted states.
    When training uses contacts observed only from teacher states, rollout or
    on-policy data may be needed to control the resulting distribution shift.

    Parameters
    ----------
    model : Any
        Finalized Newton model that owns ``shape_body``.
    codec : NeRDBodyStateCodec
        Body codec defining the per-world body ordering and reference frame.
    """

    feature_names = (
        "log_contact_count",
        "mean_normal_x",
        "mean_normal_y",
        "mean_normal_z",
        "mean_body_point_x",
        "mean_body_point_y",
        "mean_body_point_z",
    )

    def __init__(self, model: Any, codec: NeRDBodyStateCodec) -> None:
        if not isinstance(codec, NeRDBodyStateCodec):
            raise TypeError("codec must be a NeRDBodyStateCodec")
        self.codec = codec
        shape_body = getattr(model, "shape_body", None)
        if shape_body is None:
            raise ValueError("model must provide shape_body for rigid contacts")
        self._shape_body = field_to_torch(shape_body).long()
        self.device = self._shape_body.device
        body_count = int(getattr(model, "body_count", 0))
        if (
            body_count <= 0
            or self._shape_body.ndim != 1
            or self._shape_body.numel() == 0
        ):
            raise ValueError("model must contain rigid bodies and shapes")
        indices = torch.as_tensor(codec.indices, dtype=torch.long, device=self.device)
        if indices.ndim != 2:
            raise ValueError("codec indices must have shape [worlds, bodies]")
        self._body_lookup = torch.full(
            (body_count,), -1, dtype=torch.long, device=self.device
        )
        flat = indices.reshape(-1)
        if bool(((flat < 0) | (flat >= body_count)).any()):
            raise ValueError("codec body indices fall outside the Newton model")
        if torch.unique(flat).numel() != flat.numel():
            raise ValueError("codec body indices must be unique across worlds")
        self._body_lookup[flat] = torch.arange(
            flat.numel(), dtype=torch.long, device=self.device
        )
        self._token_count = int(flat.numel())
        self._output_shape = (*indices.shape, len(self.feature_names))

    def read(self, state: Any, contacts: Any) -> torch.Tensor:
        """Read fixed per-body contact features from live Newton buffers.

        Parameters
        ----------
        state : Any
            Newton state corresponding to ``contacts``. It is used to rotate
            normals into the body codec's reference frame.
        contacts : Any
            Newton contacts populated for ``state``.

        Returns
        -------
        torch.Tensor
            Contact features shaped ``[worlds, bodies, 7]`` in codec body order.

        Notes
        -----
        This method does not run collision detection. Use :meth:`read_env` when
        contacts have not already been refreshed for ``state``.
        """
        if contacts is None:
            raise ValueError("contacts are required for NeRDRigidContactInput")
        required = (
            "rigid_contact_count",
            "rigid_contact_shape0",
            "rigid_contact_shape1",
            "rigid_contact_normal",
            "rigid_contact_point0",
            "rigid_contact_point1",
        )
        missing = [name for name in required if getattr(contacts, name, None) is None]
        if missing:
            raise ValueError(
                "contacts do not provide required rigid-contact fields: "
                + ", ".join(missing)
            )

        with torch_warp_stream(self.device):
            shape0 = field_to_torch(contacts.rigid_contact_shape0).long()
            shape1 = field_to_torch(contacts.rigid_contact_shape1).long()
            normals = field_to_torch(contacts.rigid_contact_normal, dtype=torch.float32)
            point0 = field_to_torch(contacts.rigid_contact_point0, dtype=torch.float32)
            point1 = field_to_torch(contacts.rigid_contact_point1, dtype=torch.float32)
            count = field_to_torch(contacts.rigid_contact_count).long().reshape(-1)
            if count.numel() != 1:
                raise ValueError("rigid_contact_count must contain one scalar")
            if (
                shape0.ndim != 1
                or shape1.shape != shape0.shape
                or normals.shape != (shape0.shape[0], 3)
                or point0.shape != normals.shape
                or point1.shape != normals.shape
            ):
                raise ValueError("rigid contact buffers have inconsistent shapes")

            slots = torch.arange(shape0.shape[0], device=self.device)
            active = slots < count[0].clamp(min=0, max=shape0.shape[0])
            normal_sums = normals.new_zeros((self._token_count, 3))
            point_sums = normals.new_zeros((self._token_count, 3))
            counts = normals.new_zeros((self._token_count, 1))
            self._accumulate_side(
                normal_sums,
                point_sums,
                counts,
                shape0,
                normals,
                point0,
                active,
                1.0,
            )
            self._accumulate_side(
                normal_sums,
                point_sums,
                counts,
                shape1,
                normals,
                point1,
                active,
                -1.0,
            )

            selected_normals = normal_sums / counts.clamp_min(1.0)
            selected_normals = selected_normals.reshape(*self._output_shape[:-1], 3)
            body_state = self.codec.read(state)
            selected_normals = self.codec.world_vectors_to_model_frame(
                body_state, selected_normals
            )
            selected_points = (point_sums / counts.clamp_min(1.0)).reshape(
                *self._output_shape[:-1], 3
            )
            count_feature = torch.log1p(counts).reshape(*self._output_shape[:-1], 1)
            return torch.cat((count_feature, selected_normals, selected_points), dim=-1)

    def read_env(self, env: Any, *, refresh: bool = True) -> torch.Tensor:
        """Read contacts from a live ``NewtonEnv``.

        Parameters
        ----------
        env : Any
            Environment exposing ``state``, ``contacts``, and ``collide``.
        refresh : bool, optional
            Whether to recompute contacts from the current state before
            encoding. Disable only when the caller has already collided the same
            state.

        Returns
        -------
        torch.Tensor
            Fixed per-body contact features aligned with :attr:`codec`.
        """
        if refresh:
            env.collide()
        return self.read(env.state, env.contacts)

    def from_step(
        self,
        state: Any,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> torch.Tensor:
        """Read contacts through the Newton step-model callback signature.

        Parameters
        ----------
        state : Any
            Current Newton state.
        control : Any
            Current Newton control. It is unused.
        contacts : Any
            Contacts already populated for ``state``.
        dt : float
            Current step duration. It is unused.

        Returns
        -------
        torch.Tensor
            Contact features shaped ``[worlds, bodies, 7]``.
        """
        del control, dt
        return self.read(state, contacts)

    def _accumulate_side(
        self,
        normal_sums: torch.Tensor,
        point_sums: torch.Tensor,
        counts: torch.Tensor,
        shapes: torch.Tensor,
        normals: torch.Tensor,
        points: torch.Tensor,
        active: torch.Tensor,
        sign: float,
    ) -> None:
        valid_shape = active & (shapes >= 0) & (shapes < self._shape_body.numel())
        safe_shape = shapes.clamp(min=0, max=self._shape_body.numel() - 1)
        bodies = self._shape_body.index_select(0, safe_shape)
        valid_body = valid_shape & (bodies >= 0) & (bodies < self._body_lookup.numel())
        safe_body = bodies.clamp(min=0, max=self._body_lookup.numel() - 1)
        tokens = self._body_lookup.index_select(0, safe_body)
        valid = valid_body & (tokens >= 0)
        index = tokens.clamp_min(0)
        weight = valid.to(normals.dtype).unsqueeze(-1)
        expanded_index = index[:, None].expand(-1, 3)
        normal_sums.scatter_add_(0, expanded_index, sign * normals * weight)
        point_sums.scatter_add_(0, expanded_index, points * weight)
        counts.scatter_add_(0, index[:, None], weight)
