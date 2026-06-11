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

"""Codec self-description and reconstruction for portable NeRD checkpoints.

The built-in codecs serialize to a small descriptor and rebuild from it on load,
or rebuild against a live deployment model. Custom codecs return ``None`` and
must be supplied explicitly by the caller.
"""

from __future__ import annotations

from typing import Any

from physicsnemo.experimental.integrations.newton.nerd.codecs import (
    JointLayout,
    NeRDBodyHeadingFrame,
    NeRDBodyStateCodec,
    NeRDCompositeStateCodec,
    NeRDFixedFrame,
    NeRDJointStateCodec,
    NeRDParticleStateCodec,
    NeRDStateCodec,
)


def _deployment_codec(codec: NeRDStateCodec, model: Any) -> NeRDStateCodec:
    """Rebuild a built-in codec against a live deployment model."""

    if type(codec) is NeRDJointStateCodec:
        return NeRDJointStateCodec(model, robot_centric=codec.robot_centric)
    if type(codec) is NeRDBodyStateCodec:
        return NeRDBodyStateCodec(
            model,
            reference_frame=codec.reference_frame,
        )
    if type(codec) is NeRDParticleStateCodec:
        return NeRDParticleStateCodec(model)
    if type(codec) is NeRDCompositeStateCodec:
        return NeRDCompositeStateCodec(
            *(_deployment_codec(component, model) for component in codec.components)
        )
    raise ValueError(
        "a custom NeRD codec cannot be inferred from newton_model; "
        "pass state_codec explicitly"
    )


def _codec_descriptor(codec: NeRDStateCodec) -> dict[str, Any] | None:
    if type(codec) is NeRDJointStateCodec:
        layout = codec.layout
        return {
            "type": "joint",
            "delta_schema": "quaternion_rotvec_v1",
            "robot_centric": codec.robot_centric,
            "layout": {
                "world_count": layout.world_count,
                "dof_q": layout.dof_q,
                "dof_qd": layout.dof_qd,
                "continuous_q_mask": layout.continuous_q_mask.cpu(),
                "base_translation_mask": layout.base_translation_mask.cpu(),
                "root_is_free": layout.root_is_free,
                "up_axis_index": layout.up_axis_index,
                "quaternion_q_starts": layout.quaternion_q_starts,
                "q_indices": (
                    layout.q_indices.cpu() if layout.q_indices is not None else None
                ),
                "qd_indices": (
                    layout.qd_indices.cpu() if layout.qd_indices is not None else None
                ),
                "semantic_signature": layout.semantic_signature,
            },
        }
    if type(codec) is NeRDBodyStateCodec:
        descriptor = {
            "type": "body",
            "indices": codec.indices.cpu(),
            "semantic_ids": codec.semantic_ids,
        }
        if isinstance(codec.reference_frame, NeRDFixedFrame):
            descriptor["reference_frame"] = {
                "type": "fixed",
                "position": codec.reference_frame.position,
                "quaternion": codec.reference_frame.quaternion,
            }
        elif isinstance(codec.reference_frame, NeRDBodyHeadingFrame):
            descriptor["reference_frame"] = {
                "type": "body_heading",
                "body": codec.reference_frame.body,
                "up_axis": codec.reference_frame.up_axis,
            }
        return descriptor
    if type(codec) is NeRDParticleStateCodec:
        return {
            "type": "particle",
            "indices": codec.indices.cpu(),
            "semantic_ids": codec.semantic_ids,
        }
    if type(codec) is NeRDCompositeStateCodec:
        components = [_codec_descriptor(component) for component in codec.components]
        return (
            {"type": "composite", "components": components}
            if all(component is not None for component in components)
            else None
        )
    return None


def _codec_from_descriptor(
    descriptor: dict[str, Any] | None,
) -> NeRDStateCodec | None:
    if descriptor is None:
        return None
    codec_type = descriptor["type"]
    if codec_type == "joint":
        if descriptor.get("delta_schema") != "quaternion_rotvec_v1":
            raise ValueError(
                "this checkpoint uses an obsolete joint-delta schema; retrain it "
                "with quaternion rotation-vector targets"
            )
        return NeRDJointStateCodec(
            layout=JointLayout(**descriptor["layout"]),
            robot_centric=descriptor["robot_centric"],
        )
    if codec_type == "body":
        if any(key in descriptor for key in ("robot_centric", "root_body", "up_axis")):
            raise ValueError(
                "this checkpoint uses the obsolete implicit body-frame schema; "
                "retrain it with an explicit NeRDFixedFrame, "
                "NeRDBodyHeadingFrame, or world-frame body codec"
            )
        reference_descriptor = descriptor.get("reference_frame")
        reference_frame = None
        if reference_descriptor is not None:
            reference_type = reference_descriptor["type"]
            if reference_type == "fixed":
                reference_frame = NeRDFixedFrame(
                    position=reference_descriptor["position"],
                    quaternion=reference_descriptor["quaternion"],
                )
            elif reference_type == "body_heading":
                reference_frame = NeRDBodyHeadingFrame(
                    body=reference_descriptor["body"],
                    up_axis=reference_descriptor["up_axis"],
                )
            else:
                raise ValueError(
                    f"unknown saved NeRD body reference frame {reference_type!r}"
                )
        return NeRDBodyStateCodec(
            None,
            indices=descriptor["indices"],
            semantic_ids=tuple(descriptor["semantic_ids"]),
            reference_frame=reference_frame,
        )
    if codec_type == "particle":
        return NeRDParticleStateCodec(
            None,
            indices=descriptor["indices"],
            semantic_ids=tuple(descriptor["semantic_ids"]),
        )
    if codec_type == "composite":
        components = [
            _codec_from_descriptor(component) for component in descriptor["components"]
        ]
        if any(component is None for component in components):
            return None
        return NeRDCompositeStateCodec(*components)
    raise ValueError(f"unknown saved NeRD codec type {codec_type!r}")
