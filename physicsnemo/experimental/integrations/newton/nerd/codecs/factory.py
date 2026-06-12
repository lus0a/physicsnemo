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

"""Factory that builds a ready NeRD state codec from a finalized Newton model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from physicsnemo.experimental.integrations.newton.nerd.codecs.base import (
    NeRDStateCodec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.composite import (
    NeRDCompositeStateCodec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.entity import (
    NeRDBodyHeadingFrame,
    NeRDBodyStateCodec,
    NeRDFixedFrame,
    NeRDParticleStateCodec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.joint import (
    NeRDJointStateCodec,
)


def nerd_state_codec(
    model: Any,
    representation: str | Sequence[str],
    *,
    joint_robot_centric: bool = True,
    body_reference_frame: NeRDFixedFrame | NeRDBodyHeadingFrame | None = None,
) -> NeRDStateCodec:
    """Build a ready NeRD state codec from a finalized Newton model.

    Pass ``"joint"``, ``"body"``, ``"particle"``, or an explicit sequence such
    as ``("joint", "particle")``. ``joint_robot_centric`` controls only
    generalized joint coordinates. Maximal-coordinate body state uses Newton
    world coordinates unless ``body_reference_frame`` explicitly supplies a
    fixed-environment or moving-body frame. Particle components remain in world
    coordinates; coupled applications that need transformed particle state
    should provide a custom codec.
    """
    if not isinstance(representation, str):
        return NeRDCompositeStateCodec(
            *(
                nerd_state_codec(
                    model,
                    item,
                    joint_robot_centric=joint_robot_centric,
                    body_reference_frame=body_reference_frame,
                )
                for item in representation
            )
        )
    if representation == "joint":
        return NeRDJointStateCodec(model, robot_centric=joint_robot_centric)
    if representation == "body":
        return NeRDBodyStateCodec(
            model,
            reference_frame=body_reference_frame,
        )
    if representation == "particle":
        return NeRDParticleStateCodec(model)
    raise ValueError(
        "representation must be 'joint', 'body', 'particle', or an explicit "
        f"sequence; got {representation!r}"
    )
