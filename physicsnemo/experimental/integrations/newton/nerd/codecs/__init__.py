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

"""Newton state codecs: layout inference, encoding, and integration.

A :class:`NeRDStateCodec` reads, encodes, integrates, and writes one
fixed-topology Newton state. Concrete codecs cover generalized joints, rigid
bodies, particles, and composites of those representations, plus the layout and
per-world indexing inference that builds them from a finalized Newton model.

This subpackage is split by concern:

- :mod:`.base` defines the :class:`NeRDStateCodec` ABC and the shared label,
  semantic, and per-world indexing helpers.
- :mod:`.joint` covers generalized-coordinate (joint) state.
- :mod:`.entity` covers maximal-coordinate body and particle state.
- :mod:`.composite` flattens several codecs into one vector state.
- :mod:`.factory` builds a ready codec from a finalized Newton model.
"""

from __future__ import annotations

from physicsnemo.experimental.integrations.newton.nerd.codecs.base import NeRDStateCodec
from physicsnemo.experimental.integrations.newton.nerd.codecs.composite import (
    NeRDCompositeStateCodec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.entity import (
    NeRDBodyHeadingFrame,
    NeRDBodyStateCodec,
    NeRDFixedFrame,
    NeRDParticleStateCodec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.factory import (
    nerd_state_codec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.joint import (
    JointLayout,
    NeRDJointStateCodec,
)

__all__ = [
    "JointLayout",
    "NeRDBodyHeadingFrame",
    "NeRDStateCodec",
    "NeRDJointStateCodec",
    "NeRDBodyStateCodec",
    "NeRDFixedFrame",
    "NeRDParticleStateCodec",
    "NeRDCompositeStateCodec",
    "nerd_state_codec",
]
