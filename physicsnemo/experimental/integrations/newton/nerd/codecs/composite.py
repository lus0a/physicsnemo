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

"""Composite NeRD codec flattening several codecs into one vector state."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from physicsnemo.experimental.integrations.newton.nerd.codecs.base import (
    NeRDStateCodec,
    _validate_state_value,
)


class NeRDCompositeStateCodec(NeRDStateCodec):
    """Flatten several codecs into one vector-state NeRD problem.

    Composite state is the general escape hatch for coupled Newton scenes. Each
    component retains its physically correct relative-delta codec; only the
    model-facing representation is flattened and concatenated.
    """

    def __init__(self, *components: NeRDStateCodec) -> None:
        if not components:
            raise ValueError("at least one component codec is required")
        batch_sizes = {component.batch_size for component in components}
        if len(batch_sizes) != 1:
            raise ValueError("all composite codecs must use the same world batch size")
        owners: dict[str, NeRDStateCodec] = {}
        for component in components:
            if component.state_fields is None:
                raise ValueError(
                    f"{type(component).__name__} must declare state_fields before "
                    "it can be used in a composite codec"
                )
            for state_field in sorted(component.state_fields):
                previous = owners.get(state_field)
                if previous is not None:
                    raise ValueError(
                        "composite codecs must own disjoint Newton state fields; "
                        f"{state_field!r} is owned by both "
                        f"{type(previous).__name__} and {type(component).__name__}"
                    )
                owners[state_field] = component
        self.components = tuple(components)
        self.state_fields = frozenset(owners)
        self.name = "+".join(component.name for component in components)
        self.batch_size = components[0].batch_size
        self._state_sizes = tuple(
            int(np.prod(component.state_shape)) for component in components
        )
        self._prediction_sizes = tuple(
            int(np.prod(component.prediction_shape)) for component in components
        )
        self.state_shape = (sum(self._state_sizes),)
        self.prediction_shape = (sum(self._prediction_sizes),)

    def read(self, state: Any) -> torch.Tensor:
        """Read and concatenate each component's flattened state."""
        return torch.cat(
            tuple(
                component.read(state).flatten(start_dim=1)
                for component in self.components
            ),
            dim=-1,
        )

    def write(self, state: Any, value: torch.Tensor) -> None:
        """Split flattened state and delegate writes to each component."""
        value = _validate_state_value(value, self.batch_size, self.state_shape)
        for component, part in zip(
            self.components, value.split(self._state_sizes, dim=-1)
        ):
            component.write(
                state, part.reshape(self.batch_size, *component.state_shape)
            )

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode each component and concatenate its flattened representation."""
        encoded = []
        for component, part in zip(
            self.components, state.split(self._state_sizes, dim=-1)
        ):
            shaped = part.reshape(*part.shape[:-1], *component.state_shape)
            encoded.append(
                component.encode_state(shaped).flatten(
                    start_dim=-len(component.state_shape)
                )
            )
        return torch.cat(tuple(encoded), dim=-1)

    def state_to_delta(
        self, current: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """Encode component-relative transitions into one flat delta."""
        deltas = []
        current_parts = current.split(self._state_sizes, dim=-1)
        next_parts = next_state.split(self._state_sizes, dim=-1)
        for component, cur, nxt in zip(self.components, current_parts, next_parts):
            prefix = cur.shape[:-1]
            cur = cur.reshape(*prefix, *component.state_shape)
            nxt = nxt.reshape(*prefix, *component.state_shape)
            deltas.append(
                component.state_to_delta(cur, nxt).flatten(
                    start_dim=-len(component.prediction_shape)
                )
            )
        return torch.cat(tuple(deltas), dim=-1)

    def delta_to_state(
        self, current: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Integrate each component delta and concatenate resulting state."""
        states = []
        current_parts = current.split(self._state_sizes, dim=-1)
        delta_parts = delta.split(self._prediction_sizes, dim=-1)
        for component, cur, part in zip(self.components, current_parts, delta_parts):
            prefix = cur.shape[:-1]
            cur = cur.reshape(*prefix, *component.state_shape)
            part = part.reshape(*prefix, *component.prediction_shape)
            states.append(
                component.delta_to_state(cur, part).flatten(
                    start_dim=-len(component.state_shape)
                )
            )
        return torch.cat(tuple(states), dim=-1)

    def finalize_state(self, state: Any) -> None:
        """Run dependent-state finalization for every component codec."""
        for component in self.components:
            component.finalize_state(state)

    def compatibility_signature(self) -> tuple[Any, ...]:
        """Return ordered compatibility signatures for all components."""
        return (
            type(self).__module__,
            type(self).__qualname__,
            tuple(component.compatibility_signature() for component in self.components),
        )
