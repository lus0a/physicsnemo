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

"""Shared NeRD codec base class and model/world introspection helpers.

This module is the lowest level of the codec subpackage: it defines the
:class:`NeRDStateCodec` abstract base, the label/semantic helpers used to build
batch-independent compatibility signatures, the per-world index inference used
by every concrete codec, and the state-value validation shared across codecs.
It imports nothing from the other codec submodules.
"""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from physicsnemo.experimental.integrations.newton.data import (
    field_to_torch,
)

_WORLD_LABEL_PREFIX = re.compile(
    r"^(?:(?:world|env|environment)[_-]?\d+)/",
    flags=re.IGNORECASE,
)


def _normalized_label(label: Any) -> str:
    """Remove only the replication prefix from a Newton entity label."""
    return _WORLD_LABEL_PREFIX.sub("", str(label), count=1)


def _model_labels(model: Any, entity: str) -> tuple[str, ...]:
    if model is None:
        return ()
    labels = getattr(model, f"{entity}_label", None)
    return tuple(str(label) for label in labels) if labels is not None else ()


def _optional_numpy_field(model: Any, name: str) -> np.ndarray | None:
    value = getattr(model, name, None)
    if value is None:
        return None
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _body_label(
    labels: tuple[str, ...],
    body_indices: np.ndarray | None,
    joint: int,
) -> str:
    if body_indices is None:
        return ""
    body = int(body_indices[joint])
    if body < 0:
        return "<world>"
    return _normalized_label(labels[body]) if body < len(labels) else ""


def _relative_index_signature(indices: torch.Tensor | None) -> tuple[int, ...]:
    """Describe row-local ordering without depending on world count or offsets."""
    if indices is None:
        return ()
    rows = torch.as_tensor(indices, dtype=torch.long).detach().cpu().numpy()
    signatures = []
    for row in rows:
        order = {int(index): rank for rank, index in enumerate(np.sort(row))}
        signatures.append(tuple(order[int(index)] for index in row))
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("codec indices must use the same entity order in every world")
    return signatures[0]


def _entity_semantic_ids(
    model: Any,
    entity: str,
    indices: torch.Tensor,
) -> tuple[Any, ...]:
    """Return batch-independent ordering metadata for an entity codec."""
    relative_order = _relative_index_signature(indices)
    labels = _model_labels(model, entity)
    if not labels:
        return (relative_order, ())
    rows = tuple(
        tuple(_normalized_label(labels[int(index)]) for index in row)
        for row in indices.detach().cpu().numpy()
    )
    if any(row != rows[0] for row in rows[1:]):
        raise ValueError(
            f"NeRD requires the same ordered {entity} labels in every world"
        )
    return (relative_order, rows[0])


class NeRDStateCodec(ABC):
    """Read, encode, integrate, and write one fixed-topology Newton state.

    A codec exposes one dense batch of equal-topology Newton worlds. The first
    dimension is always the world/trajectory batch. Remaining dimensions are
    model state, for example ``(features,)`` for one articulation or
    ``(entities, features)`` for bodies and particles.

    ``state_fields`` declares every Newton state attribute changed by
    :meth:`write` or :meth:`finalize_state`. Composite codecs require this
    declaration so overlapping component ownership cannot silently overwrite a
    prediction.
    """

    name: str
    batch_size: int
    state_shape: tuple[int, ...]
    prediction_shape: tuple[int, ...]
    state_fields: frozenset[str] | None = None

    @abstractmethod
    def read(self, state: Any) -> torch.Tensor:
        """Read a stable state snapshot shaped ``[batch, *state_shape]``."""

    @abstractmethod
    def write(self, state: Any, value: torch.Tensor) -> None:
        """Write ``value`` shaped ``[batch, *state_shape]`` into Newton state."""

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Transform raw state into model input state, preserving its shape."""
        return state

    def state_to_delta(
        self, current: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """Encode one transition as a relative dynamics target."""
        return next_state - current

    def delta_to_state(
        self, current: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Integrate a predicted relative dynamics target."""
        return current + delta

    def finalize_state(self, state: Any) -> None:
        """Update dependent Newton fields after :meth:`write`.

        Most maximal-coordinate codecs need no finalization. Generalized-coordinate
        codecs use this hook to refresh body poses and velocities.
        """

    def compatibility_signature(self) -> tuple[Any, ...]:
        """Batch-independent signature used to reject semantically incompatible codecs."""
        return (
            type(self).__module__,
            type(self).__qualname__,
            self.state_shape,
            self.prediction_shape,
        )


def _uniform_world_indices(model: Any, entity: str) -> torch.Tensor:
    """Return per-world ``entity`` indices for the learned topology.

    World id ``-1`` denotes global/shared entities, which are excluded from the
    learned per-world topology -- except in the all-(-1) unreplicated
    single-world case, where every entity forms the one learned topology. Note
    that Newton "worlds" (replicated simulation instances) are unrelated to the
    distributed ``world_size`` used for DDP sharding elsewhere in this module.
    """
    world_count = int(getattr(model, "world_count", 0) or 1)
    count = int(getattr(model, f"{entity}_count", 0))
    world_field = getattr(model, f"{entity}_world", None)
    if world_field is None:
        if world_count != 1 or count == 0:
            raise ValueError(
                f"cannot infer per-world {entity} indices from this Newton model"
            )
        return torch.arange(count, dtype=torch.long).reshape(1, count)
    world_ids = field_to_torch(world_field).long()
    if world_count == 1 and not bool((world_ids == 0).any()) and count > 0:
        # Newton commonly leaves every entity at world -1 for an unreplicated
        # single-world scene. In that case all entities belong to the one learned
        # topology; for replicated scenes, -1 remains reserved for shared/global
        # entities and is intentionally excluded below.
        return torch.arange(count, dtype=torch.long).reshape(1, count)
    if bool((world_ids == -1).any()):
        warnings.warn(
            f"some {entity} entities have world id -1 (global/shared) and are "
            "excluded from the learned per-world topology",
            stacklevel=2,
        )
    groups = [
        torch.nonzero(world_ids == world, as_tuple=False).flatten()
        for world in range(world_count)
    ]
    counts = {int(group.numel()) for group in groups}
    if len(counts) != 1 or not counts or next(iter(counts)) == 0:
        raise ValueError(
            f"NeRD requires equal non-zero {entity} counts per world; "
            f"got {[int(group.numel()) for group in groups]}"
        )
    return torch.stack(tuple(groups))


def _has_per_world_entities(model: Any, entity: str) -> bool:
    """Whether ``entity`` has state owned by at least one learned world."""
    count = int(getattr(model, f"{entity}_count", 0))
    if count <= 0:
        return False
    world_count = int(getattr(model, "world_count", 0) or 1)
    world_field = getattr(model, f"{entity}_world", None)
    if world_count == 1 or world_field is None:
        return True
    world_ids = field_to_torch(world_field).long()
    return bool(((world_ids >= 0) & (world_ids < world_count)).any())


def _validate_state_value(
    value: torch.Tensor, batch_size: int, state_shape: tuple[int, ...]
) -> torch.Tensor:
    expected = (batch_size, *state_shape)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"state value must have shape {expected}, got {tuple(value.shape)}"
        )
    return value
