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

"""Reusable bounded design spaces, differentiable geometric regularizers, and
design selection/verification helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np
import torch

VariableScale = Literal["linear", "log"]
VariableKind = Literal["continuous", "integer"]
CoordinateSpace = Literal["normalized", "physical"]


@dataclass(frozen=True)
class VerifiedDesignSelection:
    """Result of comparing simulator-verified proposals with incumbents.

    All indices are absolute row indices into the ``scores``/``sources`` arrays
    passed to :func:`select_verified_design`.

    Parameters
    ----------
    index:
        Absolute index of the accepted winner. Equals ``proposal_index`` when
        ``accepted`` is True, otherwise ``incumbent_index``.
    proposal_index:
        Absolute index of the best (lowest-score) proposal.
    incumbent_index:
        Absolute index of the best (lowest-score) incumbent.
    accepted:
        Whether the best proposal beat the best incumbent by more than
        ``min_improvement``.
    """

    index: int
    proposal_index: int
    incumbent_index: int
    accepted: bool


def select_diverse_designs(
    designs: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
    min_distance: float = 0.0,
    group_ids: np.ndarray | None = None,
    min_per_group: int = 0,
    required_indices: Sequence[int] = (),
) -> np.ndarray:
    """Select strong designs while preferring separation and group coverage.

    Designs are considered in ascending ``scores`` order. A candidate is first
    accepted only when its Euclidean distance from every selected design is at
    least ``min_distance``. If that leaves fewer than ``count`` rows, the best
    remaining rows fill the result. This makes diversity a preference rather
    than a reason to return an unexpectedly short verification batch.

    When ``group_ids`` and ``min_per_group`` are supplied, the selector first
    reserves up to ``min_per_group`` candidates from each group. This is useful
    for retaining coverage across optimizer trajectories, generator seeds, or
    ensemble members before globally ranking the remaining verification budget.
    ``required_indices`` can additionally preserve known anchors such as the
    first update from every optimizer trajectory.

    Parameters
    ----------
    designs:
        Two-dimensional array with one normalized or consistently scaled design
        per row.
    scores:
        One finite, lower-is-better score per design.
    count:
        Maximum number of selected rows.
    min_distance:
        Preferred minimum Euclidean distance between selected rows.
    group_ids:
        Optional one-dimensional group label per design.
    min_per_group:
        Minimum candidates reserved per group before global selection.
    required_indices:
        Candidate rows that must appear in the result, in the supplied order.
        These anchors are always included regardless of ``min_distance`` and are
        not checked for separation against one another, but they act as fixed
        anchors: every subsequently considered candidate must stay at least
        ``min_distance`` away from them.
    """
    values = np.asarray(designs)
    objective = np.asarray(scores)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("designs must be a non-empty two-dimensional array")
    if objective.shape != (len(values),):
        raise ValueError("scores must contain one value per design")
    if count <= 0:
        raise ValueError("count must be positive")
    if not np.isfinite(values).all() or not np.isfinite(objective).all():
        raise ValueError("designs and scores must be finite")
    if not np.isfinite(min_distance) or min_distance < 0.0:
        raise ValueError("min_distance must be finite and non-negative")
    if min_per_group < 0:
        raise ValueError("min_per_group must be non-negative")
    groups = None if group_ids is None else np.asarray(group_ids)
    if groups is None:
        if min_per_group:
            raise ValueError("group_ids are required when min_per_group is positive")
    elif groups.shape != (len(values),):
        raise ValueError("group_ids must contain one value per design")
    required = tuple(int(index) for index in required_indices)
    if len(set(required)) != len(required):
        raise ValueError("required_indices must be unique")
    if any(index < 0 or index >= len(values) for index in required):
        raise ValueError("required_indices must select valid design rows")
    if len(required) > min(count, len(values)):
        raise ValueError("count must accommodate every required index")

    order = np.argsort(objective, kind="stable")
    selected: list[int] = list(required)
    if len(selected) == min(count, len(values)):
        return np.asarray(selected, dtype=np.int64)

    def is_separated(index: int) -> bool:
        if not selected:
            return True
        return bool(
            np.linalg.norm(values[index] - values[selected], axis=1).min()
            >= min_distance
        )

    if groups is not None and min_per_group:
        unique_groups = np.unique(groups)
        group_order = sorted(
            unique_groups,
            key=lambda group: float(objective[groups == group].min()),
        )
        for reserve_round in range(min_per_group):
            for group in group_order:
                candidates = [
                    int(index)
                    for index in order
                    if groups[index] == group and int(index) not in selected
                ]
                if not candidates:
                    continue
                separated = next(
                    (index for index in candidates if is_separated(index)),
                    None,
                )
                # For a group's first reserved slot we still take its best
                # candidate even if no separated one exists, so each group is
                # represented. Subsequent reserved slots only accept a genuinely
                # separated candidate; otherwise we skip the group and let the
                # global phase fill the budget, avoiding near-duplicate reserves.
                if separated is None:
                    if reserve_round == 0:
                        separated = candidates[0]
                    else:
                        continue
                selected.append(separated)
                if len(selected) == min(count, len(values)):
                    return np.asarray(selected, dtype=np.int64)

    for index in order:
        if int(index) in selected or not is_separated(int(index)):
            continue
        selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < min(count, len(values)):
        selected_set = set(selected)
        selected.extend(int(index) for index in order if int(index) not in selected_set)
    return np.asarray(selected[: min(count, len(values))], dtype=np.int64)


def select_verified_design(
    scores: np.ndarray,
    sources: Sequence[str],
    *,
    proposal_source: str = "proposal",
    incumbent_source: str = "incumbent",
    min_improvement: float = 0.0,
) -> VerifiedDesignSelection:
    """Accept a proposal only when verification beats the incumbent.

    ``scores`` must come from the authoritative evaluator, such as a simulator
    or experiment, rather than the surrogate used to create the proposals.
    Lower scores are better. Positive infinity may mark a candidate that failed
    verification and is therefore ineligible. The best proposal is accepted
    only when it improves on the best incumbent by more than
    ``min_improvement``; otherwise the incumbent is retained.
    """
    objective = np.asarray(scores)
    labels = np.asarray(tuple(sources))
    if objective.ndim != 1 or len(objective) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if labels.shape != objective.shape:
        raise ValueError("sources must contain one label per score")
    if np.isnan(objective).any() or np.isneginf(objective).any():
        raise ValueError("scores must not contain NaN or negative infinity")
    if not np.isfinite(min_improvement) or min_improvement < 0.0:
        raise ValueError("min_improvement must be finite and non-negative")
    proposal_indices = np.flatnonzero(labels == proposal_source)
    incumbent_indices = np.flatnonzero(labels == incumbent_source)
    if len(proposal_indices) == 0:
        raise ValueError(f"no candidates use proposal source {proposal_source!r}")
    if len(incumbent_indices) == 0:
        raise ValueError(f"no candidates use incumbent source {incumbent_source!r}")
    eligible_indices = np.concatenate((proposal_indices, incumbent_indices))
    if not np.isfinite(objective[eligible_indices]).any():
        raise ValueError(
            "scores must contain at least one finite proposal or incumbent candidate"
        )
    proposal_index = int(proposal_indices[np.argmin(objective[proposal_indices])])
    incumbent_index = int(incumbent_indices[np.argmin(objective[incumbent_indices])])
    accepted = bool(
        objective[proposal_index] + min_improvement < objective[incumbent_index]
    )
    return VerifiedDesignSelection(
        index=proposal_index if accepted else incumbent_index,
        proposal_index=proposal_index,
        incumbent_index=incumbent_index,
        accepted=accepted,
    )


@dataclass(frozen=True)
class DesignVariable:
    """One named coordinate in a bounded physical design space.

    Parameters
    ----------
    name:
        Stable identifier used by constraints, reports, and cached datasets.
    lower, upper:
        Inclusive physical bounds.
    scale:
        Interpolation used between normalized and physical coordinates.
    kind:
        ``"integer"`` keeps a continuous relaxation during optimization and
        rounds only when :meth:`DesignSpace.decode` realizes a design.
    tags:
        Optional semantic groups such as ``"length"`` or ``"finger"``.
    """

    name: str
    lower: float
    upper: float
    scale: VariableScale = "linear"
    kind: VariableKind = "continuous"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        tags = tuple(self.tags)
        if not self.name:
            raise ValueError("design variable names must not be empty")
        if not np.isfinite((lower, upper)).all():
            raise ValueError(f"bounds for {self.name!r} must be finite")
        if lower >= upper:
            raise ValueError(f"lower bound must be below upper bound for {self.name!r}")
        if self.scale not in ("linear", "log"):
            raise ValueError(f"unsupported scale {self.scale!r} for {self.name!r}")
        if self.scale == "log" and lower <= 0.0:
            raise ValueError(f"log-scaled variable {self.name!r} must be positive")
        if self.kind not in ("continuous", "integer"):
            raise ValueError(f"unsupported kind {self.kind!r} for {self.name!r}")
        if self.kind == "integer" and (
            not lower.is_integer() or not upper.is_integer()
        ):
            raise ValueError(f"integer variable {self.name!r} requires integer bounds")
        if len(set(tags)) != len(tags):
            raise ValueError(f"tags for {self.name!r} must be unique")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "tags", tags)

    def to_config(self) -> dict[str, object]:
        """Return a JSON-serializable variable definition."""
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "scale": self.scale,
            "kind": self.kind,
            "tags": list(self.tags),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> DesignVariable:
        """Construct a variable from :meth:`to_config` output."""
        tags = config.get("tags", ())
        if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
            raise ValueError("design variable tags must be a sequence")
        return cls(
            name=str(config["name"]),
            lower=float(config["lower"]),
            upper=float(config["upper"]),
            scale=cast(VariableScale, str(config.get("scale", "linear"))),
            kind=cast(VariableKind, str(config.get("kind", "continuous"))),
            tags=tuple(str(tag) for tag in tags),
        )


@dataclass(frozen=True)
class DesignSpace:
    """Named bounded variables with NumPy/Torch-safe coordinate transforms."""

    variables: tuple[DesignVariable, ...]

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        if not variables:
            raise ValueError("a design space requires at least one variable")
        names = tuple(variable.name for variable in variables)
        if len(set(names)) != len(names):
            raise ValueError("design variable names must be unique")
        object.__setattr__(self, "variables", variables)

    @property
    def dimension(self) -> int:
        """Number of independently optimized coordinates."""
        return len(self.variables)

    @property
    def names(self) -> tuple[str, ...]:
        """Variable names in vector order."""
        return tuple(variable.name for variable in self.variables)

    @property
    def lower(self) -> np.ndarray:
        """Physical lower bounds in vector order."""
        return np.asarray(
            [variable.lower for variable in self.variables], dtype=np.float32
        )

    @property
    def upper(self) -> np.ndarray:
        """Physical upper bounds in vector order."""
        return np.asarray(
            [variable.upper for variable in self.variables], dtype=np.float32
        )

    @property
    def fingerprint(self) -> str:
        """Stable digest suitable for dataset and checkpoint compatibility."""
        encoded = json.dumps(
            [variable.to_config() for variable in self.variables],
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_config(self) -> dict[str, object]:
        """Return a JSON-serializable design-space definition."""
        return {
            "variables": [variable.to_config() for variable in self.variables],
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> DesignSpace:
        """Reconstruct a design space from :meth:`to_config` output."""
        variables = config.get("variables")
        if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)):
            raise ValueError("design-space config requires a variables sequence")
        if any(not isinstance(variable, Mapping) for variable in variables):
            raise ValueError("each design-space variable must be a mapping")
        return cls(
            tuple(
                DesignVariable.from_config(variable)
                for variable in variables
                if isinstance(variable, Mapping)
            )
        )

    def index(self, name: str) -> int:
        """Return the vector index for ``name``."""
        try:
            return self.names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown design variable {name!r}") from error

    def indices(
        self,
        *,
        names: Sequence[str] | None = None,
        tag: str | None = None,
    ) -> tuple[int, ...]:
        """Resolve explicit names or all variables carrying a semantic tag."""
        if (names is None) == (tag is None):
            raise ValueError("provide exactly one of names or tag")
        if names is not None:
            return tuple(self.index(name) for name in names)
        result = tuple(
            index
            for index, variable in enumerate(self.variables)
            if tag in variable.tags
        )
        if not result:
            raise KeyError(f"no design variables carry tag {tag!r}")
        return result

    def decode(
        self,
        normalized: np.ndarray | torch.Tensor,
        *,
        realize_discrete: bool = False,
        clip: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Map normalized coordinates to physical values.

        Torch inputs remain differentiable unless ``realize_discrete`` rounds
        integer variables. The input may have any leading batch dimensions.
        """
        self._check_last_dimension(normalized)
        if torch.is_tensor(normalized):
            units = normalized.clamp(0.0, 1.0) if clip else normalized
            lower = torch.as_tensor(self.lower, dtype=units.dtype, device=units.device)
            upper = torch.as_tensor(self.upper, dtype=units.dtype, device=units.device)
            physical = lower + units * (upper - lower)
            log_mask = torch.as_tensor(
                [variable.scale == "log" for variable in self.variables],
                dtype=torch.bool,
                device=units.device,
            )
            if bool(log_mask.any()):
                safe_lower = torch.where(log_mask, lower, torch.ones_like(lower))
                safe_upper = torch.where(log_mask, upper, torch.ones_like(upper))
                log_values = torch.exp(
                    torch.log(safe_lower)
                    + units * (torch.log(safe_upper) - torch.log(safe_lower))
                )
                physical = torch.where(log_mask, log_values, physical)
            if realize_discrete:
                integer_mask = torch.as_tensor(
                    [variable.kind == "integer" for variable in self.variables],
                    dtype=torch.bool,
                    device=units.device,
                )
                realized = physical.round().clamp(lower, upper)
                physical = torch.where(integer_mask, realized, physical)
            return physical

        units = np.asarray(normalized)
        output_dtype = np.result_type(units.dtype, np.float32)
        units = units.astype(output_dtype, copy=False)
        if clip:
            units = np.clip(units, 0.0, 1.0)
        lower = self.lower.astype(output_dtype)
        upper = self.upper.astype(output_dtype)
        physical = lower + units * (upper - lower)
        log_mask = np.asarray(
            [variable.scale == "log" for variable in self.variables], dtype=bool
        )
        if log_mask.any():
            physical[..., log_mask] = np.exp(
                np.log(lower[log_mask])
                + units[..., log_mask]
                * (np.log(upper[log_mask]) - np.log(lower[log_mask]))
            )
        if realize_discrete:
            integer_mask = np.asarray(
                [variable.kind == "integer" for variable in self.variables],
                dtype=bool,
            )
            physical[..., integer_mask] = np.clip(
                np.rint(physical[..., integer_mask]),
                lower[integer_mask],
                upper[integer_mask],
            )
        return physical.astype(output_dtype, copy=False)

    def decode_named(
        self,
        normalized: np.ndarray | torch.Tensor,
        *,
        realize_discrete: bool = False,
        clip: bool = True,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Decode physical coordinates into a dictionary keyed by name.

        A scene can intentionally reuse one named value for several parts to
        encode exact parameter sharing or symmetry.
        """
        physical = self.decode(
            normalized,
            realize_discrete=realize_discrete,
            clip=clip,
        )
        return {name: physical[..., index] for index, name in enumerate(self.names)}

    def encode(
        self,
        physical: np.ndarray | torch.Tensor,
        *,
        clip: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Map physical values to normalized coordinates."""
        self._check_last_dimension(physical)
        if torch.is_tensor(physical):
            values = physical
            lower = torch.as_tensor(
                self.lower, dtype=values.dtype, device=values.device
            )
            upper = torch.as_tensor(
                self.upper, dtype=values.dtype, device=values.device
            )
            if clip:
                values = values.clamp(lower, upper)
            units = (values - lower) / (upper - lower)
            log_mask = torch.as_tensor(
                [variable.scale == "log" for variable in self.variables],
                dtype=torch.bool,
                device=values.device,
            )
            if bool(log_mask.any()):
                if not clip and bool((values[..., log_mask] <= 0.0).any()):
                    raise ValueError(
                        "log-scaled variables require positive physical values; "
                        "got a value <= 0 with clip=False"
                    )
                safe_values = torch.where(log_mask, values, torch.ones_like(values))
                safe_lower = torch.where(log_mask, lower, torch.ones_like(lower))
                safe_upper = torch.where(log_mask, upper, torch.ones_like(upper))
                log_units = (torch.log(safe_values) - torch.log(safe_lower)) / (
                    torch.log(safe_upper) - torch.log(safe_lower)
                )
                units = torch.where(log_mask, log_units, units)
            return units.clamp(0.0, 1.0) if clip else units

        values = np.asarray(physical)
        output_dtype = np.result_type(values.dtype, np.float32)
        values = values.astype(output_dtype, copy=False)
        lower = self.lower.astype(output_dtype)
        upper = self.upper.astype(output_dtype)
        if clip:
            values = np.clip(values, lower, upper)
        units = (values - lower) / (upper - lower)
        log_mask = np.asarray(
            [variable.scale == "log" for variable in self.variables], dtype=bool
        )
        if log_mask.any():
            if not clip and (values[..., log_mask] <= 0.0).any():
                raise ValueError(
                    "log-scaled variables require positive physical values; "
                    "got a value <= 0 with clip=False"
                )
            units[..., log_mask] = (
                np.log(values[..., log_mask]) - np.log(lower[log_mask])
            ) / (np.log(upper[log_mask]) - np.log(lower[log_mask]))
        if clip:
            units = np.clip(units, 0.0, 1.0)
        return units.astype(output_dtype, copy=False)

    def sample_sobol(
        self,
        count: int,
        *,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Sample normalized designs with a reproducible scrambled Sobol sequence."""
        if count <= 0:
            raise ValueError("count must be positive")
        samples = torch.quasirandom.SobolEngine(
            self.dimension, scramble=True, seed=seed
        ).draw(count)
        if device is not None:
            return samples.to(device)
        return samples.numpy()

    def _check_last_dimension(self, values: np.ndarray | torch.Tensor) -> None:
        if values.ndim == 0 or values.shape[-1] != self.dimension:
            raise ValueError(
                f"design vectors must end with dimension {self.dimension}; "
                f"got shape {tuple(values.shape)}"
            )


class DesignConstraint(Protocol):
    """Protocol for differentiable penalties over normalized designs."""

    def penalty(
        self, normalized: torch.Tensor, design_space: DesignSpace
    ) -> torch.Tensor:
        """Return one non-negative penalty per leading design row."""


@dataclass(frozen=True)
class SimilarityConstraint:
    """Penalize variation among one or more groups of related variables.

    Exact symmetry is better represented by sharing one design variable.
    This soft constraint is intended for designs where related parts may vary
    but should remain visually or mechanically similar.
    """

    groups: tuple[tuple[str, ...], ...]
    weight: float = 1.0
    coordinate_space: CoordinateSpace = "normalized"

    def __post_init__(self) -> None:
        groups = tuple(tuple(group) for group in self.groups)
        if not groups or any(len(group) < 2 for group in groups):
            raise ValueError("similarity groups must each contain at least two names")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("constraint weight must be non-negative")
        if self.coordinate_space not in ("normalized", "physical"):
            raise ValueError("coordinate_space must be 'normalized' or 'physical'")
        object.__setattr__(self, "groups", groups)

    def penalty(
        self, normalized: torch.Tensor, design_space: DesignSpace
    ) -> torch.Tensor:
        """Return weighted within-group variance for each design."""
        if not torch.is_tensor(normalized):
            raise TypeError("design constraints require a torch.Tensor")
        values = (
            design_space.decode(normalized)
            if self.coordinate_space == "physical"
            else normalized
        )
        result = torch.zeros(
            values.shape[:-1], dtype=values.dtype, device=values.device
        )
        for group in self.groups:
            selected = values[..., design_space.indices(names=group)]
            result = result + (
                selected - selected.mean(dim=-1, keepdim=True)
            ).square().mean(dim=-1)
        return self.weight * result


@dataclass(frozen=True)
class SmoothnessConstraint:
    """Penalize first- or second-order differences along an ordered profile."""

    names: tuple[str, ...]
    weight: float = 1.0
    order: Literal[1, 2] = 1
    coordinate_space: CoordinateSpace = "normalized"

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if len(names) <= self.order:
            raise ValueError("smoothness requires more names than its difference order")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("constraint weight must be non-negative")
        if self.order not in (1, 2):
            raise ValueError("smoothness order must be 1 or 2")
        if self.coordinate_space not in ("normalized", "physical"):
            raise ValueError("coordinate_space must be 'normalized' or 'physical'")
        object.__setattr__(self, "names", names)

    def penalty(
        self, normalized: torch.Tensor, design_space: DesignSpace
    ) -> torch.Tensor:
        """Return weighted profile-difference energy for each design."""
        if not torch.is_tensor(normalized):
            raise TypeError("design constraints require a torch.Tensor")
        values = (
            design_space.decode(normalized)
            if self.coordinate_space == "physical"
            else normalized
        )
        selected = values[..., design_space.indices(names=self.names)]
        differences = torch.diff(selected, n=self.order, dim=-1)
        return self.weight * differences.square().mean(dim=-1)


@dataclass(frozen=True)
class DesignRegularizer:
    """Compose reusable differentiable constraints into one optimizer callback."""

    design_space: DesignSpace
    constraints: tuple[DesignConstraint, ...]

    def __post_init__(self) -> None:
        constraints = tuple(self.constraints)
        if not constraints:
            raise ValueError("a design regularizer requires at least one constraint")
        object.__setattr__(self, "constraints", constraints)

    def __call__(self, normalized: torch.Tensor) -> torch.Tensor:
        self.design_space._check_last_dimension(normalized)
        if not torch.is_tensor(normalized):
            raise TypeError("design regularizers require a torch.Tensor")
        result = torch.zeros(
            normalized.shape[:-1],
            dtype=normalized.dtype,
            device=normalized.device,
        )
        for constraint in self.constraints:
            result = result + constraint.penalty(normalized, self.design_space)
        return result


__all__ = [
    "DesignConstraint",
    "DesignRegularizer",
    "DesignSpace",
    "DesignVariable",
    "SimilarityConstraint",
    "SmoothnessConstraint",
    "VerifiedDesignSelection",
    "select_diverse_designs",
    "select_verified_design",
]
