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

"""Shared-design optimization over grouped task or control candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from physicsnemo.experimental.integrations.newton.design_space import DesignSpace
from physicsnemo.experimental.integrations.newton.distributed import resolve_device

GroupedLossFunction = Callable[[torch.Tensor], torch.Tensor]
RegularizerFunction = Callable[[torch.Tensor], torch.Tensor]


def shortlist_grouped_candidates(
    losses: np.ndarray,
    *,
    count: int = 1,
) -> np.ndarray:
    """Return the lowest-loss candidate indices for every leading group.

    ``losses`` are the differentiable per-candidate objectives produced by the
    grouped co-design loop (the same minimized quantity as
    ``GroupedDesignResult.losses``), so this helper uses *loss* rather than
    *score*; see :func:`optimize_design` for the score-vs-loss convention. The
    array may have any number of leading dimensions; its final dimension is
    interpreted as the candidate axis. Returned indices are ordered from lowest
    to highest loss and have the same leading dimensions plus ``count``. This is
    useful for surrogate-first workflows that send only a small action or pose
    shortlist back to an authoritative simulator.
    """
    values = np.asarray(losses)
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("losses must have a non-empty candidate dimension")
    if count <= 0:
        raise ValueError("count must be positive")
    if not np.isfinite(values).all():
        raise ValueError("losses must be finite")
    return np.argsort(values, axis=-1, kind="stable")[
        ..., : min(count, values.shape[-1])
    ]


def grouped_candidate_ranking_loss(
    predicted_losses: torch.Tensor,
    target_losses: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Train candidate selection within each task group.

    Inputs have shape ``(batch, rows)`` and ``group_ids`` assigns each row to a
    task or object. The loss combines a soft target distribution with explicit
    best-candidate classification. Applying the comparison to ``log1p(loss)``
    keeps large failed simulations from overwhelming distinctions among useful
    candidates.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if predicted_losses.ndim != 2 or target_losses.shape != predicted_losses.shape:
        raise ValueError(
            "predicted_losses and target_losses must have matching (batch, rows) shapes"
        )
    if group_ids.ndim != 1 or group_ids.shape[0] != predicted_losses.shape[1]:
        raise ValueError("group_ids must contain one entry per row")
    if (
        not torch.isfinite(predicted_losses).all()
        or not torch.isfinite(target_losses).all()
    ):
        raise ValueError("candidate losses must be finite")
    if bool((predicted_losses < 0.0).any()) or bool((target_losses < 0.0).any()):
        raise ValueError(
            "candidate losses must be non-negative loss magnitudes; "
            "negative values cannot be ranked through log1p"
        )

    losses = []
    for group in torch.unique(group_ids, sorted=True):
        mask = group_ids == group
        predicted = torch.log1p(predicted_losses[:, mask].clamp_min(0.0))
        target = torch.log1p(target_losses[:, mask].clamp_min(0.0))
        logits = -predicted / temperature
        target_distribution = torch.softmax(-target / temperature, dim=-1)
        soft_ranking = -(target_distribution * torch.log_softmax(logits, dim=-1)).sum(
            dim=-1
        )
        best_candidate = target.argmin(dim=-1)
        top_one = torch.nn.functional.cross_entropy(
            logits,
            best_candidate,
            reduction="none",
        )
        losses.append(0.5 * soft_ranking + 0.5 * top_one)
    return torch.stack(losses, dim=-1).mean()


@dataclass(frozen=True)
class GroupedDesignResult:
    """Multi-start solutions returned by :func:`optimize_grouped_design`.

    Designs are ordered by final objective. ``candidate_indices`` contains the
    lowest-loss candidate selected for each group. The per-design objective is
    named ``losses`` because :func:`optimize_grouped_design` minimizes a
    differentiable loss; the black-box :func:`optimize_design` oracle instead
    reports an opaque *score* (see :func:`optimize_design` for the score-vs-loss
    convention).
    """

    designs: np.ndarray
    losses: np.ndarray
    candidate_indices: np.ndarray
    history: np.ndarray
    design_history: np.ndarray
    top_k_history: np.ndarray
    design_space: DesignSpace | None = None

    def __post_init__(self) -> None:
        # ``frozen=True`` only blocks attribute reassignment; the backing arrays
        # would otherwise stay writable, so basic-indexed views returned by the
        # ``best_*`` properties could silently mutate the stored result. Mark the
        # fresh ``.cpu().numpy()`` copies read-only to enforce that immutability.
        for array in (
            self.designs,
            self.losses,
            self.candidate_indices,
            self.history,
            self.design_history,
            self.top_k_history,
        ):
            array.setflags(write=False)

    @property
    def best_design(self) -> np.ndarray:
        """Lowest-loss normalized shared design."""
        return self.designs[0]

    @property
    def best_loss(self) -> float:
        """Lowest surrogate objective."""
        return float(self.losses[0])

    @property
    def best_candidate_indices(self) -> np.ndarray:
        """Selected candidate index for every task group."""
        return self.candidate_indices[0]

    @property
    def best_physical_design(self) -> np.ndarray:
        """Lowest-loss design decoded through the supplied design space."""
        if self.design_space is None:
            raise RuntimeError(
                "best_physical_design requires optimize_grouped_design("
                "design_space=...)"
            )
        return np.asarray(
            self.design_space.decode(self.best_design, realize_discrete=True)
        )

    def trajectory_candidates(
        self,
        snapshots: int = 16,
        *,
        include_initial: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten evenly spaced optimizer snapshots into a candidate table.

        Returns ``(designs, step_indices, trajectory_indices)``. The metadata
        arrays align row-for-row with ``designs`` and make it straightforward
        to score, thin, verify, and later visualize candidates from throughout
        a multi-start optimization rather than considering only its endpoints.
        """
        if snapshots <= 0:
            raise ValueError("snapshots must be positive")
        first_step = 0 if include_initial else 1
        if first_step >= len(self.design_history):
            raise ValueError("design history has no requested trajectory steps")
        step_count = len(self.design_history) - first_step
        step_indices = np.unique(
            np.linspace(
                first_step,
                len(self.design_history) - 1,
                min(snapshots, step_count),
                dtype=np.int64,
            )
        )
        trajectory_count = self.design_history.shape[1]
        return (
            self.design_history[step_indices].reshape(
                len(step_indices) * trajectory_count,
                self.design_history.shape[-1],
            ),
            np.repeat(step_indices, trajectory_count),
            np.tile(np.arange(trajectory_count, dtype=np.int64), len(step_indices)),
        )


def optimize_grouped_design(
    grouped_losses: GroupedLossFunction,
    *,
    design_dim: int | None = None,
    design_space: DesignSpace | None = None,
    starts: int | np.ndarray = 16,
    steps: int = 300,
    lr: float = 4.0e-2,
    top_k_schedule: Sequence[tuple[float, int]] = (
        (0.0, 5),
        (0.65, 3),
        (0.85, 1),
    ),
    trust_radius: float | None = None,
    regularizer: RegularizerFunction | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> GroupedDesignResult:
    """Optimize one bounded design while selecting a candidate per task group.

    ``grouped_losses`` receives normalized designs of shape
    ``(starts, design_dim)`` and must return differentiable losses with shape
    ``(starts, groups, candidates)``. At every step the best ``k`` candidates
    in each group are averaged, then the group means are summed. The scheduled
    reduction from several candidates to one mirrors robust co-design loops
    that first maintain pose diversity and later commit to a control choice.

    Optimization happens in logit space, keeping every design coordinate in
    ``[0, 1]`` without projection. ``trust_radius`` optionally limits every
    coordinate to a neighborhood of its initial value, which is useful when a
    learned loss is only reliable near sampled designs. A scrambled Sobol
    sequence initializes integer ``starts``; an explicit array can be supplied
    instead. Pass a
    :class:`~physicsnemo.experimental.integrations.newton.design_space.DesignSpace` to
    infer ``design_dim``, retain the physical schema in the result, and decode
    :attr:`GroupedDesignResult.best_physical_design`.
    """
    if design_space is not None:
        if design_dim is not None and design_dim != design_space.dimension:
            raise ValueError("design_dim does not match design_space.dimension")
        design_dim = design_space.dimension
    if design_dim is None or design_dim <= 0:
        raise ValueError("design_dim must be positive")
    if steps <= 0 or lr <= 0.0:
        raise ValueError("steps and lr must be positive")
    if trust_radius is not None and not 0.0 < trust_radius <= 1.0:
        raise ValueError("trust_radius must lie in (0, 1]")
    schedule = _validate_schedule(top_k_schedule)
    torch_device = resolve_device(device)
    initial = _initial_designs(starts, design_dim, seed, torch_device)

    lower, upper = _design_limits(initial, trust_radius)
    eps = torch.finfo(initial.dtype).eps
    initial_local = (initial - lower) / (upper - lower)
    logits = torch.nn.Parameter(torch.logit(initial_local.clamp(eps, 1.0 - eps)))
    optimizer = torch.optim.Adam((logits,), lr=lr)
    start_count = initial.shape[0]
    # ``history`` is allocated lazily on step 0 so it preserves the dtype of the
    # user objective (e.g. a double-precision loss) rather than silently
    # downcasting it to the float32 optimizer precision.
    history: torch.Tensor | None = None
    design_history = torch.empty(
        (steps + 1, start_count, design_dim), device=torch_device
    )
    top_k_history = torch.empty((steps + 1,), dtype=torch.int64, device=torch_device)
    designs = losses = objective = None
    for step in range(steps + 1):
        designs = lower + torch.sigmoid(logits) * (upper - lower)
        losses = _checked_grouped_losses(grouped_losses(designs), start_count)
        top_k = _schedule_value(schedule, step / steps)
        objective = _reduce_grouped_losses(losses, top_k)
        if regularizer is not None:
            penalty = regularizer(designs)
            if penalty.shape != (start_count,):
                raise ValueError(
                    "regularizer must return one value per start; "
                    f"got {tuple(penalty.shape)}"
                )
            objective = objective + penalty
        if not torch.isfinite(objective).all():
            raise ValueError("grouped design objective must be finite")

        if history is None:
            history = torch.empty(
                (steps + 1, start_count),
                dtype=objective.dtype,
                device=torch_device,
            )
        history[step] = objective.detach()
        design_history[step] = designs.detach()
        top_k_history[step] = min(top_k, losses.shape[-1])
        if step == steps:
            break

        # Compute the gradient explicitly and assign it rather than calling
        # ``objective.sum().backward()``; the schedule advances top_k between
        # steps, so each iteration builds a fresh graph and there is no
        # accumulation to clear. (zero_grad would be redundant here.)
        logits.grad = torch.autograd.grad(objective.sum(), logits)[0]
        optimizer.step()

    # The final loop iteration (step == steps) already evaluated the schedule's
    # terminal top_k (schedule[-1][1]) and broke before stepping the optimizer,
    # so the logits are unchanged. Reuse that draw to build the returned result
    # rather than re-evaluating the (possibly expensive or stochastic) oracle.
    with torch.no_grad():
        final_objective = objective.detach()
        candidate_indices = torch.argmin(losses, dim=-1)
        order = torch.argsort(final_objective)
        return GroupedDesignResult(
            designs=designs.detach()[order].cpu().numpy(),
            losses=final_objective[order].cpu().numpy(),
            candidate_indices=candidate_indices[order].cpu().numpy(),
            history=history[:, order].cpu().numpy(),
            design_history=design_history[:, order].cpu().numpy(),
            top_k_history=top_k_history.cpu().numpy(),
            design_space=design_space,
        )


def _initial_designs(
    starts: int | np.ndarray,
    design_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(starts, int):
        if starts <= 0:
            raise ValueError("starts must be positive")
        values = (
            torch.quasirandom.SobolEngine(design_dim, scramble=True, seed=seed)
            .draw(starts)
            .to(device)
        )
    else:
        array = np.asarray(starts, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != design_dim or array.shape[0] == 0:
            raise ValueError(f"explicit starts must have shape (starts, {design_dim})")
        if not np.isfinite(array).all() or (array < 0.0).any() or (array > 1.0).any():
            raise ValueError("explicit starts must be finite and lie in [0, 1]")
        values = torch.as_tensor(array, device=device)
    return values.to(dtype=torch.float32)


def _design_limits(
    initial: torch.Tensor,
    trust_radius: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if trust_radius is None:
        return torch.zeros_like(initial), torch.ones_like(initial)
    return (
        (initial - trust_radius).clamp_min(0.0),
        (initial + trust_radius).clamp_max(1.0),
    )


def _validate_schedule(
    schedule: Sequence[tuple[float, int]],
) -> tuple[tuple[float, int], ...]:
    if not schedule:
        raise ValueError("top_k_schedule must not be empty")
    result = tuple((float(fraction), int(top_k)) for fraction, top_k in schedule)
    fractions = [entry[0] for entry in result]
    if fractions[0] != 0.0:
        raise ValueError("top_k_schedule must begin at fraction 0.0")
    if fractions != sorted(fractions) or any(
        fraction < 0.0 or fraction > 1.0 for fraction in fractions
    ):
        raise ValueError("top_k_schedule fractions must be sorted within [0, 1]")
    if any(top_k <= 0 for _, top_k in result):
        raise ValueError("top_k_schedule values must be positive")
    if result[-1][1] != 1:
        raise ValueError(
            "top_k_schedule must end at top_k == 1 so the optimization commits "
            "to a single candidate per group; otherwise the reported "
            "candidate_indices (a single argmin) would not match the "
            "top-k-averaged losses"
        )
    return result


def _schedule_value(schedule: Sequence[tuple[float, int]], fraction: float) -> int:
    value = schedule[0][1]
    for threshold, candidate in schedule:
        if fraction < threshold:
            break
        value = candidate
    return value


def _checked_grouped_losses(losses: torch.Tensor, start_count: int) -> torch.Tensor:
    if losses.ndim != 3 or losses.shape[0] != start_count:
        raise ValueError(
            "grouped_losses must return shape (starts, groups, candidates); "
            f"got {tuple(losses.shape)}"
        )
    if losses.shape[1] == 0 or losses.shape[2] == 0:
        raise ValueError("grouped losses require at least one group and candidate")
    if not torch.isfinite(losses).all():
        raise ValueError("grouped_losses returned non-finite values")
    return losses


def _reduce_grouped_losses(losses: torch.Tensor, top_k: int) -> torch.Tensor:
    selected = torch.topk(
        losses,
        k=min(top_k, losses.shape[-1]),
        dim=-1,
        largest=False,
    ).values
    return selected.mean(dim=-1).sum(dim=-1)
