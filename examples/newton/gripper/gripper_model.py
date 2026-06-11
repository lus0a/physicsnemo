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

"""PhysicsNeMo point-cloud surrogate for Newton gripper outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.experimental.integrations.newton import (
    grouped_candidate_ranking_loss,
)
from physicsnemo.models.pointnet import PointNetMLP


@dataclass
class MetaData(ModelMetaData):
    """Capabilities of the composed gripper outcome model."""

    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    auto_grad: bool = True


class GraspOutcomeModel(Module):
    """Predict physical outcomes and the robust Newton objective."""

    def __init__(
        self,
        design_dim: int,
        *,
        point_channels: int = 3,
        context_features: int = 11,
        point_features: int = 96,
        hidden_features: int = 256,
        max_loss: float = 10.0,
    ) -> None:
        super().__init__(meta=MetaData())
        self.design_dim = int(design_dim)
        self.point_channels = int(point_channels)
        self.context_features = int(context_features)
        self.point_features = int(point_features)
        self.hidden_features = int(hidden_features)
        self.max_loss = float(max_loss)
        if self.max_loss <= 0.0:
            raise ValueError("max_loss must be positive")
        self.core = PointNetMLP(
            point_channels=point_channels,
            global_features=design_dim + context_features,
            out_features=5,
            point_features=point_features,
            point_hidden_channels=(
                max(16, point_features),
                max(32, 2 * point_features),
            ),
            hidden_features=hidden_features,
            hidden_layers=5,
            activation_fn="silu",
            pooling="max",
        )

    def forward(
        self,
        points: torch.Tensor,
        normalized_designs: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return physical outcomes plus ``log1p(clipped Newton loss)``."""
        if context_features.ndim == 1:
            context_features = context_features[:, None]
        raw = self.core(
            points,
            torch.cat((normalized_designs, context_features), dim=-1),
        )
        lift = 1.15 * torch.sigmoid(raw[:, 0])
        slip = 0.18 * torch.sigmoid(raw[:, 1])
        rotation = torch.pi * torch.sigmoid(raw[:, 2])
        success = torch.sigmoid(raw[:, 3])
        log_loss = np.log1p(self.max_loss) * torch.sigmoid(raw[:, 4])
        return torch.stack((lift, slip, rotation, success, log_loss), dim=-1)


@dataclass(frozen=True)
class TrainingResult:
    """Loss history and held-out design diagnostics."""

    train_history: np.ndarray
    validation_history: np.ndarray
    best_epoch: int
    validation_rmse: np.ndarray
    validation_loss_rmse: float
    validation_pose_accuracy: float
    validation_pose_top3_recall: float
    train_design_ids: np.ndarray
    validation_design_ids: np.ndarray


def outcome_loss_torch(outcomes: torch.Tensor) -> torch.Tensor:
    """Differentiable counterpart of ``gripper_scene.outcome_loss``.

    The coefficients/scales/penalty below are deliberately inlined rather than
    imported from ``gripper_scene`` so this module stays decoupled and the loss
    remains a pure torch graph; they must stay in sync with
    ``gripper_scene.LOSS_*`` (lift 1.8, slip 0.65/0.08, rotation 0.30/1.25,
    success penalty 0.80).
    """
    lift = outcomes[..., 0].clamp(0.0, 1.0)
    slip = outcomes[..., 1]
    rotation = outcomes[..., 2]
    loss = (
        1.8 * (1.0 - lift).square()
        + 0.65 * (slip / 0.08).square()
        + 0.30 * (rotation / 1.25).square()
    )
    if outcomes.shape[-1] > 3:
        loss = loss + 0.80 * (1.0 - outcomes[..., 3])
    return loss


def surrogate_loss_torch(outcomes: torch.Tensor) -> torch.Tensor:
    """Return the directly learned Newton objective when available."""
    if outcomes.shape[-1] < 5:
        return outcome_loss_torch(outcomes)
    return torch.expm1(outcomes[..., 4])


def _complete_finite_design_mask(
    design_sample_ids: np.ndarray,
    finite_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only designs whose complete set of rows is finite."""
    design_sample_ids = np.asarray(design_sample_ids, dtype=np.int64)
    finite_rows = np.asarray(finite_rows, dtype=bool)
    if design_sample_ids.ndim != 1 or finite_rows.shape != design_sample_ids.shape:
        raise ValueError("finite row flags must align with one-dimensional design IDs")
    incomplete_ids = np.asarray(
        [
            design_id
            for design_id in np.unique(design_sample_ids)
            if not finite_rows[design_sample_ids == design_id].all()
        ],
        dtype=np.int64,
    )
    return ~np.isin(design_sample_ids, incomplete_ids), incomplete_ids


def fit_outcome_model(
    model: GraspOutcomeModel,
    *,
    point_clouds: np.ndarray,
    normalized_designs: np.ndarray,
    context_features: np.ndarray,
    outcomes: np.ndarray,
    design_sample_ids: np.ndarray,
    group_ids: np.ndarray | None = None,
    epochs: int = 260,
    batch_size: int = 256,
    lr: float = 2.0e-3,
    validation_fraction: float = 0.18,
    weight_decay: float = 2.0e-4,
    design_noise: float = 0.015,
    point_noise: float = 0.004,
    patience: int = 45,
    seed: int = 11,
) -> TrainingResult:
    """Train on complete design groups and hold out complete geometries."""
    if epochs <= 0 or batch_size <= 0 or lr <= 0.0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    if weight_decay < 0.0 or design_noise < 0.0 or point_noise < 0.0:
        raise ValueError("regularization values must be non-negative")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    device = next(model.parameters()).device
    point_clouds = np.asarray(point_clouds)
    normalized_designs = np.asarray(normalized_designs)
    context_features = np.asarray(context_features)
    outcomes = np.asarray(outcomes)
    design_sample_ids = np.asarray(design_sample_ids, dtype=np.int64)
    if point_clouds.ndim != 3:
        raise ValueError("point_clouds must have shape (rows, points, channels)")
    for name, values in (
        ("normalized_designs", normalized_designs),
        ("context_features", context_features),
        ("outcomes", outcomes),
    ):
        if values.ndim != 2:
            raise ValueError(f"{name} must have shape (rows, features)")
    if design_sample_ids.ndim != 1:
        raise ValueError("design_sample_ids must be one-dimensional")
    row_count = len(design_sample_ids)
    if any(
        len(values) != row_count
        for values in (
            point_clouds,
            normalized_designs,
            context_features,
            outcomes,
        )
    ):
        raise ValueError("all training arrays must contain the same number of rows")
    expected_widths = {
        "point_clouds": (point_clouds.shape[-1], model.point_channels),
        "normalized_designs": (
            normalized_designs.shape[-1],
            model.design_dim,
        ),
        "context_features": (
            context_features.shape[-1],
            model.context_features,
        ),
        "outcomes": (outcomes.shape[-1], 4),
    }
    for name, (actual, expected) in expected_widths.items():
        if actual != expected:
            raise ValueError(f"{name} must contain {expected} features, got {actual}")
    if group_ids is None:
        group_ids = np.zeros_like(design_sample_ids)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    if group_ids.shape != design_sample_ids.shape:
        raise ValueError("group_ids must align with design_sample_ids")
    finite_mask = (
        np.isfinite(point_clouds).all(axis=(1, 2))
        & np.isfinite(normalized_designs).all(axis=1)
        & np.isfinite(context_features).all(axis=1)
        & np.isfinite(outcomes).all(axis=1)
    )
    if not finite_mask.all():
        keep_mask, incomplete_ids = _complete_finite_design_mask(
            design_sample_ids,
            finite_mask,
        )
        dropped = int((~keep_mask).sum())
        print(
            f"dropping {len(incomplete_ids)} incomplete design groups "
            f"({dropped} rows) before training"
        )
        point_clouds = point_clouds[keep_mask]
        normalized_designs = normalized_designs[keep_mask]
        context_features = context_features[keep_mask]
        outcomes = outcomes[keep_mask]
        design_sample_ids = design_sample_ids[keep_mask]
        group_ids = group_ids[keep_mask]
    point_tensor = torch.as_tensor(point_clouds, dtype=torch.float32)
    design_tensor = torch.as_tensor(normalized_designs, dtype=torch.float32)
    context_tensor = torch.as_tensor(context_features, dtype=torch.float32)
    outcome_tensor = torch.as_tensor(outcomes, dtype=torch.float32)

    unique_ids = np.unique(design_sample_ids)
    if len(unique_ids) < 2:
        raise ValueError("dataset requires at least two finite design samples")
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(unique_ids)
    validation_count = min(
        len(shuffled) - 1,
        max(1, int(round(validation_fraction * len(shuffled)))),
    )
    validation_ids = np.sort(shuffled[:validation_count])
    train_ids = np.sort(shuffled[validation_count:])
    grouped_indices = tuple(
        np.flatnonzero(design_sample_ids == design_id) for design_id in unique_ids
    )
    rows_per_design = {len(indices) for indices in grouped_indices}
    if len(rows_per_design) != 1:
        raise ValueError("every design sample must contain the same number of rows")
    reference_groups = group_ids[grouped_indices[0]]
    if any(
        not np.array_equal(group_ids[indices], reference_groups)
        for indices in grouped_indices[1:]
    ):
        raise ValueError("task-group ordering must be identical for every design")
    grouped_index_tensor = torch.as_tensor(np.stack(grouped_indices), dtype=torch.long)
    grouped_points = point_tensor[grouped_index_tensor]
    grouped_designs = design_tensor[grouped_index_tensor]
    grouped_context = context_tensor[grouped_index_tensor]
    grouped_outcomes = outcome_tensor[grouped_index_tensor]
    train_design_mask = np.isin(unique_ids, train_ids)
    validation_design_mask = ~train_design_mask
    train_dataset = TensorDataset(
        grouped_points[train_design_mask],
        grouped_designs[train_design_mask],
        grouped_context[train_design_mask],
        grouped_outcomes[train_design_mask],
    )
    validation_data = (
        grouped_points[validation_design_mask].to(device),
        grouped_designs[validation_design_mask].to(device),
        grouped_context[validation_design_mask].to(device),
        grouped_outcomes[validation_design_mask].to(device),
    )
    group_tensor = torch.as_tensor(reference_groups, dtype=torch.long, device=device)
    row_count = grouped_points.shape[1]
    loader = DataLoader(
        train_dataset,
        batch_size=min(max(1, batch_size // row_count), len(train_dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.08 * lr
    )
    scales = torch.tensor((1.0, 0.08, 1.25), device=device)
    train_history = np.empty(epochs, dtype=np.float32)
    validation_history = np.empty(epochs, dtype=np.float32)
    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for points, designs, context, targets in loader:
            points = points.to(device, non_blocking=True)
            designs = designs.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            design_batch, row_batch = designs.shape[:2]
            if point_noise > 0.0:
                points = points.clone()
                points[..., :3] += point_noise * torch.randn_like(points[..., :3])
            if design_noise > 0.0:
                perturbation = design_noise * torch.randn(
                    (design_batch, 1, model.design_dim),
                    dtype=designs.dtype,
                    device=device,
                )
                designs = (designs + perturbation).clamp(0.0, 1.0)
            flat_points = points.reshape(design_batch * row_batch, *points.shape[2:])
            flat_designs = designs.reshape(design_batch * row_batch, -1)
            flat_context = context.reshape(design_batch * row_batch, -1)
            flat_targets = targets.reshape(design_batch * row_batch, -1)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(flat_points, flat_designs, flat_context)
            row_loss = _training_loss(
                prediction,
                flat_targets,
                scales,
                max_loss=model.max_loss,
            )
            design_loss = _grouped_design_loss(
                prediction.reshape(design_batch, row_batch, -1),
                targets,
                group_tensor,
                max_loss=model.max_loss,
            )
            loss = row_loss + 0.55 * design_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * design_batch
            seen += design_batch
        scheduler.step()
        train_history[epoch] = running / seen
        validation_history[epoch] = _normalized_validation_loss(
            model,
            validation_data,
            scales,
            group_tensor,
            max_loss=model.max_loss,
        )
        if validation_history[epoch] < best_validation:
            best_validation = float(validation_history[epoch])
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            train_history[epoch + 1 :] = np.nan
            validation_history[epoch + 1 :] = np.nan
            break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        points, designs, context, targets = validation_data
        design_count, row_count = designs.shape[:2]
        prediction = model(
            points.reshape(design_count * row_count, *points.shape[2:]),
            designs.reshape(design_count * row_count, -1),
            context.reshape(design_count * row_count, -1),
        )
        targets = targets.reshape(design_count * row_count, -1)
        validation_rmse = (
            torch.sqrt(torch.mean((prediction[:, :3] - targets[:, :3]).square(), dim=0))
            .cpu()
            .numpy()
        )
        validation_loss_rmse = float(
            torch.sqrt(
                torch.mean(
                    (
                        surrogate_loss_torch(prediction)
                        - outcome_loss_torch(targets).clamp_max(model.max_loss)
                    ).square()
                )
            )
        )
        predicted_losses = surrogate_loss_torch(prediction).reshape(
            design_count, row_count
        )
        target_losses = outcome_loss_torch(targets).reshape(design_count, row_count)
        validation_pose_accuracy = _candidate_topk_recall(
            predicted_losses,
            target_losses,
            group_tensor,
            count=1,
        )
        validation_pose_top3_recall = _candidate_topk_recall(
            predicted_losses,
            target_losses,
            group_tensor,
            count=3,
        )
    return TrainingResult(
        train_history=train_history,
        validation_history=validation_history,
        best_epoch=best_epoch,
        validation_rmse=validation_rmse.astype(np.float32),
        validation_loss_rmse=validation_loss_rmse,
        validation_pose_accuracy=validation_pose_accuracy,
        validation_pose_top3_recall=validation_pose_top3_recall,
        train_design_ids=train_ids,
        validation_design_ids=validation_ids,
    )


@torch.no_grad()
def predict_outcomes(
    model: GraspOutcomeModel,
    *,
    point_clouds: np.ndarray,
    normalized_designs: np.ndarray,
    context_features: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run batched surrogate inference."""
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(normalized_designs), batch_size):
        stop = min(len(normalized_designs), start + batch_size)
        outputs.append(
            model(
                torch.as_tensor(
                    point_clouds[start:stop], dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    normalized_designs[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    context_features[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs, axis=0)


def _normalized_validation_loss(
    model: GraspOutcomeModel,
    validation_data: tuple[torch.Tensor, ...],
    scales: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    max_loss: float,
) -> float:
    model.eval()
    with torch.no_grad():
        points, designs, context, targets = validation_data
        design_count, row_count = designs.shape[:2]
        prediction = model(
            points.reshape(design_count * row_count, *points.shape[2:]),
            designs.reshape(design_count * row_count, -1),
            context.reshape(design_count * row_count, -1),
        )
        row_loss = _training_loss(
            prediction,
            targets.reshape(design_count * row_count, -1),
            scales,
            max_loss=max_loss,
        )
        design_loss = _grouped_design_loss(
            prediction.reshape(design_count, row_count, -1),
            targets,
            group_ids,
            max_loss=max_loss,
        )
        return float(row_loss + 0.55 * design_loss)


def _training_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    scales: torch.Tensor,
    *,
    max_loss: float,
) -> torch.Tensor:
    """Joint physical-outcome and task-objective supervision."""
    regression = nn.functional.smooth_l1_loss(
        prediction[:, :3] / scales,
        targets[:, :3] / scales,
        beta=0.12,
    )
    classification = nn.functional.binary_cross_entropy(prediction[:, 3], targets[:, 3])
    target_log_loss = torch.log1p(outcome_loss_torch(targets).clamp_max(max_loss))
    objective_per_sample = nn.functional.smooth_l1_loss(
        prediction[:, 4],
        target_log_loss,
        beta=0.12,
        reduction="none",
    )
    objective_weight = (
        1.0 + 2.0 * torch.exp(-torch.expm1(target_log_loss) / 0.50) + targets[:, 3]
    )
    objective = (objective_weight * objective_per_sample).mean()
    return 0.45 * regression + 0.20 * classification + 0.70 * objective


def _grouped_design_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    max_loss: float,
) -> torch.Tensor:
    """Match robust best-candidate scores and their ordering across designs."""
    predicted = surrogate_loss_torch(prediction).clamp_max(max_loss)
    measured = outcome_loss_torch(targets).clamp_max(max_loss)
    predicted_robust = _robust_grouped_objective(predicted, group_ids)
    measured_robust = _robust_grouped_objective(measured, group_ids)
    candidate_ranking = grouped_candidate_ranking_loss(
        predicted,
        measured,
        group_ids,
    )
    regression = nn.functional.smooth_l1_loss(
        torch.log1p(predicted_robust),
        torch.log1p(measured_robust),
        beta=0.10,
    )
    if len(predicted_robust) < 2:
        return regression + 0.80 * candidate_ranking
    measured_delta = measured_robust[:, None] - measured_robust[None, :]
    predicted_delta = predicted_robust[:, None] - predicted_robust[None, :]
    mask = torch.triu(measured_delta.abs() > 0.08, diagonal=1)
    if not bool(mask.any()):
        return regression + 0.80 * candidate_ranking
    ranking = nn.functional.softplus(
        -measured_delta[mask].sign() * predicted_delta[mask] / 0.25
    ).mean()
    return regression + 0.12 * ranking + 0.80 * candidate_ranking


def _candidate_topk_recall(
    predicted_losses: torch.Tensor,
    target_losses: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    count: int,
) -> float:
    """Return how often the measured-best candidate appears in predicted top-k."""
    matches = []
    for group in torch.unique(group_ids, sorted=True):
        mask = group_ids == group
        predicted = predicted_losses[:, mask]
        target = target_losses[:, mask]
        shortlist = predicted.topk(
            k=min(count, predicted.shape[1]),
            dim=1,
            largest=False,
        ).indices
        target_best = target.argmin(dim=1, keepdim=True)
        matches.append((shortlist == target_best).any(dim=1).float())
    return float(torch.cat(matches).mean())


def _robust_grouped_objective(
    losses: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    best_candidates = torch.stack(
        [
            losses[:, group_ids == group].amin(dim=1)
            for group in torch.unique(group_ids, sorted=True)
        ],
        dim=1,
    )
    # Robust aggregation weights kept in sync with gripper_scene.ROBUST_MEAN_WEIGHT
    # / ROBUST_MAX_WEIGHT (0.65 / 0.35); inlined here to keep this a pure torch graph.
    return 0.65 * best_candidates.mean(dim=1) + 0.35 * best_candidates.amax(dim=1)
