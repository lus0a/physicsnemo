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

"""Offline neural co-design of an articulated gripper with PhysicsNeMo/Newton."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import gripper_model as models
import gripper_scene as scene
import numpy as np
import torch

from physicsnemo.experimental.integrations.newton import (
    DesignRegularizer,
    SmoothnessConstraint,
    optimize_grouped_design,
    resolve_device,
    select_diverse_designs,
    select_verified_design,
    shortlist_grouped_candidates,
)


def _dataset_cache_key(
    args: argparse.Namespace,
    *,
    poses,
    options: scene.SimulationOptions,
) -> str:
    """Return a local cache key for inputs that change Newton supervision."""
    poses_array = np.ascontiguousarray(_training_pose_values(poses), dtype=np.float32)
    payload = {
        "scene_version": scene.SCENE_VERSION,
        "design_space": scene.GRIPPER_DESIGN_SPACE.fingerprint,
        "training_objects": scene.TRAIN_OBJECT_LIBRARY_FINGERPRINT,
        "training_object_indices": scene.TRAIN_OBJECT_INDICES,
        "simulation_options": asdict(options),
        "design_samples": args.design_samples,
        "seed": args.seed,
        "training_poses_shape": poses_array.shape,
        "training_poses": hashlib.sha256(poses_array.tobytes()).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _training_pose_values(poses) -> np.ndarray:
    """Return only pose rows that participate in Newton supervision."""
    return np.stack(
        [
            [pose.as_array() for pose in poses[object_index]]
            for object_index in scene.TRAIN_OBJECT_INDICES
        ]
    )


def generate_offline_dataset(
    *,
    design_samples: int,
    object_indices: tuple[int, ...],
    poses,
    clouds: np.ndarray,
    contexts: np.ndarray,
    options: scene.SimulationOptions,
    batch_size: int,
    device: str,
    seed: int,
    cache_key: str,
) -> dict[str, np.ndarray]:
    """Generate the fixed Cartesian design/object/pose Newton dataset."""
    unit_designs = np.asarray(
        scene.GRIPPER_DESIGN_SPACE.sample_sobol(design_samples, seed=seed),
        dtype=np.float32,
    )
    pose_count = len(poses[0])
    group_size = len(object_indices) * pose_count
    design_sample_ids = np.repeat(np.arange(design_samples), group_size)
    row_object_indices = np.tile(
        np.repeat(np.asarray(object_indices, dtype=np.int64), pose_count),
        design_samples,
    )
    row_pose_indices = np.tile(
        np.tile(np.arange(pose_count, dtype=np.int64), len(object_indices)),
        design_samples,
    )
    normalized_designs = np.repeat(unit_designs, group_size, axis=0)
    physical_designs = scene.unit_to_design(normalized_designs)
    started = time.perf_counter()
    metrics = scene.evaluate_designs(
        physical_designs,
        row_object_indices,
        row_pose_indices,
        poses=poses,
        options=options,
        device=device,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - started
    print(
        f"offline Newton dataset: {len(normalized_designs):,} worlds in {elapsed:.1f} s"
    )
    return {
        "cache_key": np.asarray((cache_key,)),
        "newton_generation_seconds": np.asarray((elapsed,), dtype=np.float64),
        "unit_design_samples": unit_designs,
        "normalized_designs": normalized_designs,
        "physical_designs": physical_designs,
        "design_sample_ids": design_sample_ids.astype(np.int64),
        "object_indices": row_object_indices,
        "pose_indices": row_pose_indices,
        "pose_values": _training_pose_values(poses),
        "point_clouds": clouds[row_object_indices, row_pose_indices],
        "context_features": contexts[row_object_indices, row_pose_indices],
        **metrics,
    }


def load_or_generate_dataset(
    args: argparse.Namespace,
    *,
    poses,
    clouds: np.ndarray,
    contexts: np.ndarray,
    options: scene.SimulationOptions,
) -> dict[str, np.ndarray]:
    """Load a compatible Newton cache or regenerate and persist it."""
    dataset_path = Path(args.dataset)
    cache_key = _dataset_cache_key(
        args,
        poses=poses,
        options=options,
    )
    if dataset_path.exists() and not args.regenerate_dataset:
        loaded = np.load(dataset_path)
        dataset = {key: loaded[key] for key in loaded.files}
        expected = (
            args.design_samples * len(scene.TRAIN_OBJECT_INDICES) * args.pose_count
        )
        cached_key = np.asarray(dataset.get("cache_key", ())).reshape(-1)
        cache_key_matches = len(cached_key) == 1 and str(cached_key[0]) == cache_key
        cache_is_compatible = (
            "normalized_designs" in dataset
            and "success" in dataset
            and "unit_design_samples" in dataset
            and len(dataset["normalized_designs"]) == expected
            and dataset["normalized_designs"].shape[1] == scene.DESIGN_DIM
            and len(dataset["unit_design_samples"]) == args.design_samples
            and cache_key_matches
        )
        if cache_is_compatible:
            row_objects = dataset["object_indices"].astype(np.int64)
            row_poses = dataset["pose_indices"].astype(np.int64)
            dataset["point_clouds"] = clouds[row_objects, row_poses]
            dataset["context_features"] = contexts[row_objects, row_poses]
            dataset["success"] = scene.retained_grasp_success(
                dataset["lift_fraction"],
                dataset["lateral_slip"],
            )
            dataset["loss"] = scene.outcome_loss(
                dataset["lift_fraction"],
                dataset["lateral_slip"],
                dataset["rotation_error"],
            ) + scene.LOSS_SUCCESS_PENALTY * (1.0 - dataset["success"])
            print(f"loaded fixed Newton dataset from {dataset_path}")
            return dataset
        cached_rows = (
            len(dataset["normalized_designs"]) if "normalized_designs" in dataset else 0
        )
        print(
            f"dataset cache is incompatible ({cached_rows} rows, expected "
            f"{expected} with design dimension {scene.DESIGN_DIM}; "
            "cache key changed or is missing); regenerating"
        )

    dataset = generate_offline_dataset(
        design_samples=args.design_samples,
        object_indices=scene.TRAIN_OBJECT_INDICES,
        poses=poses,
        clouds=clouds,
        contexts=contexts,
        options=options,
        batch_size=args.sim_batch_size,
        device=args.newton_device,
        seed=args.seed,
        cache_key=cache_key,
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **dataset)
    return dataset


def grouped_surrogate_loss(
    model: models.GraspOutcomeModel,
    *,
    clouds: np.ndarray,
    contexts: np.ndarray,
    object_indices: tuple[int, ...],
):
    """Create a differentiable ``(starts, objects, poses)`` loss callback."""
    device = next(model.parameters()).device
    points = torch.as_tensor(
        clouds[np.asarray(object_indices)],
        dtype=torch.float32,
        device=device,
    )
    context = torch.as_tensor(
        contexts[np.asarray(object_indices)],
        dtype=torch.float32,
        device=device,
    )
    group_count, pose_count, point_count, point_channels = points.shape

    def predict_loss(designs: torch.Tensor) -> torch.Tensor:
        start_count = designs.shape[0]
        point_batch = (
            points[None]
            .expand(start_count, -1, -1, -1, -1)
            .reshape(
                start_count * group_count * pose_count,
                point_count,
                point_channels,
            )
        )
        design_batch = (
            designs[:, None, None, :]
            .expand(-1, group_count, pose_count, -1)
            .reshape(-1, scene.DESIGN_DIM)
        )
        context_batch = (
            context[None]
            .expand(start_count, -1, -1, -1)
            .reshape(start_count * group_count * pose_count, -1)
        )
        outcomes = model(point_batch, design_batch, context_batch)
        return models.surrogate_loss_torch(outcomes).reshape(
            start_count, group_count, pose_count
        )

    return predict_loss


PROFILE_REGULARIZER = DesignRegularizer(
    scene.GRIPPER_DESIGN_SPACE,
    (
        SmoothnessConstraint(
            tuple(scene.DESIGN_NAMES[scene.LENGTH_SLICE]), weight=0.002
        ),
        SmoothnessConstraint(
            tuple(scene.DESIGN_NAMES[scene.RADIUS_SLICE]), weight=0.002
        ),
        SmoothnessConstraint(
            tuple(scene.DESIGN_NAMES[scene.REST_ANGLE_SLICE]), weight=0.003
        ),
    ),
)


def design_regularizer(designs: torch.Tensor) -> torch.Tensor:
    """Favor smooth profiles, modest curvature, and modest material use."""
    material = (
        0.45 * designs[:, scene.LENGTH_SLICE].mean(dim=1)
        + 0.30 * designs[:, scene.RADIUS_SLICE].mean(dim=1)
        + 0.15 * designs[:, scene.ROOT_HALF_WIDTH_INDEX]
        + 0.05 * designs[:, scene.PAD_PROTRUSION_INDEX]
        + 0.05 * designs[:, scene.PAD_HALF_WIDTH_INDEX]
    )
    curvature_complexity = designs[:, scene.REST_ANGLE_SLICE].square().mean(dim=1)
    return (
        0.006 * curvature_complexity + 0.0015 * material + PROFILE_REGULARIZER(designs)
    )


def design_simplicity_penalty(designs: np.ndarray) -> np.ndarray:
    """NumPy counterpart used for Newton finalist selection and reporting."""
    values = torch.as_tensor(np.asarray(designs), dtype=torch.float32)
    return design_regularizer(values).detach().cpu().numpy()


def _seed_reproducibly(seed: int) -> None:
    """Seed model training and select deterministic Torch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def optimization_starts(
    dataset: dict[str, np.ndarray],
    *,
    count: int,
    pose_count: int,
) -> np.ndarray:
    """Select strong measured starts while preserving geometry diversity."""
    design_count = len(dataset["unit_design_samples"])
    grouped = dataset["loss"].reshape(
        design_count, len(scene.TRAIN_OBJECT_INDICES), pose_count
    )
    best_pose_loss = grouped.min(axis=2)
    robust = scene.ROBUST_MEAN_WEIGHT * best_pose_loss.mean(
        axis=1
    ) + scene.ROBUST_MAX_WEIGHT * best_pose_loss.max(axis=1)
    unit_designs = dataset["unit_design_samples"]
    if not len(unit_designs):
        raise ValueError("optimization requires at least one measured design")
    selected = select_diverse_designs(
        unit_designs,
        robust,
        count=count,
        min_distance=0.10,
    )
    return unit_designs[selected].astype(np.float32)


def select_poses(
    model: models.GraspOutcomeModel,
    unit_design: np.ndarray,
    *,
    clouds: np.ndarray,
    contexts: np.ndarray,
    object_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Select the lowest predicted-loss pose for each object."""
    pose_count = clouds.shape[1]
    row_objects = np.repeat(np.asarray(object_indices, dtype=np.int64), pose_count)
    row_poses = np.tile(np.arange(pose_count, dtype=np.int64), len(object_indices))
    rows = len(row_objects)
    outcomes = models.predict_outcomes(
        model,
        point_clouds=clouds[row_objects, row_poses],
        normalized_designs=np.repeat(unit_design[None], rows, axis=0),
        context_features=contexts[row_objects, row_poses],
    )
    losses = _surrogate_loss_numpy(outcomes).reshape(len(object_indices), pose_count)
    return np.argmin(losses, axis=1).astype(np.int64), losses


def validate_finalists(
    model: models.GraspOutcomeModel,
    finalists: np.ndarray,
    *,
    clouds: np.ndarray,
    contexts: np.ndarray,
    poses,
    options: scene.SimulationOptions,
    device: str,
    batch_size: int,
    shortlist_count: int,
) -> tuple[int, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Newton-verify surrogate-shortlisted poses for candidate designs."""
    finalist_count = len(finalists)
    selection_object_indices = (
        *scene.TRAIN_OBJECT_INDICES,
        *scene.VALIDATION_OBJECT_INDICES,
    )
    object_count = len(selection_object_indices)
    # select_poses returns (best_pose_argmin, per_pose_loss_grid); only the loss
    # grid is wanted here, so the argmin vector ([0]) is intentionally discarded.
    predicted_losses = np.stack(
        [
            select_poses(
                model,
                design,
                clouds=clouds,
                contexts=contexts,
                object_indices=selection_object_indices,
            )[1]
            for design in finalists
        ]
    )
    pose_shortlists = shortlist_grouped_candidates(
        predicted_losses,
        count=shortlist_count,
    )
    shortlist_size = pose_shortlists.shape[-1]
    row_designs = np.repeat(
        scene.unit_to_design(finalists), object_count * shortlist_size, axis=0
    )
    row_objects = np.tile(
        np.repeat(
            np.asarray(selection_object_indices, dtype=np.int64),
            shortlist_size,
        ),
        finalist_count,
    )
    row_poses = pose_shortlists.reshape(-1)
    metrics = scene.evaluate_designs(
        row_designs,
        row_objects,
        row_poses,
        poses=poses,
        options=options,
        device=device,
        batch_size=batch_size,
    )
    losses = metrics["loss"].reshape(finalist_count, object_count, shortlist_size)
    selected_shortlist_indices = np.argmin(losses, axis=2)
    selected_poses = np.take_along_axis(
        pose_shortlists,
        selected_shortlist_indices[..., None],
        axis=2,
    ).squeeze(axis=2)
    shaped = {}
    for key, value in metrics.items():
        values = value.reshape(
            (finalist_count, object_count, shortlist_size, *value.shape[1:])
        )
        gather_index = selected_shortlist_indices[
            (...,) + (None,) * (values.ndim - selected_shortlist_indices.ndim)
        ]
        shaped[key] = np.take_along_axis(values, gather_index, axis=2).squeeze(axis=2)
    selected_losses = shaped["loss"]
    tail_count = max(1, int(np.ceil(selected_losses.shape[1] / 3)))
    upper_tail = np.sort(selected_losses, axis=1)[:, -tail_count:].mean(axis=1)
    robust = 0.75 * selected_losses.mean(axis=1) + 0.25 * upper_tail
    validation_success = shaped["success"][:, len(scene.TRAIN_OBJECT_INDICES) :]
    selection_objective = _apply_validation_gate(
        robust + design_simplicity_penalty(finalists),
        validation_success,
    )
    best = int(np.argmin(selection_objective))
    return best, shaped, selection_objective, selected_poses


def _apply_validation_gate(
    objective: np.ndarray,
    validation_success: np.ndarray,
) -> np.ndarray:
    """Make candidates that fail any validation object ineligible."""
    objective = np.asarray(objective, dtype=np.float64)
    validation_success = np.asarray(validation_success)
    if validation_success.ndim != 2 or validation_success.shape[0] != len(objective):
        raise ValueError(
            "validation_success must have shape [candidates, validation_objects]"
        )
    if validation_success.shape[1] == 0:
        raise ValueError("validation_success must contain at least one object")
    eligible = np.all(validation_success >= 0.5, axis=1)
    if not np.any(eligible):
        raise RuntimeError(
            "no finalist succeeded on every validation object; refusing to "
            "select a geometry that failed the validation gate"
        )
    return np.where(eligible, objective, np.inf)


def validate_named_design(
    model: models.GraspOutcomeModel,
    unit_design: np.ndarray,
    *,
    clouds: np.ndarray,
    contexts: np.ndarray,
    poses,
    options: scene.SimulationOptions,
    device: str,
    batch_size: int,
    shortlist_count: int,
    object_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Shortlist poses and validate one named design with Newton."""
    if object_indices is None:
        object_indices = tuple(range(len(scene.OBJECTS)))
    if not object_indices:
        raise ValueError("object_indices must not be empty")
    neural_selected, predicted_losses = select_poses(
        model,
        unit_design,
        clouds=clouds,
        contexts=contexts,
        object_indices=object_indices,
    )
    pose_shortlists = shortlist_grouped_candidates(
        predicted_losses,
        count=shortlist_count,
    )
    shortlist_size = pose_shortlists.shape[-1]
    row_objects = np.repeat(np.asarray(object_indices), shortlist_size)
    row_poses = pose_shortlists.reshape(-1)
    all_metrics = scene.evaluate_designs(
        np.repeat(
            scene.unit_to_design(unit_design[None]),
            len(object_indices) * shortlist_size,
            axis=0,
        ),
        row_objects,
        row_poses,
        poses=poses,
        options=options,
        device=device,
        batch_size=batch_size,
    )
    loss_grid = all_metrics["loss"].reshape(len(object_indices), shortlist_size)
    selected_shortlist_indices = np.argmin(loss_grid, axis=1)
    verified_selected = np.take_along_axis(
        pose_shortlists,
        selected_shortlist_indices[:, None],
        axis=1,
    ).squeeze(axis=1)
    metrics = {}
    for key, value in all_metrics.items():
        values = value.reshape((len(object_indices), shortlist_size, *value.shape[1:]))
        gather_index = selected_shortlist_indices[
            (...,) + (None,) * (values.ndim - selected_shortlist_indices.ndim)
        ]
        metrics[key] = np.take_along_axis(values, gather_index, axis=1).squeeze(axis=1)
    return neural_selected, verified_selected, metrics, predicted_losses


def _outcome_loss_numpy(outcomes: np.ndarray) -> np.ndarray:
    # Delegate to the single source of truth in gripper_scene; scene.outcome_loss
    # takes the three positional metrics and excludes the success penalty, so
    # re-add it here when the packed outcome carries a success column.
    loss = scene.outcome_loss(
        outcomes[..., 0],
        outcomes[..., 1],
        outcomes[..., 2],
    )
    if outcomes.shape[-1] > 3:
        loss = loss + scene.LOSS_SUCCESS_PENALTY * (1.0 - outcomes[..., 3])
    return loss.astype(np.float32)


def _surrogate_loss_numpy(outcomes: np.ndarray) -> np.ndarray:
    """Return the directly learned objective when the model provides one."""
    if outcomes.shape[-1] < 5:
        return _outcome_loss_numpy(outcomes)
    return np.expm1(outcomes[..., 4]).astype(np.float32)


def _robust_score_numpy(losses: np.ndarray) -> float:
    best_candidates = np.min(losses, axis=-1)
    return float(
        scene.ROBUST_MEAN_WEIGHT * best_candidates.mean()
        + scene.ROBUST_MAX_WEIGHT * best_candidates.max()
    )


def _sampled_design_objectives(
    dataset: dict[str, np.ndarray],
    *,
    pose_count: int,
    model: models.GraspOutcomeModel | None = None,
    shortlist_count: int = 1,
) -> np.ndarray:
    design_count = len(dataset["unit_design_samples"])
    losses = dataset["loss"].reshape(
        design_count, len(scene.TRAIN_OBJECT_INDICES), pose_count
    )
    if model is None:
        best_candidates = losses.min(axis=2)
    else:
        prediction = models.predict_outcomes(
            model,
            point_clouds=dataset["point_clouds"],
            normalized_designs=dataset["normalized_designs"],
            context_features=dataset["context_features"],
        )
        predicted_losses = _surrogate_loss_numpy(prediction).reshape(losses.shape)
        shortlists = shortlist_grouped_candidates(
            predicted_losses,
            count=shortlist_count,
        )
        shortlisted_losses = np.take_along_axis(losses, shortlists, axis=2)
        best_candidates = shortlisted_losses.min(axis=2)
    robust = scene.ROBUST_MEAN_WEIGHT * best_candidates.mean(
        axis=1
    ) + scene.ROBUST_MAX_WEIGHT * best_candidates.max(axis=1)
    return robust + design_simplicity_penalty(dataset["unit_design_samples"])


def _select_refined_candidates(
    model: models.GraspOutcomeModel,
    *,
    grouped_loss,
    optimization,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select low-loss, geometrically diverse points along optimizer paths."""
    if count <= 0:
        raise ValueError("refined candidate count must be positive")
    candidates, step_indices, trajectory_indices = optimization.trajectory_candidates(
        snapshots=20
    )
    candidate_scores = _score_grouped_designs(
        grouped_loss,
        candidates,
        device=next(model.parameters()).device,
    )
    trajectory_order = sorted(
        np.unique(trajectory_indices),
        key=lambda trajectory: float(
            candidate_scores[trajectory_indices == trajectory].min()
        ),
    )
    anchor_trajectories = trajectory_order[: min(count, len(trajectory_order))]
    anchor_indices = tuple(
        int(indices[np.argmin(step_indices[indices])])
        for trajectory in anchor_trajectories
        for indices in (np.flatnonzero(trajectory_indices == trajectory),)
    )
    selected = select_diverse_designs(
        candidates,
        candidate_scores,
        count=count,
        min_distance=0.020,
        group_ids=trajectory_indices,
        min_per_group=1,
        required_indices=anchor_indices,
    )
    return (
        candidates[selected].astype(np.float32),
        candidate_scores[selected].astype(np.float32),
        step_indices[selected],
        trajectory_indices[selected],
    )


def _score_grouped_designs(
    grouped_loss,
    designs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Score normalized designs with the same grouped surrogate contract."""
    scores = []
    for start in range(0, len(designs), batch_size):
        batch = torch.as_tensor(
            designs[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            losses = grouped_loss(batch)
            best_candidates = losses.amin(dim=-1)
            score = best_candidates.sum(dim=-1) + design_regularizer(batch)
        scores.append(score.cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def _sampled_incumbents(
    model: models.GraspOutcomeModel,
    dataset: dict[str, np.ndarray],
    *,
    pose_count: int,
    shortlist_count: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the strongest measured designs from the fixed Sobol table."""
    if count <= 0:
        raise ValueError("sampled incumbent count must be positive")
    objectives = _sampled_design_objectives(
        dataset,
        pose_count=pose_count,
        model=model,
        shortlist_count=shortlist_count,
    )
    indices = np.argsort(objectives)[: min(count, len(objectives))]
    return (
        dataset["unit_design_samples"][indices].astype(np.float32),
        objectives[indices].astype(np.float32),
    )


def _surrogate_input_audit(
    model: models.GraspOutcomeModel,
    *,
    dataset: dict[str, np.ndarray],
    training,
) -> dict[str, float]:
    """Measure whether held-out predictions depend on object surface geometry."""
    validation_mask = np.isin(
        dataset["design_sample_ids"], training.validation_design_ids
    )
    points = dataset["point_clouds"][validation_mask]
    designs = dataset["normalized_designs"][validation_mask]
    context = dataset["context_features"][validation_mask]
    measured = dataset["loss"][validation_mask]
    design_ids = dataset["design_sample_ids"][validation_mask]
    predictions = {
        "pointnet": _surrogate_loss_numpy(
            models.predict_outcomes(
                model,
                point_clouds=points,
                normalized_designs=designs,
                context_features=context,
            )
        ),
        "surface_masked": _surrogate_loss_numpy(
            models.predict_outcomes(
                model,
                point_clouds=np.zeros_like(points),
                normalized_designs=designs,
                context_features=context,
            )
        ),
    }
    metrics: dict[str, float] = {}
    for label, predicted in predictions.items():
        measured_scores = []
        predicted_scores = []
        top_one = []
        top_three = []
        for design_id in training.validation_design_ids:
            design_mask = design_ids == design_id
            measured_grid = measured[design_mask].reshape(
                len(scene.TRAIN_OBJECT_INDICES), -1
            )
            predicted_grid = predicted[design_mask].reshape(
                len(scene.TRAIN_OBJECT_INDICES), -1
            )
            measured_scores.append(_robust_score_numpy(measured_grid))
            predicted_scores.append(_robust_score_numpy(predicted_grid))
            best_measured = np.argmin(measured_grid, axis=1)
            predicted_order = np.argsort(predicted_grid, axis=1)
            top_one.extend(predicted_order[:, 0] == best_measured)
            top_three.extend(
                np.any(predicted_order[:, :3] == best_measured[:, None], axis=1)
            )

        measured_scores = np.asarray(measured_scores)
        predicted_scores = np.asarray(predicted_scores)
        measured_delta = measured_scores[:, None] - measured_scores[None, :]
        predicted_delta = predicted_scores[:, None] - predicted_scores[None, :]
        ranking_mask = np.triu(np.abs(measured_delta) > 0.08, k=1)
        metrics[f"{label}_ranking_accuracy"] = (
            float(
                np.mean(
                    np.sign(measured_delta[ranking_mask])
                    == np.sign(predicted_delta[ranking_mask])
                )
            )
            if ranking_mask.any()
            else 0.5
        )
        metrics[f"{label}_pose_top1_recall"] = float(np.mean(top_one))
        metrics[f"{label}_pose_top3_recall"] = float(np.mean(top_three))
    return metrics


def _plot_scorecard(
    path: Path,
    *,
    surrogate_audit: dict[str, float],
    evaluations: dict[str, dict[str, np.ndarray]],
    measured_pipeline_seconds: float,
    optimization_seconds: float,
    equivalent_worlds: int,
    estimated_newton_search_seconds: float,
    sampled_best_objective: float,
    optimized_objective: float,
) -> None:
    """Explain why the surrogate is useful."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    green, blue, orange = "#76b900", "#2474b5", "#ef7d32"
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0))
    fig.suptitle(
        "What the PointNet-conditioned neural physics model contributes",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )

    masked_ranking = 100.0 * surrogate_audit["surface_masked_ranking_accuracy"]
    pointnet_ranking = 100.0 * surrogate_audit["pointnet_ranking_accuracy"]
    geometry_bars = axes[0].bar(
        ("surface\nmasked", "PointNet\nsurface"),
        (masked_ranking, pointnet_ranking),
        color=("#9aa4ad", blue),
        width=0.58,
    )
    axes[0].bar_label(
        geometry_bars,
        labels=(f"{masked_ranking:.0f}%", f"{pointnet_ranking:.0f}%"),
        padding=4,
        fontsize=14,
        fontweight="bold",
    )
    masked_top3 = 100.0 * surrogate_audit["surface_masked_pose_top3_recall"]
    pointnet_top3 = 100.0 * surrogate_audit["pointnet_pose_top3_recall"]
    axes[0].set(
        title=(
            "1. PointNet reads local object geometry\n"
            f"best-pose top-3 recall: {masked_top3:.0f}% -> {pointnet_top3:.0f}%"
        ),
        ylabel="correctly ordered held-out gripper designs",
        ylim=(0.0, 105.0),
    )

    timing_values = (
        max(optimization_seconds, 1.0e-4),
        max(estimated_newton_search_seconds, 1.0e-4),
    )
    timing_bars = axes[1].bar(
        ("PhysicsNeMo\nsurrogate", "Newton replay\n(estimated)"),
        timing_values,
        color=(blue, orange),
        width=0.58,
    )
    axes[1].set_yscale("log")
    axes[1].bar_label(
        timing_bars,
        labels=(
            f"{optimization_seconds:.2f} s",
            f"{estimated_newton_search_seconds / 3600.0:.1f} h",
        ),
        padding=5,
        fontsize=13,
        fontweight="bold",
    )
    search_ratio = estimated_newton_search_seconds / max(optimization_seconds, 1.0e-8)
    axes[1].set(
        title=(
            "2. Smooth gradients make the inner loop practical\n"
            f"{equivalent_worlds:,} queries; {search_ratio:,.0f}x search-stage ratio*"
        ),
        ylabel="search time [s, log scale]",
        ylim=(0.5, max(timing_values) * 8.0),
    )

    sampled_best = sampled_best_objective
    sampled_valid = np.isfinite(sampled_best)
    if sampled_valid:
        codesign_improvement = 100.0 * (
            1.0 - optimized_objective / max(sampled_best, 1.0e-8)
        )
        comparison = axes[2].bar(
            ("best measured\nincumbent", "verified neural\nproposal"),
            (sampled_best, optimized_objective),
            color=(orange, green),
            width=0.58,
        )
        axes[2].bar_label(
            comparison,
            labels=(f"{sampled_best:.3f}", f"{optimized_objective:.3f}"),
            padding=4,
            fontsize=13,
            fontweight="bold",
        )
        selection_title = (
            f"3. Newton accepts a {codesign_improvement:.0f}% better design\n"
            if codesign_improvement > 0.0
            else "3. Newton retains the measured incumbent\n"
        )
        objective_limit = max(sampled_best, optimized_objective, 1.0e-3)
    else:
        proposal = axes[2].bar(
            (1,),
            (optimized_objective,),
            color=(green,),
            width=0.58,
        )
        axes[2].bar_label(
            proposal,
            labels=(f"{optimized_objective:.3f}",),
            padding=4,
            fontsize=13,
            fontweight="bold",
        )
        axes[2].set_xticks(
            (0, 1),
            ("measured incumbent", "verified neural\nproposal"),
        )
        objective_limit = max(optimized_objective, 1.0e-3)
        axes[2].text(
            0,
            0.55 * objective_limit,
            "failed\nvalidation",
            color=orange,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        selection_title = "3. Newton accepts the verified neural proposal\n"
    baseline_successes = int(evaluations["baseline"]["success"].sum())
    optimized_successes = int(evaluations["optimized"]["success"].sum())
    axes[2].set(
        title=(
            selection_title + f"{baseline_successes}/{len(scene.OBJECTS)} -> "
            f"{optimized_successes}/{len(scene.OBJECTS)} retained; "
            f"full pipeline {measured_pipeline_seconds / 60.0:.1f} min"
        ),
        ylabel="Newton performance + simplicity\n(lower is better)",
        ylim=(0.0, 1.40 * objective_limit),
    )

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(axis.get_title(), pad=14)
    axes[1].text(
        0.5,
        -0.23,
        "*Newton replay uses measured throughput; it is not a timed brute-force run.",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8.5,
        color="#52606d",
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.22,
        top=0.76,
        wspace=0.34,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_report(
    *,
    args: argparse.Namespace,
    dataset: dict[str, np.ndarray],
    training,
    learning_seconds: float,
    optimization_seconds: float,
    candidate_selection_seconds: float,
    verification_seconds: float,
    measured_pipeline_seconds: float,
    selected_training_restart: int,
    finalist_sources: np.ndarray,
    refinement_accepted: bool,
    best_unit: np.ndarray,
    sampled_best_objective: float,
    optimized_objective: float,
    surrogate_audit: dict[str, float],
    equivalent_worlds: int,
    selected_poses: np.ndarray,
    evaluations: dict[str, dict[str, np.ndarray]],
) -> str:
    physical = scene.unit_to_design(best_unit)
    parameter_rows = "\n".join(
        f"| `{name}` | {value:.5f} |"
        for name, value in zip(scene.DESIGN_NAMES, physical)
    )
    object_rows = []
    optimized = evaluations["optimized"]
    for index, grasp_object in enumerate(scene.OBJECTS):
        object_rows.append(
            "| "
            f"{grasp_object.name} | {scene.object_split(grasp_object)} | "
            f"{int(selected_poses[index])} | "
            f"{optimized['lift_fraction'][index]:.3f} | "
            f"{optimized['lateral_slip'][index]:.4f} | "
            f"{optimized['rotation_error'][index]:.3f} | "
            f"{int(optimized['success'][index])} |"
        )
    train_success = int(optimized["success"][list(scene.TRAIN_OBJECT_INDICES)].sum())
    validation_success = int(
        optimized["success"][list(scene.VALIDATION_OBJECT_INDICES)].sum()
    )
    holdout_success = int(
        optimized["success"][list(scene.HOLDOUT_OBJECT_INDICES)].sum()
    )
    sampled_best = sampled_best_objective
    if np.isfinite(sampled_best):
        sampled_best_text = f"{sampled_best:.4f}"
        codesign_improvement = (
            f"{100.0 * (1.0 - optimized_objective / max(sampled_best, 1.0e-8)):.1f}%"
        )
    else:
        sampled_best_text = "failed validation"
        codesign_improvement = "not comparable"
    generation_seconds = float(
        dataset.get("newton_generation_seconds", np.asarray((np.nan,)))[0]
    )
    if np.isfinite(generation_seconds):
        newton_world_rate = len(dataset["loss"]) / generation_seconds
        estimated_newton_search_seconds = equivalent_worlds / newton_world_rate
        estimated_pipeline_speedup = estimated_newton_search_seconds / max(
            measured_pipeline_seconds, 1.0e-8
        )
    else:
        estimated_newton_search_seconds = np.nan
        estimated_pipeline_speedup = np.nan
    estimated_search_speedup = estimated_newton_search_seconds / max(
        optimization_seconds, 1.0e-8
    )
    return (
        "# Articulated gripper co-design run\n\n"
        "| metric | value |\n"
        "| --- | ---: |\n"
        f"| fixed Newton training worlds | {len(dataset['loss']):,} |\n"
        f"| sampled physical designs | {args.design_samples} |\n"
        f"| candidate poses per object | {args.pose_count} |\n"
        f"| surrogate pose shortlist | top {args.pose_shortlist} |\n"
        f"| held-out-design pose top-1 accuracy | "
        f"{100.0 * training.validation_pose_accuracy:.1f}% |\n"
        f"| held-out-design pose top-3 recall | "
        f"{100.0 * training.validation_pose_top3_recall:.1f}% |\n"
        f"| held-out design-ranking accuracy with PointNet surface | "
        f"{100.0 * surrogate_audit['pointnet_ranking_accuracy']:.1f}% |\n"
        f"| held-out design-ranking accuracy with surface masked | "
        f"{100.0 * surrogate_audit['surface_masked_ranking_accuracy']:.1f}% |\n"
        f"| held-out pose top-3 recall with surface masked | "
        f"{100.0 * surrogate_audit['surface_masked_pose_top3_recall']:.1f}% |\n"
        f"| held-out lift RMSE | {training.validation_rmse[0]:.4f} |\n"
        f"| held-out slip RMSE [m] | {training.validation_rmse[1]:.4f} |\n"
        f"| held-out rotation RMSE [rad] | {training.validation_rmse[2]:.4f} |\n"
        f"| held-out objective RMSE | {training.validation_loss_rmse:.4f} |\n"
        f"| selected surrogate epoch | {training.best_epoch} |\n"
        f"| selected deterministic training restart | {selected_training_restart} |\n"
        f"| Newton-validated refined candidates | {int(np.sum(finalist_sources == 'refined'))} |\n"
        f"| Newton-validated sampled incumbents | {int(np.sum(finalist_sources == 'sampled'))} |\n"
        f"| refined geometry accepted | {'yes' if refinement_accepted else 'no; sampled incumbent retained'} |\n"
        f"| best sampled performance + simplicity | {sampled_best_text} |\n"
        f"| co-designed performance + simplicity | {optimized_objective:.4f} |\n"
        f"| improvement over sampled search | {codesign_improvement} |\n"
        f"| Newton dataset generation [s] | {generation_seconds:.1f} |\n"
        f"| surrogate training + gradient search [s] | {learning_seconds:.1f} |\n"
        f"| surrogate inner-loop physics queries | {equivalent_worlds:,} |\n"
        f"| gradient search [s] | {optimization_seconds:.2f} |\n"
        f"| estimated direct-replay / gradient-search ratio | "
        f"{estimated_search_speedup:.1f}x |\n"
        f"| refined-candidate scoring [s] | {candidate_selection_seconds:.2f} |\n"
        f"| Newton verification [s] | {verification_seconds:.1f} |\n"
        f"| measured full pipeline [s] | {measured_pipeline_seconds:.1f} |\n"
        f"| estimated equivalent Newton search [s] | {estimated_newton_search_seconds:.1f} |\n"
        f"| estimated direct-sweep / full-pipeline ratio | {estimated_pipeline_speedup:.1f}x |\n"
        f"| training-object successes | {train_success}/{len(scene.TRAIN_OBJECT_INDICES)} |\n"
        f"| validation-object successes | {validation_success}/{len(scene.VALIDATION_OBJECT_INDICES)} |\n"
        f"| untouched-test-object successes | {holdout_success}/{len(scene.HOLDOUT_OBJECT_INDICES)} |\n\n"
        "## Optimized physical design\n\n"
        "| parameter | value |\n"
        "| --- | ---: |\n"
        f"{parameter_rows}\n\n"
        "## Newton validation\n\n"
        "| object | split | shortlist-verified pose | retained lift | slip [m] | rotation [rad] | success |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"{chr(10).join(object_rows)}\n"
    )


def run(args: argparse.Namespace) -> str:
    """Generate data, train the surrogate, optimize, and verify in Newton."""
    _seed_reproducibly(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = scene.generate_pose_candidates(args.pose_count, seed=args.seed + 100)
    clouds, contexts = scene.surrogate_feature_table(
        poses,
        num_points=args.point_count,
        seed=args.seed + 200,
    )
    # Recompute the clean, authoritative mass features for the saved npz; the
    # copy embedded in `contexts` is normalized and concatenated with other
    # features, so it is not cleanly reusable here.
    masses = scene.object_mass_features()
    options = scene.SimulationOptions(
        fps=args.fps,
        substeps=args.substeps,
        sim_time=args.sim_time,
        solver_iterations=args.solver_iterations,
        disturbance_force=args.disturbance_force,
    )
    dataset = load_or_generate_dataset(
        args,
        poses=poses,
        clouds=clouds,
        contexts=contexts,
        options=options,
    )

    torch_device = resolve_device(args.torch_device)
    training_started = time.perf_counter()
    best_training_score = float("inf")
    best_training_state = None
    training = None
    selected_training_restart = 0
    training_outcomes = np.stack(
        (
            dataset["lift_fraction"],
            dataset["lateral_slip"],
            dataset["rotation_error"],
            dataset["success"],
        ),
        axis=1,
    )
    for restart in range(args.training_restarts):
        restart_seed = args.seed + 1009 * restart
        _seed_reproducibly(restart_seed)
        candidate_model = models.GraspOutcomeModel(
            scene.DESIGN_DIM,
            point_channels=clouds.shape[-1],
            context_features=contexts.shape[-1],
            point_features=args.point_features,
            hidden_features=args.hidden_features,
        ).to(torch_device)
        candidate_training = models.fit_outcome_model(
            candidate_model,
            point_clouds=dataset["point_clouds"],
            normalized_designs=dataset["normalized_designs"],
            context_features=dataset["context_features"],
            outcomes=training_outcomes,
            design_sample_ids=dataset["design_sample_ids"],
            group_ids=dataset["object_indices"],
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=args.learning_rate,
            patience=args.training_patience,
            seed=args.seed,
        )
        validation_score = float(np.nanmin(candidate_training.validation_history))
        print(
            f"training restart {restart + 1}/{args.training_restarts}: "
            f"held-out design loss {validation_score:.4f} at epoch "
            f"{candidate_training.best_epoch}; pose top-1 "
            f"{100.0 * candidate_training.validation_pose_accuracy:.1f}%, "
            f"top-3 {100.0 * candidate_training.validation_pose_top3_recall:.1f}%"
        )
        if validation_score < best_training_score:
            best_training_score = validation_score
            best_training_state = {
                name: value.detach().cpu().clone()
                for name, value in candidate_model.state_dict().items()
            }
            training = candidate_training
            selected_training_restart = restart + 1
        del candidate_model
    if best_training_state is None or training is None:
        raise RuntimeError("no surrogate training restart produced a checkpoint")
    model = models.GraspOutcomeModel(
        scene.DESIGN_DIM,
        point_channels=clouds.shape[-1],
        context_features=contexts.shape[-1],
        point_features=args.point_features,
        hidden_features=args.hidden_features,
    ).to(torch_device)
    model.load_state_dict(best_training_state)
    training_seconds = time.perf_counter() - training_started
    print(
        "held-out design RMSE "
        f"[lift={training.validation_rmse[0]:.4f}, "
        f"slip={training.validation_rmse[1]:.4f} m, "
        f"rotation={training.validation_rmse[2]:.4f} rad, "
        f"objective={training.validation_loss_rmse:.4f}]"
    )
    print(
        "held-out design pose selection "
        f"top-1={100.0 * training.validation_pose_accuracy:.1f}%, "
        f"top-3={100.0 * training.validation_pose_top3_recall:.1f}%"
    )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    grouped_loss = grouped_surrogate_loss(
        model,
        clouds=clouds,
        contexts=contexts,
        object_indices=scene.TRAIN_OBJECT_INDICES,
    )
    initial_starts = optimization_starts(
        dataset,
        count=args.optimization_starts,
        pose_count=args.pose_count,
    )
    optimization_started = time.perf_counter()
    optimization = optimize_grouped_design(
        grouped_loss,
        design_space=scene.GRIPPER_DESIGN_SPACE,
        starts=initial_starts,
        steps=args.optimization_steps,
        lr=args.optimization_lr,
        top_k_schedule=((0.0, 5), (0.62, 3), (0.84, 1)),
        trust_radius=args.optimization_radius,
        regularizer=design_regularizer,
        seed=args.seed + 300,
        device=torch_device,
    )
    optimization_seconds = time.perf_counter() - optimization_started

    candidate_selection_started = time.perf_counter()
    (
        refined_designs,
        refined_surrogate_scores,
        refined_step_indices,
        refined_trajectory_indices,
    ) = _select_refined_candidates(
        model,
        grouped_loss=grouped_loss,
        optimization=optimization,
        count=args.finalists,
    )
    sampled_designs, sampled_incumbent_scores = _sampled_incumbents(
        model,
        dataset,
        pose_count=args.pose_count,
        shortlist_count=args.pose_shortlist,
        count=args.sampled_finalists,
    )
    candidate_selection_seconds = time.perf_counter() - candidate_selection_started
    learning_seconds = (
        training_seconds + optimization_seconds + candidate_selection_seconds
    )
    finalist_designs = np.concatenate((refined_designs, sampled_designs), axis=0)
    finalist_sources = np.asarray(
        ("refined",) * len(refined_designs) + ("sampled",) * len(sampled_designs)
    )
    finalist_validation_started = time.perf_counter()
    (
        _,
        finalist_metrics,
        finalist_scores,
        finalist_verified_poses,
    ) = validate_finalists(
        model,
        finalist_designs,
        clouds=clouds,
        contexts=contexts,
        poses=poses,
        options=options,
        device=args.newton_device,
        batch_size=args.sim_batch_size,
        shortlist_count=args.pose_shortlist,
    )
    finalist_validation_seconds = time.perf_counter() - finalist_validation_started
    verified_selection = select_verified_design(
        finalist_scores,
        finalist_sources,
        proposal_source="refined",
        incumbent_source="sampled",
        min_improvement=1.0e-7,
    )
    best_refined_index = verified_selection.proposal_index
    best_sampled_index = verified_selection.incumbent_index
    refinement_accepted = verified_selection.accepted
    finalist_index = verified_selection.index
    best_unit = finalist_designs[finalist_index]
    best_physical = scene.unit_to_design(best_unit)
    print(
        f"Newton selected finalist {finalist_index + 1}/{len(finalist_designs)}, "
        f"performance + simplicity {finalist_scores[finalist_index]:.4f}; "
        f"source={finalist_sources[finalist_index]}, "
        f"refinement accepted={'yes' if refinement_accepted else 'no'}"
    )

    evaluations = {}
    neural_selected = {}
    selected = {}
    named_validation_started = time.perf_counter()
    (
        neural_selected["baseline"],
        selected["baseline"],
        evaluations["baseline"],
        _,
    ) = validate_named_design(
        model,
        _baseline_unit(),
        clouds=clouds,
        contexts=contexts,
        poses=poses,
        options=options,
        device=args.newton_device,
        batch_size=args.sim_batch_size,
        shortlist_count=args.pose_shortlist,
    )
    selection_object_indices = (
        *scene.TRAIN_OBJECT_INDICES,
        *scene.VALIDATION_OBJECT_INDICES,
    )
    optimized_selection_neural, _ = select_poses(
        model,
        best_unit,
        clouds=clouds,
        contexts=contexts,
        object_indices=selection_object_indices,
    )
    (
        optimized_holdout_neural,
        optimized_holdout_selected,
        optimized_holdout_metrics,
        _,
    ) = validate_named_design(
        model,
        best_unit,
        clouds=clouds,
        contexts=contexts,
        poses=poses,
        options=options,
        device=args.newton_device,
        batch_size=args.sim_batch_size,
        shortlist_count=args.pose_shortlist,
        object_indices=scene.HOLDOUT_OBJECT_INDICES,
    )
    neural_selected["optimized"] = np.empty(len(scene.OBJECTS), dtype=np.int64)
    selected["optimized"] = np.empty(len(scene.OBJECTS), dtype=np.int64)
    neural_selected["optimized"][list(selection_object_indices)] = (
        optimized_selection_neural
    )
    neural_selected["optimized"][list(scene.HOLDOUT_OBJECT_INDICES)] = (
        optimized_holdout_neural
    )
    selected["optimized"][list(selection_object_indices)] = finalist_verified_poses[
        finalist_index
    ]
    selected["optimized"][list(scene.HOLDOUT_OBJECT_INDICES)] = (
        optimized_holdout_selected
    )
    evaluations["optimized"] = {}
    for key, values in finalist_metrics.items():
        selection_values = values[finalist_index]
        holdout_values = optimized_holdout_metrics[key]
        combined = np.empty(
            (len(scene.OBJECTS), *selection_values.shape[1:]),
            dtype=selection_values.dtype,
        )
        combined[list(selection_object_indices)] = selection_values
        combined[list(scene.HOLDOUT_OBJECT_INDICES)] = holdout_values
        evaluations["optimized"][key] = combined
    named_validation_seconds = time.perf_counter() - named_validation_started
    verification_seconds = finalist_validation_seconds + named_validation_seconds
    for label in ("baseline", "optimized"):
        print(
            f"{label}: {int(evaluations[label]['success'].sum())}/{len(scene.OBJECTS)} "
            "successful objects"
        )
    sampled_best_objective = float(finalist_scores[best_sampled_index])
    optimized_selection_objective = float(finalist_scores[finalist_index])
    sampled_summary = (
        f"{sampled_best_objective:.4f}"
        if np.isfinite(sampled_best_objective)
        else "failed validation"
    )
    print(
        "Newton performance + simplicity: "
        f"best sampled={sampled_summary}, "
        f"co-designed={optimized_selection_objective:.4f}"
    )
    generation_seconds = float(
        dataset.get("newton_generation_seconds", np.asarray((np.nan,)))[0]
    )
    equivalent_worlds = (
        (args.optimization_steps + 1)
        * args.optimization_starts
        * len(scene.TRAIN_OBJECT_INDICES)
        * args.pose_count
    )
    if np.isfinite(generation_seconds):
        newton_world_rate = len(dataset["loss"]) / generation_seconds
        estimated_newton_search_seconds = equivalent_worlds / newton_world_rate
        measured_pipeline_seconds = (
            generation_seconds + learning_seconds + verification_seconds
        )
        estimated_pipeline_speedup = estimated_newton_search_seconds / max(
            measured_pipeline_seconds, 1.0e-8
        )
        print(
            "co-design timing: measured full pipeline "
            f"{measured_pipeline_seconds:.1f} s, including the fixed dataset and "
            f"{verification_seconds:.1f} s of Newton verification; an equivalent "
            f"direct Newton trajectory sweep is estimated at "
            f"{estimated_newton_search_seconds:.1f} s "
            f"({estimated_pipeline_speedup:.1f}x ratio)"
        )
    else:
        estimated_newton_search_seconds = np.nan
        measured_pipeline_seconds = learning_seconds + verification_seconds
        estimated_pipeline_speedup = np.nan

    surrogate_audit = _surrogate_input_audit(
        model,
        dataset=dataset,
        training=training,
    )
    print(
        "PointNet surface sensitivity on held-out designs: "
        f"ranking {100.0 * surrogate_audit['surface_masked_ranking_accuracy']:.1f}% "
        f"-> {100.0 * surrogate_audit['pointnet_ranking_accuracy']:.1f}%, "
        f"pose top-3 "
        f"{100.0 * surrogate_audit['surface_masked_pose_top3_recall']:.1f}% "
        f"-> {100.0 * surrogate_audit['pointnet_pose_top3_recall']:.1f}%"
    )
    report = _format_report(
        args=args,
        dataset=dataset,
        training=training,
        learning_seconds=learning_seconds,
        optimization_seconds=optimization_seconds,
        candidate_selection_seconds=candidate_selection_seconds,
        verification_seconds=verification_seconds,
        measured_pipeline_seconds=measured_pipeline_seconds,
        selected_training_restart=selected_training_restart,
        finalist_sources=finalist_sources,
        refinement_accepted=refinement_accepted,
        best_unit=best_unit,
        sampled_best_objective=sampled_best_objective,
        optimized_objective=optimized_selection_objective,
        surrogate_audit=surrogate_audit,
        equivalent_worlds=equivalent_worlds,
        selected_poses=selected["optimized"],
        evaluations=evaluations,
    )
    (output_dir / "gripper_design_report.md").write_text(report, encoding="utf-8")
    _plot_scorecard(
        output_dir / "gripper_design.png",
        surrogate_audit=surrogate_audit,
        evaluations=evaluations,
        measured_pipeline_seconds=measured_pipeline_seconds,
        optimization_seconds=optimization_seconds,
        equivalent_worlds=equivalent_worlds,
        estimated_newton_search_seconds=estimated_newton_search_seconds,
        sampled_best_objective=sampled_best_objective,
        optimized_objective=optimized_selection_objective,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "design_dim": scene.DESIGN_DIM,
            "design_space_fingerprint": scene.GRIPPER_DESIGN_SPACE.fingerprint,
            "object_library_fingerprint": scene.OBJECT_LIBRARY_FINGERPRINT,
            "point_channels": clouds.shape[-1],
            "context_features": contexts.shape[-1],
            "point_features": args.point_features,
            "hidden_features": args.hidden_features,
            "max_loss": model.max_loss,
            "best_unit_design": best_unit,
            "best_physical_design": best_physical,
            "validation_rmse": training.validation_rmse,
            "validation_loss_rmse": training.validation_loss_rmse,
            "validation_pose_accuracy": training.validation_pose_accuracy,
            "validation_pose_top3_recall": training.validation_pose_top3_recall,
            "learning_seconds": learning_seconds,
            "training_seconds": training_seconds,
            "optimization_seconds": optimization_seconds,
            "candidate_selection_seconds": candidate_selection_seconds,
            "verification_seconds": verification_seconds,
            "measured_pipeline_seconds": measured_pipeline_seconds,
            "selected_training_restart": selected_training_restart,
            "refinement_accepted": refinement_accepted,
            "surrogate_audit": surrogate_audit,
        },
        output_dir / "gripper_surrogate.pt",
    )
    np.savez_compressed(
        output_dir / "gripper_design_data.npz",
        best_unit_design=best_unit,
        best_physical_design=best_physical,
        design_names=np.asarray(scene.DESIGN_NAMES),
        design_space_fingerprint=np.asarray((scene.GRIPPER_DESIGN_SPACE.fingerprint,)),
        object_library_fingerprint=np.asarray((scene.OBJECT_LIBRARY_FINGERPRINT,)),
        object_names=np.asarray([grasp_object.name for grasp_object in scene.OBJECTS]),
        object_splits=np.asarray(
            [scene.object_split(grasp_object) for grasp_object in scene.OBJECTS]
        ),
        poses=np.stack(
            [[pose.as_array() for pose in object_poses] for object_poses in poses]
        ),
        point_clouds=clouds,
        context_features=contexts,
        mass_features=masses,
        neural_selected_pose_indices=neural_selected["optimized"],
        selected_pose_indices=selected["optimized"],
        baseline_selected_pose_indices=selected["baseline"],
        optimization_designs=optimization.designs,
        optimization_losses=optimization.losses,
        optimization_history=optimization.history,
        optimization_design_history=optimization.design_history,
        optimization_top_k=optimization.top_k_history,
        finalist_scores=finalist_scores,
        finalist_designs=finalist_designs,
        finalist_sources=finalist_sources,
        refined_surrogate_scores=refined_surrogate_scores,
        refined_step_indices=refined_step_indices,
        refined_trajectory_indices=refined_trajectory_indices,
        sampled_incumbent_scores=sampled_incumbent_scores,
        sampled_incumbent_unit_design=finalist_designs[best_sampled_index],
        sampled_incumbent_physical_design=scene.unit_to_design(
            finalist_designs[best_sampled_index]
        ),
        sampled_incumbent_verified_pose_indices=finalist_verified_poses[
            best_sampled_index
        ],
        selected_refined_finalist_index=np.asarray((best_refined_index,)),
        selected_sampled_finalist_index=np.asarray((best_sampled_index,)),
        finalist_verified_pose_indices=finalist_verified_poses,
        train_history=training.train_history,
        validation_history=training.validation_history,
        validation_rmse=training.validation_rmse,
        validation_loss_rmse=np.asarray((training.validation_loss_rmse,)),
        validation_pose_accuracy=np.asarray((training.validation_pose_accuracy,)),
        validation_pose_top3_recall=np.asarray((training.validation_pose_top3_recall,)),
        newton_generation_seconds=np.asarray((generation_seconds,)),
        learning_seconds=np.asarray((learning_seconds,)),
        training_seconds=np.asarray((training_seconds,)),
        optimization_seconds=np.asarray((optimization_seconds,)),
        candidate_selection_seconds=np.asarray((candidate_selection_seconds,)),
        verification_seconds=np.asarray((verification_seconds,)),
        measured_pipeline_seconds=np.asarray((measured_pipeline_seconds,)),
        estimated_newton_search_seconds=np.asarray((estimated_newton_search_seconds,)),
        estimated_pipeline_speedup=np.asarray((estimated_pipeline_speedup,)),
        sampled_best_objective=np.asarray((sampled_best_objective,)),
        optimized_selection_objective=np.asarray((optimized_selection_objective,)),
        refinement_accepted=np.asarray((refinement_accepted,)),
        selected_training_restart=np.asarray((selected_training_restart,)),
        equivalent_surrogate_queries=np.asarray((equivalent_worlds,)),
        **{
            f"surrogate_audit_{name}": np.asarray((value,))
            for name, value in surrogate_audit.items()
        },
        **{f"finalist_metrics_{key}": value for key, value in finalist_metrics.items()},
        **{
            f"{label.replace(' ', '_')}_{key}": value
            for label, metrics in evaluations.items()
            for key, value in metrics.items()
        },
    )
    return report


def _baseline_unit() -> np.ndarray:
    return scene.design_to_unit(scene.baseline_design())
