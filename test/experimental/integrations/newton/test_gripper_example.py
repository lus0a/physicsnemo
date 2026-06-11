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

"""Focused tests for the Newton gripper example's selection contract."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_gripper_module(name: str):
    pytest.importorskip("newton")
    directory = Path(__file__).resolve().parents[4] / "examples" / "newton" / "gripper"
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(directory))


def test_validation_gate_rejects_failed_candidates() -> None:
    workflow = _load_gripper_module("gripper_workflow")
    scores = workflow._apply_validation_gate(
        np.asarray([0.4, 0.1, 0.3]),
        np.asarray([[1, 1], [1, 0], [1, 1]]),
    )
    assert np.array_equal(scores[[0, 2]], np.asarray([0.4, 0.3]))
    assert np.isinf(scores[1])
    assert int(np.argmin(scores)) == 2


def test_validation_gate_refuses_an_all_failed_shortlist() -> None:
    workflow = _load_gripper_module("gripper_workflow")
    with pytest.raises(RuntimeError, match="no finalist succeeded"):
        workflow._apply_validation_gate(
            np.asarray([0.1, 0.2]),
            np.asarray([[1, 0], [0, 1]]),
        )


def test_gripper_scene_builds_with_supported_newton_joint_arguments() -> None:
    scene = _load_gripper_module("gripper_scene")
    poses = scene.generate_pose_candidates(1, seed=7)
    metrics = scene.evaluate_designs(
        scene.baseline_design()[None],
        np.asarray([0]),
        np.asarray([0]),
        poses=poses,
        options=scene.SimulationOptions(
            fps=10.0,
            substeps=1,
            sim_time=0.1,
            solver_iterations=1,
        ),
        device="cpu",
    )
    assert metrics["loss"].shape == (1,)


def test_gripper_trainer_accepts_four_physical_target_columns() -> None:
    models = _load_gripper_module("gripper_model")
    rng = np.random.default_rng(7)
    model = models.GraspOutcomeModel(
        2,
        point_channels=3,
        context_features=1,
        point_features=16,
        hidden_features=16,
    )
    result = models.fit_outcome_model(
        model,
        point_clouds=rng.normal(size=(4, 8, 3)).astype(np.float32),
        normalized_designs=rng.uniform(size=(4, 2)).astype(np.float32),
        context_features=rng.normal(size=(4, 1)).astype(np.float32),
        outcomes=np.asarray(
            [
                [0.8, 0.01, 0.10, 1.0],
                [0.7, 0.02, 0.15, 1.0],
                [0.6, 0.03, 0.20, 0.0],
                [0.9, 0.01, 0.05, 1.0],
            ],
            dtype=np.float32,
        ),
        design_sample_ids=np.asarray([0, 0, 1, 1]),
        group_ids=np.asarray([0, 1, 0, 1]),
        epochs=1,
        batch_size=2,
        patience=1,
        design_noise=0.0,
        point_noise=0.0,
        seed=7,
    )
    assert result.validation_rmse.shape == (3,)


def test_gripper_dataset_cache_key_tracks_simulation_inputs() -> None:
    workflow = _load_gripper_module("gripper_workflow")
    scene = _load_gripper_module("gripper_scene")
    args = SimpleNamespace(design_samples=8, seed=31)
    poses = scene.generate_pose_candidates(4, seed=131)
    options = scene.SimulationOptions()

    key = workflow._dataset_cache_key(args, poses=poses, options=options)
    assert key == workflow._dataset_cache_key(args, poses=poses, options=options)
    assert key != workflow._dataset_cache_key(
        args,
        poses=poses,
        options=scene.SimulationOptions(disturbance_force=1.0),
    )
    assert key != workflow._dataset_cache_key(
        SimpleNamespace(design_samples=8, seed=32),
        poses=poses,
        options=options,
    )


def test_gripper_surrogate_features_normalize_each_point_set() -> None:
    scene = _load_gripper_module("gripper_scene")
    poses = scene.generate_pose_candidates(2, seed=7)

    point_clouds, contexts = scene.surrogate_feature_table(
        poses,
        num_points=32,
        seed=7,
    )

    assert point_clouds.shape == (len(scene.OBJECTS), 2, 32, 3)
    assert contexts.shape == (len(scene.OBJECTS), 2, 11)
    np.testing.assert_allclose(point_clouds.mean(axis=-2), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        np.linalg.norm(point_clouds, axis=-1).max(axis=-1),
        1.0,
        atol=1.0e-6,
    )


def test_gripper_scorecard_handles_failed_sampled_incumbent(tmp_path: Path) -> None:
    workflow = _load_gripper_module("gripper_workflow")
    scene = _load_gripper_module("gripper_scene")
    success = np.ones(len(scene.OBJECTS), dtype=np.float32)

    workflow._plot_scorecard(
        tmp_path / "scorecard.png",
        surrogate_audit={
            "surface_masked_ranking_accuracy": 0.5,
            "pointnet_ranking_accuracy": 0.75,
            "surface_masked_pose_top3_recall": 0.5,
            "pointnet_pose_top3_recall": 0.75,
        },
        evaluations={
            "baseline": {"success": success},
            "optimized": {"success": success},
        },
        measured_pipeline_seconds=1.0,
        optimization_seconds=0.1,
        equivalent_worlds=10,
        estimated_newton_search_seconds=1.0,
        sampled_best_objective=float("inf"),
        optimized_objective=0.5,
    )

    assert (tmp_path / "scorecard.png").is_file()
