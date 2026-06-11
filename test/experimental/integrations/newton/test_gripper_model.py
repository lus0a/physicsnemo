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

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_gripper_model_module():
    path = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "newton"
        / "gripper"
        / "gripper_model.py"
    )
    spec = importlib.util.spec_from_file_location("gripper_model_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nonfinite_rows_remove_the_complete_design_group() -> None:
    module = _load_gripper_model_module()
    design_ids = np.asarray((0, 0, 0, 1, 1, 1, 2, 2, 2))
    finite_rows = np.asarray((True, False, True, True, True, True, True, True, False))

    keep, incomplete = module._complete_finite_design_mask(design_ids, finite_rows)

    np.testing.assert_array_equal(incomplete, (0, 2))
    np.testing.assert_array_equal(
        keep, (False, False, False, True, True, True) + (False,) * 3
    )


def test_grasp_outcome_model_forward_returns_bounded_outcomes() -> None:
    module = _load_gripper_model_module()
    torch.manual_seed(0)
    design_dim, context_features = 4, 3
    model = module.GraspOutcomeModel(
        design_dim,
        context_features=context_features,
        point_features=8,
        hidden_features=16,
        max_loss=10.0,
    )
    batch, num_points = 2, 16
    points = torch.randn(batch, num_points, 3)
    normalized_designs = torch.randn(batch, design_dim)
    context = torch.randn(batch, context_features)

    outcomes = model(points, normalized_designs, context)

    # [lift, slip, rotation, success, log1p(clipped Newton loss)].
    assert outcomes.shape == (batch, 5)
    assert torch.isfinite(outcomes).all()
    lift, slip, rotation, success, log_loss = outcomes.unbind(dim=-1)
    assert ((lift >= 0.0) & (lift <= 1.15)).all()
    assert ((slip >= 0.0) & (slip <= 0.18)).all()
    assert ((rotation >= 0.0) & (rotation <= float(torch.pi))).all()
    assert ((success >= 0.0) & (success <= 1.0)).all()
    assert (log_loss >= 0.0).all()

    # The two public losses consume the outcome tensor and stay finite.
    assert module.outcome_loss_torch(outcomes).shape == (batch,)
    assert torch.isfinite(module.surrogate_loss_torch(outcomes)).all()

    with pytest.raises(ValueError, match="max_loss"):
        module.GraspOutcomeModel(design_dim, max_loss=0.0)
