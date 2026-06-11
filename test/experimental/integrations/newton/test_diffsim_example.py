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

"""Focused tests for the Newton differentiable-ball benchmark setup."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _diffsim_module(monkeypatch: pytest.MonkeyPatch, name: str):
    pytest.importorskip("newton")
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "examples" / "newton" / "diffsim"))
    return importlib.import_module(name)


def test_feasible_sobol_launches_are_bounded_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ball = _diffsim_module(monkeypatch, "ball_problem")
    starts = ball.feasible_sobol_launches(16, seed=7)
    repeated = ball.feasible_sobol_launches(16, seed=7)

    np.testing.assert_array_equal(starts, repeated)
    np.testing.assert_array_equal(starts[0], ball.LAUNCH_MEAN)
    assert np.all((starts[:, 1] >= ball.LAUNCH_Y_CLIP[0]))
    assert np.all((starts[:, 1] <= ball.LAUNCH_Y_CLIP[1]))
    assert np.all((starts[:, 2] >= ball.LAUNCH_Z_CLIP[0]))
    assert np.all((starts[:, 2] <= ball.LAUNCH_Z_CLIP[1]))


def test_benchmark_selects_all_targets_before_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _diffsim_module(monkeypatch, "analyze_diffsim_optimizers")
    candidates = iter(
        [
            np.array([0.2, 0.0, 0.0], np.float32),
            np.array([1.2, 0.0, 0.0], np.float32),
            np.array([1.5, 0.0, 0.0], np.float32),
        ]
    )
    monkeypatch.setattr(
        benchmark.ball,
        "reachable_target",
        lambda _env, _rng, _steps: next(candidates),
    )
    args = SimpleNamespace(
        seed=123,
        steps=36,
        benchmark_targets=2,
        min_nominal_miss=1.0,
    )

    targets, attempts = benchmark._select_benchmark_targets(
        object(), args, np.zeros(3, np.float32)
    )

    assert attempts == 3
    np.testing.assert_allclose(targets, [[1.2, 0.0, 0.0], [1.5, 0.0, 0.0]])
