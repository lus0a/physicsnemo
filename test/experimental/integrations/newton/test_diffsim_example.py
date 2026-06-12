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

"""Focused tests for the Newton rollout-BPTT cart-pole example."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


def _cartpole_module(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("newton")
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "examples" / "newton" / "diffsim"))
    return importlib.import_module("example_diffsim_cartpole_bptt")


def test_force_sequences_are_bounded_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cartpole = _cartpole_module(monkeypatch)
    forces = cartpole.sample_force_sequences(4, 13, seed=7)
    repeated = cartpole.sample_force_sequences(4, 13, seed=7)

    torch.testing.assert_close(forces, repeated)
    assert forces.shape == (4, 13)
    assert bool((forces.abs() <= cartpole.FORCE_LIMIT).all())


def test_surrogate_inputs_use_initial_state_and_per_frame_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cartpole = _cartpole_module(monkeypatch)
    states = torch.randn(3, 6, cartpole.STATE_DIM)
    forces = torch.randn(3, 5)
    batch = cartpole.TeacherBatch(states=states, parameters=forces)

    initial_state, step_inputs = cartpole.surrogate_inputs(forces, batch)

    torch.testing.assert_close(initial_state, states[:, 0])
    torch.testing.assert_close(step_inputs, forces.unsqueeze(-1))
