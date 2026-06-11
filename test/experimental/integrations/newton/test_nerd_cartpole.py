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

"""Reference-scene checks for the NeRD Cartpole example."""

from pathlib import Path

import pytest
import torch

from physicsnemo.experimental.integrations.newton.nerd import _joint_layout


def _require_accelerated_newton():
    try:
        import newton
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Newton unavailable: {error}")
    if not torch.accelerator.is_available():
        pytest.skip("accelerator unavailable")
    return newton


def _cartpole_model(num_worlds: int):
    newton = _require_accelerated_newton()
    cartpole = newton.ModelBuilder()
    cartpole.default_joint_cfg.armature = 0.01
    cartpole.default_joint_cfg.limit_ke = 1.0e4
    cartpole.default_joint_cfg.limit_kd = 1.0e1
    cartpole.add_urdf(
        str(
            Path(__file__).resolve().parents[4]
            / "examples"
            / "newton"
            / "nerd"
            / "assets"
            / "cartpole_nerd.urdf"
        ),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )
    builder = newton.ModelBuilder()
    builder.replicate(cartpole, num_worlds, spacing=(0.0, 2.0, 0.0))
    device = torch.accelerator.current_accelerator(check_available=True)
    assert device is not None
    model = builder.finalize(device=str(device))
    return model, newton.solvers.SolverFeatherstone(
        model, update_mass_matrix_interval=5
    )


def test_joint_layout_cartpole_classifies_dofs() -> None:
    model, _ = _cartpole_model(4)
    layout = _joint_layout(model)
    assert layout.world_count == 4
    assert layout.dof_q == 2 and layout.dof_qd == 2
    assert layout.continuous_q_mask.tolist() == [False, True]
    assert layout.base_translation_mask.tolist() == [True, False]
    assert layout.root_is_free is False


def test_reference_cartpole_matches_released_scene_dynamics() -> None:
    model, solver = _cartpole_model(1)
    state_0, state_1, control = model.state(), model.state(), model.control()
    state_0.joint_q.zero_()
    state_0.joint_qd.zero_()
    control.joint_f.assign([1500.0, 0.0])
    for _ in range(5):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 300.0)
        state_0, state_1 = state_1, state_0

    assert model.body_mass.numpy().tolist() == pytest.approx([50.0, 2.5])
    # Golden dynamics snapshot of newton.solvers.SolverFeatherstone, captured
    # against Newton 1.2.1 and revalidated against the newton>=1.3.0 lower bound.
    # These values are a deliberate tripwire on the reference solver, so the
    # tight tolerances are intentional: a Newton point-release that perturbs
    # these values is a real change to the released scene dynamics and should
    # be reviewed, then the constants re-captured against the new Newton version.
    assert state_0.joint_q.numpy().tolist() == pytest.approx(
        [0.00493297, -0.00722432], abs=1.0e-6
    )
    assert state_0.joint_qd.numpy().tolist() == pytest.approx(
        [0.4933021, -0.72263074], abs=1.0e-5
    )
