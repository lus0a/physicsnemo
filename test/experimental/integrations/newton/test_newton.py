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

"""Tests for the PhysicsNeMo↔Newton integration.

The integration is deliberately small: a :class:`NewtonEnv` lifecycle, a
zero-copy data bridge, a differentiable rollout, and a reusable surrogate
trainer. These tests cover that surface with lightweight fakes for the
environment plumbing, a tiny real Newton model for the differentiable rollout,
and synthetic tensors for the surrogate utilities.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

import physicsnemo.experimental.integrations.newton.adjoint as newton_adjoint
import physicsnemo.experimental.integrations.newton.visualization as newton_visualization
from physicsnemo.experimental.integrations.newton import (
    BPTTSurrogate,
    DesignSpace,
    DesignSurrogate,
    DesignVariable,
    DifferentiableRollout,
    GroupedDesignResult,
    NewtonComponents,
    NewtonEnv,
    ResidualDynamics,
    TeacherBatch,
    TeacherSample,
    bodies,
    collect_teacher_batch,
    differentiable_rollout,
    field_to_torch,
    grouped_candidate_ranking_loss,
    joints,
    load_example_scene,
    optimize_design,
    optimize_field_in_newton,
    optimize_field_in_newton_multistart,
    optimize_grouped_design,
    particles,
    shortlist_grouped_candidates,
)
from physicsnemo.experimental.integrations.newton.components import (
    _components_from_model,
    _components_from_scene,
)
from physicsnemo.experimental.integrations.newton.data import (
    _assign_value,
    _copy_newton_object,
)
from physicsnemo.experimental.integrations.newton.surrogate import (
    _filter_finite,
    _gradient_alignment_loss,
    _rollout_model,
    _RolloutStats,
    _standardize,
)


# --------------------------------------------------------------------------- #
# Fakes for fast lifecycle tests (no Newton/Warp required)
# --------------------------------------------------------------------------- #
class FakeState:
    def __init__(self) -> None:
        self.particle_q = torch.zeros(3)
        self.particle_qd = torch.zeros(3)
        self.clear_count = 0

    def clear_forces(self) -> None:
        self.clear_count += 1

    def assign(self, other: "FakeState") -> None:
        self.particle_q = other.particle_q.clone()
        self.particle_qd = other.particle_qd.clone()


def test_cpu_capture_guard_is_scoped_to_affected_newton_versions(monkeypatch) -> None:
    monkeypatch.setattr(newton_visualization, "version", lambda _: "1.2.1")
    assert newton_visualization._newton_cpu_capture_is_unsafe()

    monkeypatch.setattr(newton_visualization, "version", lambda _: "1.2.2")
    assert not newton_visualization._newton_cpu_capture_is_unsafe()


class FakeModel:
    device = "cpu"

    def state(self, requires_grad: bool = False) -> FakeState:
        return FakeState()

    def control(self, requires_grad: bool = False) -> SimpleNamespace:
        return SimpleNamespace(delta=torch.ones(3))


class FakeSolver:
    def step(self, state_in, state_out, control, contacts, dt) -> None:
        state_out.particle_qd = state_in.particle_qd.clone()
        state_out.particle_q = state_in.particle_q + state_in.particle_qd * dt


class FakePipeline:
    def __init__(self) -> None:
        self.collide_count = 0

    def contacts(self) -> SimpleNamespace:
        return SimpleNamespace(id=0)

    def collide(self, state, contacts) -> None:
        self.collide_count += 1


def observe_q(state) -> torch.Tensor:
    return field_to_torch(state.particle_q)


def fake_env(**kwargs) -> NewtonEnv:
    components = NewtonComponents(model=FakeModel(), solver=FakeSolver())
    return NewtonEnv(components, observe=observe_q, dt=0.5, **kwargs)


def _skip_without_cuda() -> None:
    try:
        wp.init()
    except Exception as error:  # noqa: BLE001 - report the runtime-specific CUDA init failure.
        pytest.skip(f"Warp CUDA unavailable: {error}")
    if not torch.cuda.is_available() or not any(
        str(d).startswith("cuda") for d in wp.get_devices()
    ):
        pytest.skip("CUDA device unavailable")


def _require_newton():
    """Import Newton or skip; the integration treats it as an optional extra."""
    try:
        import newton
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Newton unavailable: {error}")
    return newton


# --------------------------------------------------------------------------- #
# data bridge
# --------------------------------------------------------------------------- #
def test_field_to_torch_passthrough_and_conversions() -> None:
    tensor = torch.arange(3.0)
    assert field_to_torch(tensor) is tensor
    assert torch.equal(field_to_torch([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert torch.equal(field_to_torch(np.array([1, 2, 3])), torch.tensor([1, 2, 3]))


def test_field_to_torch_dtype_device_and_clone() -> None:
    source = torch.ones(2, dtype=torch.float64)
    out = field_to_torch(source, dtype=torch.float32, clone=True)
    assert out.dtype == torch.float32
    out[0] = 5.0
    assert source[0] == 1.0  # clone did not alias


def test_field_to_torch_warp_zero_copy_cpu() -> None:
    array = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
    view = field_to_torch(array)
    array.assign([4.0, 5.0, 6.0])
    assert torch.equal(view, torch.tensor([4.0, 5.0, 6.0]))  # aliases the Warp buffer


def test_field_to_torch_warp_zero_copy_writeback_cpu() -> None:
    array = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
    view = field_to_torch(array)
    with torch.no_grad():
        view[0] = 5.0
    assert array.numpy()[0] == 5.0  # Torch write reaches the Warp buffer


def test_body_view_component_setters_write_back_into_warp_arrays() -> None:
    body_q = wp.zeros((2, 7), dtype=wp.float32, device="cpu")
    body_qd = wp.zeros((2, 6), dtype=wp.float32, device="cpu")
    view = bodies(SimpleNamespace(body_q=body_q, body_qd=body_qd))
    view.positions = [1.0, 2.0, 3.0]
    view.linear_velocities = [4.0, 5.0, 6.0]
    assert np.allclose(body_q.numpy()[:, :3], [[1, 2, 3], [1, 2, 3]])
    assert np.allclose(body_qd.numpy()[:, :3], [[4, 5, 6], [4, 5, 6]])
    assert np.allclose(body_q.numpy()[:, 3:], 0.0)  # other components preserved


def test_field_to_torch_warp_zero_copy_cuda() -> None:
    _skip_without_cuda()
    array = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cuda")
    view = field_to_torch(array)
    assert view.is_cuda
    array.assign([7.0, 8.0, 9.0])
    assert torch.equal(view.cpu(), torch.tensor([7.0, 8.0, 9.0]))


def test_assign_value_into_warp_torch_numpy() -> None:
    warp_array = wp.zeros(3, dtype=wp.float32, device="cpu")
    _assign_value(warp_array, torch.tensor([1.0, 2.0, 3.0]))
    assert np.allclose(warp_array.numpy(), [1.0, 2.0, 3.0])

    torch_target = torch.zeros(2)
    _assign_value(torch_target, np.array([4.0, 5.0]))
    assert torch.equal(torch_target, torch.tensor([4.0, 5.0]))

    numpy_target = np.zeros(2)
    _assign_value(numpy_target, torch.tensor([6.0, 7.0]))
    assert np.allclose(numpy_target, [6.0, 7.0])


def test_assign_value_writes_cuda_torch_tensor_into_warp_array() -> None:
    # A CUDA Torch source must land in the Warp array on device. wp.array.assign()
    # routes through host NumPy and cannot represent a CUDA tensor, so _assign_value
    # writes through the array's Torch view instead.
    _skip_without_cuda()
    warp_array = wp.zeros(3, dtype=wp.float32, device="cuda")
    _assign_value(warp_array, torch.tensor([1.0, 2.0, 3.0], device="cuda"))
    wp.synchronize_device("cuda")
    assert np.allclose(warp_array.numpy(), [1.0, 2.0, 3.0])

    state = SimpleNamespace(
        particle_q=wp.zeros((2, 3), dtype=wp.float32, device="cuda"),
        particle_qd=wp.zeros((2, 3), dtype=wp.float32, device="cuda"),
    )
    particles(state).velocities = torch.full((2, 3), 5.0, device="cuda")
    wp.synchronize_device("cuda")
    assert np.allclose(state.particle_qd.numpy(), 5.0)


def test_assign_value_and_write_state_fields_support_leaf_tensors() -> None:
    from physicsnemo.experimental.integrations.newton import write_state_fields

    target = torch.zeros(2, requires_grad=True)
    _assign_value(target, torch.ones(2))
    assert torch.equal(target, torch.ones(2))
    state = SimpleNamespace(particle_q=torch.zeros(2, requires_grad=True))
    write_state_fields(state, particle_q=torch.full((2,), 3.0))
    assert torch.equal(state.particle_q, torch.full((2,), 3.0))


def test_copy_newton_object_copies_state() -> None:
    source, target = FakeState(), FakeState()
    source.particle_q = torch.tensor([1.0, 2.0, 3.0])
    _copy_newton_object(target, source)
    assert torch.equal(target.particle_q, torch.tensor([1.0, 2.0, 3.0]))


def test_copy_newton_object_rejects_unsupported_fields() -> None:
    with pytest.raises(TypeError, match="cannot copy Newton field"):
        _copy_newton_object(
            SimpleNamespace(label="target"), SimpleNamespace(label="source")
        )


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #
def test_components_from_scene_reads_attributes() -> None:
    pipeline = FakePipeline()
    scene = SimpleNamespace(
        model=FakeModel(),
        solver=FakeSolver(),
        collision_pipeline=pipeline,
        control="ctrl",
        contacts="ctc",
    )
    components = _components_from_scene(scene)
    assert isinstance(components.model, FakeModel)
    assert components.pipeline is pipeline
    assert components.control == "ctrl"
    assert components.contacts == "ctc"


def test_components_from_scene_requires_model_and_solver() -> None:
    with pytest.raises(AttributeError):
        _components_from_scene(SimpleNamespace(solver=FakeSolver()))
    with pytest.raises(AttributeError):
        _components_from_scene(SimpleNamespace(model=FakeModel()))


def test_components_from_model_keeps_explicit_solver() -> None:
    solver = FakeSolver()
    components = _components_from_model(FakeModel(), solver=solver)
    assert components.solver is solver


# --------------------------------------------------------------------------- #
# environment lifecycle
# --------------------------------------------------------------------------- #
def test_env_from_scene_infers_timing() -> None:
    scene = SimpleNamespace(
        model=FakeModel(),
        solver=FakeSolver(),
        frame_dt=1.0 / 60.0,
        sim_substeps=4,
        sim_steps=12,
    )
    env = NewtonEnv.from_scene(scene, observe=observe_q)
    assert env.substeps == 4
    assert env.dt == pytest.approx(1.0 / 60.0 / 4)
    assert env.horizon == 12


def test_env_from_scene_respects_explicit_timing() -> None:
    scene = SimpleNamespace(
        model=FakeModel(), solver=FakeSolver(), sim_substeps=4, sim_dt=0.1
    )
    env = NewtonEnv.from_scene(scene, observe=observe_q, dt=0.02, substeps=2)
    assert (env.dt, env.substeps) == (0.02, 2)


def test_env_from_scene_preserves_frame_duration_when_only_substeps_change() -> None:
    scene = SimpleNamespace(
        model=FakeModel(), solver=FakeSolver(), sim_substeps=4, sim_dt=0.1
    )
    env = NewtonEnv.from_scene(scene, observe=observe_q, substeps=2)
    assert env.substeps == 2
    assert env.dt == pytest.approx(0.2)
    assert env.frame_dt == pytest.approx(0.4)


def test_env_rejects_invalid_timestep() -> None:
    with pytest.raises(ValueError, match="dt"):
        NewtonEnv(NewtonComponents(model=FakeModel(), solver=FakeSolver()), dt=0.0)


def test_env_from_scene_runs_explicit_collision_pipeline_without_contact_buffer() -> (
    None
):
    pipeline = FakePipeline()
    scene = SimpleNamespace(
        model=FakeModel(),
        solver=FakeSolver(),
        collision_pipeline=pipeline,
    )
    env = NewtonEnv.from_scene(scene, observe=observe_q)
    env.reset()
    env.step()
    assert pipeline.collide_count == 1
    assert env.contacts is not None


def test_env_from_scene_preserves_initial_state_and_supplied_contacts() -> None:
    pipeline = FakePipeline()
    initial = FakeState()
    initial.particle_q = torch.tensor([1.0, 2.0, 3.0])
    contacts = SimpleNamespace(id=1)
    scene = SimpleNamespace(
        model=FakeModel(),
        solver=FakeSolver(),
        collision_pipeline=pipeline,
        state_0=initial,
        contacts=contacts,
    )
    env = NewtonEnv.from_scene(
        scene,
        observe=observe_q,
        collide_each_substep=False,
    )
    env.reset()
    env.step()
    assert torch.equal(env.state.particle_q, initial.particle_q)
    assert env.contacts is contacts
    assert pipeline.collide_count == 0


def test_env_reset_applies_field_values_and_observes() -> None:
    env = fake_env()
    obs = env.reset(particle_qd=torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(env.state.particle_qd, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(obs, env.state.particle_q)


def test_env_reset_allocates_and_restores_episode_control() -> None:
    template = FakeModel().control()
    env = NewtonEnv(
        NewtonComponents(model=FakeModel(), solver=FakeSolver(), control=template),
        observe=observe_q,
    )
    env.reset()
    assert env.control is not template
    env.control.delta.zero_()
    env.reset()
    assert torch.equal(env.control.delta, torch.ones(3))


def test_env_reset_collides_when_requested() -> None:
    pipeline = FakePipeline()
    components = NewtonComponents(
        model=FakeModel(), solver=FakeSolver(), pipeline=pipeline
    )
    env = NewtonEnv(components, observe=observe_q, collide_on_reset=True)
    env.reset()
    assert pipeline.collide_count == 1


def test_env_collide_updates_current_state_contacts() -> None:
    pipeline = FakePipeline()
    components = NewtonComponents(
        model=FakeModel(), solver=FakeSolver(), pipeline=pipeline
    )
    env = NewtonEnv(components, observe=observe_q)
    env.reset()
    env.collide()
    assert pipeline.collide_count == 1
    assert env.contacts is not None


def test_env_rollout_ping_pong_integrates_and_stacks() -> None:
    env = fake_env(substeps=1)
    env.reset(particle_qd=torch.tensor([2.0, 0.0, 0.0]))
    rollout = env.rollout(3)
    assert rollout.observations.shape == (4, 3)  # initial + 3 frames
    # x advances by qd*dt each frame: 0, 1, 2, 3 with qd=2, dt=0.5
    assert torch.allclose(
        rollout.observations[:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0])
    )


def test_env_reset_clears_step_model_history() -> None:
    class StatefulStep:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def __call__(self, state_in, state_out, control, contacts, dt) -> None:
            state_out.assign(state_in)

    step_model = StatefulStep()
    env = fake_env(step_model=step_model)
    env.reset()
    env.reset()
    assert step_model.reset_count == 2


def test_env_step_runs_before_substep_after_force_clear() -> None:
    env = fake_env(substeps=2)
    env.reset()
    calls = []

    def before_substep(state, control, contacts, dt, substep) -> None:
        del control, contacts, dt
        calls.append((state.clear_count, substep))
        state.particle_qd.fill_(1.0)

    observation = env.step(before_substep=before_substep)
    assert calls == [(1, 0), (1, 1)]
    assert torch.equal(observation, torch.ones(3))


def test_env_rollout_keep_states_allocates_per_substep() -> None:
    env = fake_env(substeps=2)
    env.reset(particle_qd=torch.tensor([1.0, 0.0, 0.0]))
    rollout = env.rollout(3, keep_states=True)
    assert len(rollout.states) == 3 * 2 + 1
    assert rollout.observations.shape == (4, 3)


def test_env_rollout_keep_states_forwards_before_substep() -> None:
    env = fake_env(substeps=2)
    calls = []

    def before_substep(state, control, contacts, dt, substep) -> None:
        del state, control, contacts, dt
        calls.append(substep)

    env.rollout(3, keep_states=True, before_substep=before_substep)
    assert calls == [0, 1, 0, 1, 0, 1]


def test_env_rollout_can_skip_initial_observation() -> None:
    env = fake_env(substeps=1)
    env.reset()
    assert env.rollout(2, include_initial=False).observations.shape[0] == 2


def test_env_rollout_uses_default_horizon() -> None:
    env = fake_env(horizon=2)
    assert env.rollout().observations.shape == (3, 3)
    with pytest.raises(ValueError, match="steps is required"):
        fake_env().rollout()


def test_env_empty_rollout_stays_on_model_device() -> None:
    env = NewtonEnv(NewtonComponents(model=FakeModel(), solver=FakeSolver()), horizon=1)
    rollout = env.rollout(0, include_initial=False)
    assert rollout.observations.device.type == "cpu"
    assert rollout.observations.numel() == 0


# --------------------------------------------------------------------------- #
# differentiable rollout (real Newton)
# --------------------------------------------------------------------------- #
def _free_particle_env(device: str = "cpu") -> NewtonEnv:
    newton = _require_newton()

    builder = newton.ModelBuilder()
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, 1.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0
    )
    with wp.ScopedDevice(device):
        model = builder.finalize(requires_grad=True)
    return NewtonEnv.from_model(
        model,
        observe=lambda s: field_to_torch(s.particle_q)[0],
        dt=0.01,
        substeps=1,
        requires_grad=True,
    )


def test_env_rollout_snapshots_real_newton_ping_pong_buffers() -> None:
    env = _free_particle_env()
    rollout = env.rollout(4)
    assert torch.allclose(
        rollout.observations[:, 0],
        torch.tensor([0.0, 0.01, 0.02, 0.03, 0.04]),
        atol=1.0e-6,
    )


def test_env_from_model_auto_constructs_collision_pipeline() -> None:
    newton = _require_newton()

    builder = newton.ModelBuilder()
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 1.0), vel=wp.vec3(), mass=1.0)
    builder.add_ground_plane()
    model = builder.finalize(device="cpu")
    env = NewtonEnv.from_model(model, dt=0.01)
    assert env.pipeline is not None and env.collide_each_substep
    env.reset()
    env.step()
    assert env.contacts is not None
    assert NewtonEnv.from_model(model, dt=0.01, collisions=False).pipeline is None


def test_env_mixed_warp_torch_step_uses_current_cuda_stream() -> None:
    _skip_without_cuda()
    env = _free_particle_env("cuda:0")
    stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(stream):
        observations = env.rollout(4).observations
    stream.synchronize()
    assert torch.allclose(
        observations[:, 0],
        torch.tensor([0.0, 0.01, 0.02, 0.03, 0.04], device="cuda:0"),
        atol=1.0e-6,
    )


@wp.kernel
def _final_x_loss(q: wp.array(dtype=wp.vec3), loss: wp.array(dtype=wp.float32)):  # noqa: F722
    loss[0] = q[0][0]  # the final x position


def test_differentiable_rollout_returns_states_adjoint_loss() -> None:
    env = _free_particle_env()

    def loss_fn(states):
        loss = wp.zeros(
            1, dtype=wp.float32, requires_grad=True, device=env.model.device
        )
        wp.launch(
            _final_x_loss,
            dim=1,
            inputs=[states[-1].particle_q, loss],
            device=env.model.device,
        )
        return loss

    result = differentiable_rollout(env, steps=5, loss_fn=loss_fn, field="particle_qd")
    assert result.observations.shape == (6, 3)
    assert result.adjoint.shape == (1, 3)
    assert result.loss.ndim == 0
    assert (
        not result.loss.requires_grad and not result.observations.requires_grad
    )  # detached
    # d(final x)/d(initial vx) = total time = steps * dt; vy/vz do not affect x.
    assert result.adjoint[0, 0].item() == pytest.approx(5 * 0.01, rel=1e-3)
    assert abs(result.adjoint[0, 1].item()) < 1e-5


def test_differentiable_rollout_requires_gradients() -> None:
    newton = _require_newton()

    builder = newton.ModelBuilder()
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, 1.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0
    )
    model = builder.finalize(requires_grad=False)
    env = NewtonEnv.from_model(
        model, observe=lambda s: field_to_torch(s.particle_q)[0], requires_grad=False
    )

    def loss_fn(states):
        loss = wp.zeros(1, dtype=wp.float32, device=env.model.device)
        wp.launch(
            _final_x_loss,
            dim=1,
            inputs=[states[-1].particle_q, loss],
            device=env.model.device,
        )
        return loss

    with pytest.raises(RuntimeError):
        differentiable_rollout(env, steps=2, loss_fn=loss_fn, field="particle_qd")


# --------------------------------------------------------------------------- #
# surrogate utilities
# --------------------------------------------------------------------------- #
def test_residual_dynamics_default_mlp_is_shaped() -> None:
    model = ResidualDynamics.mlp(state_dim=4, input_dim=2, hidden_dim=8, depth=2)
    assert model(torch.zeros(5, 4), torch.zeros(5, 2)).shape == (5, 4)
    assert model(torch.ones(3, 4), torch.zeros(3, 2)).shape == (3, 4)


def test_residual_dynamics_wraps_any_core_and_adds_residual() -> None:
    # Any nn.Module of the right width can be the core; a zero core proves the
    # wrapper only adds the residual (next_state == state).
    class ZeroCore(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.new_zeros((*x.shape[:-1], 4))

    model = ResidualDynamics(ZeroCore(), state_dim=4, input_dim=2)
    state = torch.randn(6, 4)
    assert torch.equal(model(state, torch.randn(6, 2)), state)


def test_standardize_reduces_sample_and_time_axes() -> None:
    data = torch.randn(8, 5, 3)
    mean, std = _standardize(data)
    assert mean.shape == (1, 1, 3) and std.shape == (1, 1, 3)
    assert (std > 0).all()


def test_rollout_model_produces_horizon_plus_one_states() -> None:
    model = ResidualDynamics.mlp(state_dim=2, input_dim=0, hidden_dim=4, depth=1)
    stats = _RolloutStats(
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, 2),
        torch.zeros(1, 1, 0),
        torch.ones(1, 1, 0),
    )
    out = _rollout_model(
        model, torch.zeros(3, 2), torch.zeros(3, 1, 0), stats, horizon=4
    )
    assert out.shape == (3, 5, 2)


def test_gradient_alignment_loss_zero_for_identical_grads() -> None:
    grad = torch.randn(4, 3)
    assert _gradient_alignment_loss(grad, grad).item() == pytest.approx(0.0, abs=1e-6)
    assert _gradient_alignment_loss(grad, -grad).item() > 1.0


def test_teacher_batch_to_and_head() -> None:
    batch = TeacherBatch(
        torch.zeros(4, 3, 2),
        torch.zeros(4, 2),
        torch.zeros(4, 2),
        torch.zeros(4),
        task_data={"target": torch.arange(4)},
        metadata={"opaque": "kept"},
    )
    head = batch.head(2)
    assert head.states.shape[0] == 2
    assert torch.equal(head.task_data["target"], torch.arange(2))
    assert head.metadata["opaque"] == "kept"
    assert batch.to("cpu").parameters.device.type == "cpu"
    with pytest.raises(ValueError, match="positive"):
        batch.head(0)


def test_collect_teacher_batch_stacks_and_filters_nonfinite() -> None:
    def sample(index: int) -> TeacherSample:
        loss = float("nan") if index == 1 else float(index)
        return TeacherSample(
            states=torch.zeros(3, 2),
            parameters=torch.ones(2) * index,
            adjoints=torch.ones(2),
            loss=loss,
            task_data={"target": torch.tensor([index])},
        )

    batch = collect_teacher_batch(3, sample)
    assert batch.states.shape == (2, 3, 2)  # the NaN sample was dropped
    assert batch.metadata["discarded_nonfinite_samples"] == 1
    assert "teacher_ms_per_sample" in batch.metadata


def test_collect_teacher_batch_rejects_all_nonfinite_samples() -> None:
    with pytest.raises(ValueError, match="all teacher samples were non-finite"):
        collect_teacher_batch(
            1,
            lambda _index: TeacherSample(
                states=torch.zeros(2, 2),
                parameters=torch.zeros(2),
                adjoints=torch.zeros(2),
                loss=float("nan"),
            ),
        )


def test_filter_finite_drops_bad_samples() -> None:
    states = torch.zeros(2, 2, 2)
    states[0, 0, 0] = float("inf")
    batch = TeacherBatch(
        states,
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2),
        task_data={"target": torch.tensor([[1.0], [2.0]])},
    )
    filtered = _filter_finite(batch)
    assert filtered.states.shape[0] == 1
    assert torch.equal(filtered.task_data["target"], torch.tensor([[2.0]]))


def test_teacher_batch_repeat_includes_aligned_metadata() -> None:
    batch = _synthetic_batch(n=2)
    batch.task_data["target"] = torch.tensor([[1.0], [2.0]])
    repeated = batch.repeat(3)
    assert repeated.states.shape[0] == 6
    assert repeated.task_data["target"].flatten().tolist() == [
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
    ]


def _synthetic_batch(n: int = 6, horizon: int = 4, seed: int = 0) -> TeacherBatch:
    g = torch.Generator().manual_seed(seed)
    return TeacherBatch(
        states=torch.randn(n, horizon + 1, 2, generator=g),
        parameters=torch.randn(n, 2, generator=g),
        adjoints=torch.randn(n, 2, generator=g),
        losses=torch.rand(n, generator=g),
        metadata={"teacher_ms_per_sample": 1.0},
    )


def _surrogate() -> BPTTSurrogate:
    return BPTTSurrogate(
        state_dim=2,
        param_dim=2,
        input_dim=2,
        to_inputs=lambda params, batch: (
            batch.states[:, 0, :].to(params),
            params[:, None, :],
        ),
        task_loss=lambda pred, params, batch: (pred[:, -1, :] ** 2).sum(-1),
        hidden_dim=8,
        depth=1,
    )


def test_bptt_surrogate_fit_and_evaluate() -> None:
    surrogate = _surrogate()
    metrics = surrogate.fit(_synthetic_batch(), epochs=3)
    assert {
        "samples",
        "horizon",
        "train_rollout_rmse",
        "train_adjoint_cosine_mean",
    } <= metrics.keys()
    held = surrogate.evaluate(_synthetic_batch(seed=1))
    assert {"rollout_rmse", "adjoint_cosine_mean", "adjoint_cosine_min"} <= held.keys()
    assert held["surrogate_grad_batch_size"] == 6
    assert held["surrogate_grad_ms_per_sample"] == pytest.approx(
        held["surrogate_grad_batch_ms"] / held["surrogate_grad_batch_size"]
    )
    assert held["surrogate_grad_ms"] == held["surrogate_grad_ms_per_sample"]
    assert np.isfinite(held["rollout_rmse"])
    assert surrogate.rollout(torch.zeros(2, 2), _synthetic_batch().head(2)).shape == (
        2,
        5,
        2,
    )


def test_bptt_surrogate_requires_fit_and_valid_callback_shapes() -> None:
    surrogate = _surrogate()
    with pytest.raises(RuntimeError, match="fit the surrogate"):
        surrogate.evaluate(_synthetic_batch())

    bad_loss = BPTTSurrogate(
        state_dim=2,
        param_dim=2,
        to_inputs=lambda params, batch: batch.states[:, 0],
        task_loss=lambda pred, params, batch: pred.sum(),
        hidden_dim=8,
        depth=1,
    )
    with pytest.raises(ValueError, match="one value per sample"):
        bad_loss.fit(_synthetic_batch(), epochs=1)


def test_residual_dynamics_validates_inputs_shape() -> None:
    model = ResidualDynamics(torch.nn.Linear(3, 2), state_dim=2, input_dim=1)
    with pytest.raises(ValueError, match="inputs"):
        model(torch.zeros(4, 2))
    with pytest.raises(ValueError, match="batch dimensions"):
        model(torch.zeros(4, 2), torch.zeros(3, 1))


def test_bptt_surrogate_optimize_tracks_best() -> None:
    surrogate = _surrogate()
    surrogate.fit(_synthetic_batch(), epochs=2)
    plan = surrogate.optimize(_synthetic_batch(seed=2), samples=2, steps=5)
    assert plan["best_params"].shape == (2, 2)
    assert plan["best_task_losses"].shape == (2,)
    assert plan["best_task_loss"] <= plan["initial_task_loss"] + 1e-6
    assert len(plan["history"]) == 6  # step 0 + 5 steps


def test_bptt_surrogate_optimize_multistart_selects_best_candidate() -> None:
    surrogate = _surrogate()
    surrogate.fit(_synthetic_batch(), epochs=2)
    plan = surrogate.optimize_multistart(
        _synthetic_batch(seed=2).head(1), starts=4, steps=5, seed=7
    )
    assert plan["best_params"].shape == (1, 2)
    assert plan["candidate_best_params"].shape == (4, 2)
    assert plan["starts"] == 4
    assert plan["best_task_loss"] == pytest.approx(
        float(np.min(plan["candidate_best_task_losses"]))
    )
    np.testing.assert_allclose(
        plan["best_params"],
        plan["candidate_best_params"][plan["best_index"] : plan["best_index"] + 1],
    )


def test_bptt_surrogate_optimize_clips_initial_parameters() -> None:
    surrogate = _surrogate()
    surrogate.fit(_synthetic_batch(), epochs=2)
    task = _synthetic_batch(seed=2).head(1)
    _, mean, std = surrogate._require_fitted()

    direct = surrogate.optimize(
        task,
        initial_params=np.asarray([[20.0, 20.0]], np.float32),
        steps=0,
        z_clip=0.5,
    )
    direct_z = (torch.from_numpy(direct["initial_params"]) - mean.cpu()) / std.cpu()
    assert bool((direct_z.abs() <= 0.5 + 1.0e-6).all())
    np.testing.assert_allclose(direct["best_params"], direct["initial_params"])

    task.parameters.fill_(20.0)
    plan = surrogate.optimize_multistart(task, starts=4, steps=0, z_clip=0.5, seed=7)
    candidate_z = (
        torch.from_numpy(plan["candidate_initial_params"]) - mean.cpu()
    ) / std.cpu()
    assert bool((candidate_z.abs() <= 0.5 + 1.0e-6).all())


def test_bptt_surrogate_optimize_multistart_accepts_explicit_starts() -> None:
    surrogate = _surrogate()
    surrogate.fit(_synthetic_batch(), epochs=2)
    task = _synthetic_batch(seed=2).head(1)
    _, mean, std = surrogate._require_fitted()
    normalized_starts = torch.tensor(
        [[0.0, 0.0], [1.0, -1.0], [-0.5, 0.5]], dtype=torch.float32
    )
    starts = (mean.cpu() + normalized_starts * std.cpu()).numpy()
    plan = surrogate.optimize_multistart(
        task,
        starts=3,
        initial_params=starts,
        steps=0,
        seed=7,
    )
    np.testing.assert_allclose(plan["candidate_initial_params"], starts)

    with pytest.raises(ValueError, match="exactly 4 starts"):
        surrogate.optimize_multistart(
            task,
            starts=4,
            initial_params=starts,
            steps=0,
        )


def test_bptt_surrogate_validate_in_newton() -> None:
    surrogate = _surrogate()
    surrogate.fit(_synthetic_batch(), epochs=1)
    plan = {
        "initial_params": np.array([[2.0, 0.0]], np.float32),
        "best_params": np.array([[0.5, 0.0]], np.float32),
    }
    report = surrogate.validate_in_newton(
        plan, newton_loss=lambda p: float((p**2).sum())
    )
    assert report["newton_improved"] is True
    assert report["newton_initial_loss"] == pytest.approx(4.0)
    assert report["newton_optimized_loss"] == pytest.approx(0.25)


def test_optimize_field_in_newton_multistart_returns_best_and_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_optimize(_env, *, initial, optimization_steps, **_kwargs):
        loss = float(np.square(initial).sum())
        return {
            "best_params": np.asarray(initial),
            "best_loss": loss,
            "initial_loss": loss,
            "optimization_steps": optimization_steps,
            "total_ms": 1.0,
            "solver_evals": optimization_steps + 1,
            "history": [],
        }

    monkeypatch.setattr(newton_adjoint, "optimize_field_in_newton", fake_optimize)
    result = optimize_field_in_newton_multistart(
        object(),
        loss_fn=lambda _states: None,
        initials=np.array([[2.0, 0.0], [0.5, 0.0]], np.float32),
        optimization_steps=3,
    )
    assert result["best_index"] == 1
    assert result["best_loss"] == pytest.approx(0.25)
    assert result["per_start_solver_evals"] == 4
    assert result["solver_evals"] == 8
    assert len(result["runs"]) == 2


def test_optimize_field_defaults_to_current_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = fake_env()
    env.reset(particle_qd=torch.tensor([3.0, 2.0, 1.0]))

    def fake_rollout(environment, *, field, **_kwargs):
        value = getattr(environment.state, field)
        return DifferentiableRollout(
            observations=torch.empty(0),
            adjoint=torch.zeros_like(value),
            loss=torch.tensor(float(value.square().sum())),
        )

    monkeypatch.setattr(newton_adjoint, "differentiable_rollout", fake_rollout)
    result = optimize_field_in_newton(
        env,
        loss_fn=lambda _states: None,
        optimization_steps=0,
        steps=1,
    )
    np.testing.assert_array_equal(result["best_params"], [3.0, 2.0, 1.0])


# --------------------------------------------------------------------------- #
# headless scene loading (real Newton)
# --------------------------------------------------------------------------- #
def test_load_example_scene_is_headless_and_overrides_substeps() -> None:
    _require_newton()
    from newton.examples.diffsim.example_diffsim_ball import Example as BallScene

    original_capture = BallScene.capture
    scene = load_example_scene(BallScene, device="cpu", substeps=2)
    assert BallScene.capture is original_capture
    assert scene.graph is None  # capture skipped
    assert scene.model is not None and scene.solver is not None
    assert scene.sim_substeps == 2
    env = NewtonEnv.from_scene(scene, observe=lambda s: field_to_torch(s.particle_q)[0])
    assert env.substeps == 2
    assert env.collide_each_substep is True
    authored_contacts = NewtonEnv.from_scene(
        scene,
        observe=lambda s: field_to_torch(s.particle_q)[0],
        collide_each_substep=False,
    )
    assert authored_contacts.collide_each_substep is False
    with pytest.raises(ValueError, match="positive"):
        load_example_scene(BallScene, device="cpu", substeps=0)


def test_load_example_scene_uses_parser_defaults_and_argument_overrides() -> None:
    _require_newton()
    import argparse

    class ArgumentScene:
        @staticmethod
        def create_parser():
            parser = argparse.ArgumentParser()
            parser.add_argument("--world-count", type=int, default=4)
            return parser

        def __init__(self, viewer, args):
            self.viewer = viewer
            self.world_count = args.world_count
            self.capture()

    assert load_example_scene(ArgumentScene, device="cpu").world_count == 4
    assert (
        load_example_scene(
            ArgumentScene, device="cpu", arg_overrides={"world_count": 7}
        ).world_count
        == 7
    )


def test_env_from_example_builds_and_wraps_in_one_call() -> None:
    _require_newton()

    class Example:
        def __init__(self, viewer, args):
            del viewer, args
            self.model = FakeModel()
            self.solver = FakeSolver()
            self.frame_dt = 0.2
            self.sim_substeps = 2
            self.sim_dt = 0.1
            self.capture()

        def capture(self) -> None:
            raise AssertionError("headless construction must override capture")

    env = NewtonEnv.from_example(Example, observe=observe_q, device="cpu")
    assert env.frame_dt == pytest.approx(0.2)
    assert env.reset().shape == (3,)


# --------------------------------------------------------------------------- #
# optional-dependency boundary
# --------------------------------------------------------------------------- #
def test_import_does_not_require_newton_or_usd() -> None:
    script = (
        "import importlib.abc, sys\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'newton' or name.startswith('newton.'): raise ImportError('blocked')\n"
        "        if name == 'pxr' or name.startswith('pxr.'): raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from physicsnemo.experimental.integrations import newton as pn\n"
        "assert 'newton' not in sys.modules\n"
        "assert 'physicsnemo.models' not in sys.modules\n"
        "assert 'physicsnemo.datapipes' not in sys.modules\n"
        "assert pn.NewtonEnv.__name__ == 'NewtonEnv'\n"
        "assert pn.BPTTSurrogate.__name__ == 'BPTTSurrogate'\n"
        "assert pn.NeRDFixedFrame.__name__ == 'NeRDFixedFrame'\n"
        "assert pn.NeRDBodyHeadingFrame.__name__ == 'NeRDBodyHeadingFrame'\n"
        "assert pn.NeRDRigidContactInput.__name__ == 'NeRDRigidContactInput'\n"
        "assert pn.concatenate_nerd_inputs.__name__ == 'concatenate_nerd_inputs'\n"
        "assert 'NeRDDataset' not in pn.__all__\n"
        "assert 'WorldView' not in pn.__all__\n"
        "assert 'newton' not in sys.modules\n"
        "assert 'physicsnemo.models' not in sys.modules\n"
        "assert 'physicsnemo.datapipes' not in sys.modules\n"
    )
    subprocess.run(  # noqa: S603 - fixed interpreter and inline test script.
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------- #
# readable state views (particles / bodies / joints)
# --------------------------------------------------------------------------- #
def test_particle_view_reads_and_writes() -> None:
    state = SimpleNamespace(particle_q=torch.zeros(4, 3), particle_qd=torch.ones(4, 3))
    view = particles(state)
    assert view.positions.shape == (4, 3) and len(view) == 4
    assert torch.equal(view.velocities, torch.ones(4, 3))
    view.positions = torch.full((4, 3), 2.0)
    assert torch.equal(state.particle_q, torch.full((4, 3), 2.0))  # wrote in place


def test_body_view_splits_transform_and_spatial_velocity() -> None:
    body_q = torch.arange(7.0).repeat(3, 1)  # [px,py,pz, qx,qy,qz,qw]
    body_qd = torch.arange(6.0).repeat(3, 1)  # [vx,vy,vz, wx,wy,wz]
    view = bodies(SimpleNamespace(body_q=body_q, body_qd=body_qd))
    assert torch.equal(view.positions, body_q[:, :3])
    assert torch.equal(view.orientations, body_q[:, 3:7])
    assert torch.equal(view.linear_velocities, body_qd[:, :3])
    assert torch.equal(view.angular_velocities, body_qd[:, 3:6])


def test_body_view_component_setters_preserve_other_components() -> None:
    state = SimpleNamespace(body_q=torch.zeros(2, 7), body_qd=torch.zeros(2, 6))
    view = bodies(state)
    view.positions = [1.0, 2.0, 3.0]
    view.orientations = [0.0, 0.0, 0.0, 1.0]
    view.linear_velocities = [4.0, 5.0, 6.0]
    view.angular_velocities = [7.0, 8.0, 9.0]
    assert torch.equal(
        state.body_q[:, :3], torch.tensor([[1.0, 2.0, 3.0]]).repeat(2, 1)
    )
    assert torch.equal(
        state.body_q[:, 3:], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    )
    assert torch.equal(
        state.body_qd[:, :3], torch.tensor([[4.0, 5.0, 6.0]]).repeat(2, 1)
    )
    assert torch.equal(
        state.body_qd[:, 3:], torch.tensor([[7.0, 8.0, 9.0]]).repeat(2, 1)
    )


def test_joint_view_reads_and_writes() -> None:
    state = SimpleNamespace(joint_q=torch.zeros(5), joint_qd=torch.ones(5))
    view = joints(state)
    assert torch.equal(view.velocities, torch.ones(5))
    view.coordinates = torch.arange(5.0)
    assert torch.equal(state.joint_q, torch.arange(5.0))


# --------------------------------------------------------------------------- #
# simulator-as-oracle design optimization (synthetic objective, no Newton required)
# --------------------------------------------------------------------------- #
def test_optimize_design_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="pool_size"):
        optimize_design(
            lambda x: x[:, 0],
            design_dim=2,
            batch_size=4,
            candidate_pool_size=3,
            log=None,
        )
    with pytest.raises(ValueError, match="one score per design"):
        optimize_design(
            lambda _x: np.zeros(1),
            design_dim=2,
            rounds=0,
            batch_size=2,
            initial_samples=2,
            surrogate_epochs=1,
            log=None,
        )


def test_optimize_design_reduces_a_smooth_objective() -> None:
    # objective minimized at u = 0.3 in every dimension
    def evaluate(units: np.ndarray) -> np.ndarray:
        return ((units - 0.3) ** 2).sum(axis=1)

    result = optimize_design(
        evaluate,
        design_dim=3,
        rounds=4,
        batch_size=8,
        initial_samples=24,
        surrogate=DesignSurrogate.mlp(3, hidden_dim=32, depth=2),
        surrogate_epochs=60,
        log=None,
    )

    assert len(result.normalized_designs) == 24 + 4 * 8
    assert result.best_score <= result.history[0]  # never worse than bootstrap
    assert all(
        b <= a for a, b in zip(result.history, result.history[1:])
    )  # monotone best-so-far
    assert result.best_score < 0.2  # found a decent design through the surrogate


def test_design_surrogate_wraps_any_core() -> None:
    # .mlp builds the default FullyConnected core; any nn.Module of the right
    # width also works (the surrogate is not tied to one architecture).
    surrogate = DesignSurrogate(torch.nn.Linear(3, 1))
    rng = np.random.default_rng(0)
    surrogate.fit(
        rng.random((20, 3)).astype(np.float32),
        rng.random(20).astype(np.float32),
        epochs=5,
    )
    assert surrogate.predict(np.zeros((4, 3), np.float32)).shape == (4,)


def test_design_surrogate_rejects_broadcastable_model_output() -> None:
    class ExtraDimension(torch.nn.Linear):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return super().forward(inputs).unsqueeze(-1)

    surrogate = DesignSurrogate(ExtraDimension(2, 1))
    with pytest.raises(ValueError, match=r"expected \(4, 1\), got \(4, 1, 1\)"):
        surrogate.fit(
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            epochs=1,
        )
    assert not bool(surrogate._is_fitted)


def test_design_surrogate_checkpoint_preserves_fitted_state() -> None:
    rng = np.random.default_rng(0)
    designs = rng.random((12, 2), dtype=np.float32)
    scores = designs.sum(axis=1)
    surrogate = DesignSurrogate.mlp(2, hidden_dim=8, depth=1, device="cpu")
    surrogate.fit(designs, scores, epochs=5)
    expected = surrogate.predict(designs)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "design_surrogate.mdlus"
        surrogate.save(path)
        loaded = DesignSurrogate.from_checkpoint(path)

    assert bool(loaded._is_fitted)
    np.testing.assert_allclose(loaded.predict(designs), expected)


def test_design_surrogate_requires_fit_and_normalized_inputs() -> None:
    surrogate = DesignSurrogate(torch.nn.Linear(2, 1))
    with pytest.raises(RuntimeError, match="fit the design surrogate"):
        surrogate.predict(np.zeros((1, 2), np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        surrogate.fit(
            np.array([[1.2, 0.0]], np.float32), np.zeros(1, np.float32), epochs=1
        )


def test_optimize_grouped_design_selects_poses_and_shared_design() -> None:
    pose_targets = torch.tensor(
        [
            [[0.15, 0.80], [0.30, 0.70]],
            [[0.65, 0.20], [0.55, 0.10]],
        ],
        dtype=torch.float32,
    )

    def grouped_losses(designs: torch.Tensor) -> torch.Tensor:
        x = designs[:, None, None, :]
        return (x - pose_targets.to(designs.device)).square().sum(dim=-1)

    result = optimize_grouped_design(
        grouped_losses,
        design_dim=2,
        starts=8,
        steps=120,
        lr=0.08,
        top_k_schedule=((0.0, 2), (0.5, 1)),
        device="cpu",
        seed=4,
    )

    assert isinstance(result, GroupedDesignResult)
    assert result.designs.shape == (8, 2)
    assert result.candidate_indices.shape == (8, 2)
    assert result.history.shape == (121, 8)
    assert result.design_history.shape == (121, 8, 2)
    assert np.all((result.designs >= 0.0) & (result.designs <= 1.0))
    assert result.best_loss < result.history[0].min()
    np.testing.assert_allclose(result.best_design, [0.425, 0.45], atol=0.12)
    candidates, steps, trajectories = result.trajectory_candidates(
        snapshots=4,
        include_initial=True,
    )
    assert candidates.shape == (32, 2)
    assert steps.shape == trajectories.shape == (32,)
    assert set(steps) == {0, 40, 80, 120}
    assert set(trajectories) == set(range(8))
    with pytest.raises(ValueError, match="positive"):
        result.trajectory_candidates(0)


def test_optimize_grouped_design_validates_callback_shape() -> None:
    with pytest.raises(ValueError, match="groups, candidates"):
        optimize_grouped_design(
            lambda designs: designs.square(),
            design_dim=2,
            starts=2,
            steps=1,
            device="cpu",
        )


def test_optimize_grouped_design_respects_trust_region() -> None:
    starts = np.asarray([[0.20, 0.80], [0.70, 0.30]], dtype=np.float32)

    def grouped_losses(designs: torch.Tensor) -> torch.Tensor:
        return designs.square().sum(dim=-1)[:, None, None]

    result = optimize_grouped_design(
        grouped_losses,
        design_dim=2,
        starts=starts,
        steps=80,
        lr=0.1,
        trust_radius=0.05,
        device="cpu",
    )

    distances = np.abs(result.designs[:, None, :] - starts[None, :, :]).max(axis=2)
    assert np.all(distances.min(axis=1) <= 0.05001)
    with pytest.raises(ValueError, match="trust_radius"):
        optimize_grouped_design(
            grouped_losses,
            design_dim=2,
            starts=starts,
            steps=1,
            trust_radius=0.0,
            device="cpu",
        )


def test_optimize_grouped_design_accepts_a_physical_design_space() -> None:
    design_space = DesignSpace(
        (
            DesignVariable("length", 0.1, 0.3),
            DesignVariable("count", 2.0, 4.0, kind="integer"),
        )
    )

    def grouped_losses(designs: torch.Tensor) -> torch.Tensor:
        target = torch.tensor((0.5, 0.7), device=designs.device)
        return (designs - target).square().sum(dim=-1)[:, None, None]

    result = optimize_grouped_design(
        grouped_losses,
        design_space=design_space,
        starts=4,
        steps=40,
        lr=0.1,
        device="cpu",
    )

    assert result.design_space is design_space
    assert result.best_physical_design.shape == (2,)
    assert result.best_physical_design[1] in (2.0, 3.0, 4.0)
    with pytest.raises(ValueError, match="does not match"):
        optimize_grouped_design(
            grouped_losses,
            design_dim=3,
            design_space=design_space,
            starts=2,
            steps=1,
            device="cpu",
        )


def test_shortlist_grouped_candidates_orders_each_group() -> None:
    losses = np.asarray(
        [
            [[3.0, 1.0, 2.0], [0.2, 0.4, 0.1]],
            [[-1.0, 4.0, 0.0], [8.0, 7.0, 6.0]],
        ]
    )

    selected = shortlist_grouped_candidates(losses, count=2)

    assert selected.shape == (2, 2, 2)
    assert np.array_equal(selected[0, 0], [1, 2])
    assert np.array_equal(selected[0, 1], [2, 0])
    assert np.array_equal(selected[1, 0], [0, 2])


def test_grouped_candidate_ranking_loss_rewards_correct_order() -> None:
    targets = torch.tensor(
        [
            [0.1, 0.5, 1.0, 0.8, 0.2, 0.6],
            [0.7, 0.2, 0.5, 0.1, 0.4, 0.9],
        ]
    )
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    accurate = targets.clone().requires_grad_(True)
    reversed_order = (1.1 - targets).requires_grad_(True)

    accurate_loss = grouped_candidate_ranking_loss(accurate, targets, groups)
    reversed_loss = grouped_candidate_ranking_loss(reversed_order, targets, groups)
    accurate_loss.backward()

    assert accurate_loss < reversed_loss
    assert accurate.grad is not None
    assert torch.isfinite(accurate.grad).all()


def test_distributed_helpers_default_to_single_process() -> None:
    from physicsnemo.experimental.integrations.newton import (
        is_main_process,
        resolve_device,
    )

    assert is_main_process() is True  # no DistributedManager initialized
    assert resolve_device() == torch.get_default_device()
    assert resolve_device("cpu") == torch.device("cpu")
    if torch.cuda.is_available():
        assert resolve_device("cuda") == torch.device(
            "cuda", torch.cuda.current_device()
        )


# --------------------------------------------------------------------------- #
# WorldView: per-world reads over a batched model
# --------------------------------------------------------------------------- #
def test_world_view_groups_and_reduces_dropping_globals() -> None:
    from physicsnemo.experimental.integrations.newton.worlds import WorldView

    model = SimpleNamespace(
        world_count=3,
        particle_world=torch.tensor([0, 0, 1, 2, 2, -1], dtype=torch.int32),
    )
    worlds = WorldView(model, "particle")
    assert torch.equal(worlds.world_index, torch.tensor([0, 0, 1, 2, 2]))
    assert torch.equal(worlds.counts, torch.tensor([2, 1, 2]))
    values = torch.tensor(
        [[1.0], [3.0], [5.0], [7.0], [9.0], [100.0]]
    )  # last row is a -1 global, must be ignored
    assert torch.allclose(
        worlds.per_world_sum(values).squeeze(1), torch.tensor([4.0, 5.0, 16.0])
    )
    assert torch.allclose(
        worlds.per_world_mean(values).squeeze(1), torch.tensor([2.0, 5.0, 8.0])
    )
    assert worlds.per_world_mean(values.double()).dtype == torch.float64
    mask = torch.tensor([True, False, True, True, False, True])
    assert torch.equal(worlds.per_world_count(mask), torch.tensor([1, 1, 1]))


def test_world_view_rejects_misaligned_values_and_masks() -> None:
    from physicsnemo.experimental.integrations.newton.worlds import WorldView

    worlds = WorldView(
        SimpleNamespace(world_count=2, particle_world=torch.tensor([0, 1]))
    )
    with pytest.raises(ValueError, match="rows"):
        worlds.per_world_sum(torch.zeros(3, 1))
    with pytest.raises(ValueError, match="boolean"):
        worlds.per_world_count(torch.ones(2))
    with pytest.raises(ValueError, match="outside"):
        WorldView(SimpleNamespace(world_count=2, particle_world=torch.tensor([0, 2])))


# --------------------------------------------------------------------------- #
# trajectory dataset: windowed rollouts for training
# --------------------------------------------------------------------------- #
def test_trajectory_windows_and_dataset_shapes() -> None:
    from physicsnemo.datapipes import DataLoader  # TensorDict-aware loader
    from physicsnemo.experimental.integrations.newton import trajectory_dataset
    from physicsnemo.experimental.integrations.newton.trajectory import (
        TrajectoryWindowReader,
    )

    traj = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)  # (time=10, feat=2)
    reader = TrajectoryWindowReader(traj, window=3, predict_steps=1)
    assert len(reader) == 7  # 10 - (3 + 1) + 1
    sample = reader._load_sample(0)
    assert torch.equal(sample["input"], traj[0:3]) and torch.equal(
        sample["target"], traj[3:4]
    )

    dataset = trajectory_dataset(traj, window=3, predict_steps=1, normalize=False)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch["input"].shape == (4, 3, 2) and batch["target"].shape == (4, 1, 2)


def test_trajectory_windows_validate_public_contract() -> None:
    from physicsnemo.experimental.integrations.newton.trajectory import (
        TrajectoryWindowReader,
    )

    with pytest.raises(ValueError, match="positive"):
        TrajectoryWindowReader(torch.zeros(4, 2), window=0)
    with pytest.raises(ValueError, match="same feature"):
        TrajectoryWindowReader(
            [torch.zeros(4, 2), torch.zeros(4, 3)], window=2, predict_steps=1
        )


def test_trajectory_dataset_normalizes() -> None:
    from physicsnemo.datapipes import DataLoader
    from physicsnemo.experimental.integrations.newton import trajectory_dataset

    torch.manual_seed(0)
    traj = torch.randn(40, 3) * 5.0 + 2.0
    dataset = trajectory_dataset(traj, window=4, predict_steps=1, normalize=True)
    batches = list(DataLoader(dataset, batch_size=8))
    inputs = torch.cat([b["input"].reshape(-1, 3) for b in batches])
    assert (
        inputs.mean().abs() < 0.3 and abs(float(inputs.std()) - 1.0) < 0.3
    )  # roughly standardized
    targets = torch.cat([b["target"].reshape(-1, 3) for b in batches])
    assert (
        targets.mean().abs() < 0.4 and abs(float(targets.std()) - 1.0) < 0.4
    )  # targets share the input statistics


# --------------------------------------------------------------------------- #
# NewtonStepModel: run a learned model in place of or after the solver
# --------------------------------------------------------------------------- #
def test_write_state_fields_writes_in_place() -> None:
    from physicsnemo.experimental.integrations.newton import write_state_fields

    state = SimpleNamespace(particle_q=torch.zeros(4, 3))
    write_state_fields(state, particle_q=torch.ones(4, 3))
    assert torch.equal(state.particle_q, torch.ones(4, 3))


def test_step_model_replace_drives_the_state() -> None:
    from physicsnemo.experimental.integrations.newton import write_state_fields

    def constant_model(state_in, state_out, control, contacts, dt) -> None:
        write_state_fields(state_out, particle_q=torch.full((3,), 7.0))

    env = NewtonEnv(
        NewtonComponents(model=FakeModel(), solver=FakeSolver()),
        observe=observe_q,
        step_model=constant_model,
        step_mode="replace",
    )
    env.reset()
    env.rollout(2, keep_states=False)
    assert torch.allclose(env.state.particle_q, torch.full((3,), 7.0))


def test_step_mode_validation() -> None:
    with pytest.raises(ValueError):
        NewtonEnv(
            NewtonComponents(model=FakeModel(), solver=FakeSolver()), step_mode="bogus"
        )
