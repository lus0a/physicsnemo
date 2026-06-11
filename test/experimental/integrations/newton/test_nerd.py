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

"""Tests for the representation-generic Newton NeRD workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from physicsnemo.experimental.integrations.newton import (
    NeRDBodyHeadingFrame,
    NeRDControlInput,
    NeRDFixedFrame,
    NeRDProblem,
    NeRDRigidContactInput,
    NeRDStepModel,
    NeRDTrainingConfig,
    TrainedNeRDModel,
    concatenate_nerd_inputs,
    fit_nerd,
    load_example_scene,
)
from physicsnemo.experimental.integrations.newton.nerd import (
    JointLayout,
    NeRDBodyStateCodec,
    NeRDCompositeStateCodec,
    NeRDDataset,
    NeRDJointStateCodec,
    NeRDModelSpec,
    NeRDNormalizers,
    NeRDParticleStateCodec,
    NeRDStateCodec,
    _anchor_base,
    _base_frame_transform,
    _body_state_to_delta,
    _body_state_tokens,
    _codec_descriptor,
    _codec_from_descriptor,
    _delta_to_body_state,
    _delta_to_next_state,
    _model_from_descriptor,
    _next_state_to_delta,
    _quat_mul,
    _resolve_nerd_model,
    _world_frame_transform,
    _wrap_to_pi,
    collect_nerd_trajectories,
    evaluate_nerd,
    nerd_state_codec,
    train_nerd,
)
from physicsnemo.experimental.integrations.newton.nerd import (
    _joint_layout as _infer_joint_layout,
)
from physicsnemo.experimental.integrations.newton.nerd import codecs as nerd_codecs
from physicsnemo.experimental.models.nerd import NeRDEntityTransformer
from physicsnemo.models.mlp import FullyConnected


def _require_newton():
    """Import Newton or skip; the integration treats it as an optional extra."""
    try:
        import newton
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Newton unavailable: {error}")
    return newton


class _State:
    def __init__(self) -> None:
        self.particle_q = torch.zeros(4, 3)
        self.particle_qd = torch.zeros(4, 3)
        self.body_q = torch.zeros(4, 7)
        self.body_q[:, 6] = 1.0
        self.body_qd = torch.zeros(4, 6)


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        world_count=2,
        particle_count=4,
        particle_world=torch.tensor([0, 0, 1, 1]),
        body_count=4,
        body_world=torch.tensor([0, 0, 1, 1]),
        body_label=["world_0/base", "world_0/tool", "world_1/base", "world_1/tool"],
        articulation_count=0,
        joint_count=0,
    )


def _fake_joint_layout() -> JointLayout:
    return JointLayout(
        world_count=1,
        dof_q=2,
        dof_qd=2,
        continuous_q_mask=torch.tensor([False, True]),
        base_translation_mask=torch.tensor([True, False]),
        root_is_free=False,
        up_axis_index=2,
        quaternion_q_starts=(),
    )


def _fake_ball_joint_layout() -> JointLayout:
    return JointLayout(
        world_count=1,
        dof_q=4,
        dof_qd=3,
        continuous_q_mask=torch.zeros(4, dtype=torch.bool),
        base_translation_mask=torch.zeros(4, dtype=torch.bool),
        root_is_free=False,
        up_axis_index=2,
        quaternion_q_starts=(0,),
    )


def test_codec_package_exports_only_public_api() -> None:
    assert all(not name.startswith("_") for name in nerd_codecs.__all__)
    assert set(nerd_codecs.__all__) == {
        "JointLayout",
        "NeRDBodyHeadingFrame",
        "NeRDBodyStateCodec",
        "NeRDCompositeStateCodec",
        "NeRDFixedFrame",
        "NeRDJointStateCodec",
        "NeRDParticleStateCodec",
        "NeRDStateCodec",
        "nerd_state_codec",
    }


def test_control_input_round_trips_the_same_per_world_features() -> None:
    adapter = NeRDControlInput("joint_f", per_world_shape=(3,))
    control = SimpleNamespace(joint_f=torch.arange(6.0))
    view = adapter.read(control)
    assert view.shape == (2, 3)
    assert view.data_ptr() == control.joint_f.data_ptr()

    values = torch.full((2, 3), 4.0)
    adapter.write(control, values)
    assert torch.equal(control.joint_f, values.reshape(-1))

    env = SimpleNamespace(control=control)
    adapter.apply(env, torch.full((2, 3), 7.0), frame=3, substep=1)
    assert torch.equal(
        adapter.from_step(None, control, None, 0.1),
        torch.full((2, 3), 7.0),
    )


def test_control_input_validates_field_and_world_shape() -> None:
    with pytest.raises(ValueError, match="positive dimensions"):
        NeRDControlInput("joint_f", per_world_shape=())
    adapter = NeRDControlInput("joint_f", per_world_shape=(4,))
    with pytest.raises(ValueError, match="cannot be grouped"):
        adapter.read(SimpleNamespace(joint_f=torch.zeros(6)))
    with pytest.raises(AttributeError, match="no field"):
        adapter.read(SimpleNamespace())


def test_concatenate_nerd_inputs_broadcasts_global_features() -> None:
    commands = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    contacts = torch.arange(12.0).reshape(2, 2, 3)
    result = concatenate_nerd_inputs(
        commands,
        contacts,
        entity_shape=(2,),
    )
    assert result.shape == (2, 2, 5)
    assert torch.equal(result[:, 0, :2], commands)
    assert torch.equal(result[..., 2:], contacts)

    with pytest.raises(ValueError, match="entity-aligned"):
        concatenate_nerd_inputs(
            commands,
            torch.zeros(2, 3, 1),
            entity_shape=(2,),
        )


def test_rigid_contact_input_aggregates_contacts_by_body_and_world() -> None:
    model = _model()
    model.shape_body = torch.tensor([0, 1, 2, 3, -1])
    codec = NeRDBodyStateCodec(model)
    adapter = NeRDRigidContactInput(model, codec)
    contacts = SimpleNamespace(
        rigid_contact_count=torch.tensor([3]),
        rigid_contact_shape0=torch.tensor([1, 3, 4, -1]),
        rigid_contact_shape1=torch.tensor([4, 4, 0, -1]),
        rigid_contact_normal=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        rigid_contact_point0=torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        rigid_contact_point1=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.7, 0.8, 0.9],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
    features = adapter.read(_State(), contacts)
    assert features.shape == (2, 2, 7)
    assert torch.allclose(
        features[0, 0],
        torch.tensor([torch.log(torch.tensor(2.0)), -1.0, 0.0, 0.0, 0.7, 0.8, 0.9]),
    )
    assert torch.equal(features[1, 0], torch.zeros(7))
    assert torch.allclose(
        features[0, 1],
        torch.tensor([torch.log(torch.tensor(2.0)), 0.0, 0.0, 1.0, 0.1, 0.2, 0.3]),
    )
    assert torch.allclose(
        features[1, 1],
        torch.tensor([torch.log(torch.tensor(2.0)), 0.0, 1.0, 0.0, 0.4, 0.5, 0.6]),
    )

    class ContactEnv:
        def __init__(self) -> None:
            self.state = _State()
            self.contacts = contacts
            self.collide_count = 0

        def collide(self) -> None:
            self.collide_count += 1

    env = ContactEnv()
    assert torch.equal(adapter.read_env(env), features)
    assert env.collide_count == 1


def test_joint_codec_wraps_angles_and_anchors_the_base() -> None:
    layout = _fake_joint_layout()
    current = torch.tensor([[2.0, torch.pi - 0.1, 0.0, 0.0]])
    next_state = torch.tensor([[3.0, -torch.pi + 0.1, 1.0, 2.0]])
    delta = _next_state_to_delta(current, next_state, layout)
    assert torch.allclose(delta, torch.tensor([[1.0, 0.2, 1.0, 2.0]]), atol=1.0e-6)
    recovered = _delta_to_next_state(current, delta, layout)
    assert torch.allclose(recovered, next_state, atol=1.0e-6)
    anchored, _ = _anchor_base(current[:, :2], current[:, 2:], layout)
    assert anchored[0, 0] == 0.0
    assert torch.allclose(_wrap_to_pi(torch.tensor([3.5])), torch.tensor([-2.7831853]))


@pytest.mark.parametrize(
    "joint_type_name",
    ("BALL", "FREE", "DISTANCE"),
)
def test_joint_layout_identifies_newton_quaternion_coordinates(
    joint_type_name: str,
) -> None:
    joint_types = _require_newton().JointType
    joint_type = getattr(joint_types, joint_type_name)
    qd_width, q_width = joint_type.dof_count(0)
    quaternion_width = joint_types.BALL.dof_count(0)[1]
    model = SimpleNamespace(
        world_count=1,
        articulation_count=1,
        joint_count=1,
        joint_world=torch.tensor([0]),
        joint_type=torch.tensor([int(joint_type)]),
        joint_q_start=torch.tensor([0, q_width]),
        joint_qd_start=torch.tensor([0, qd_width]),
        joint_limit_lower=torch.zeros(qd_width),
        joint_limit_upper=torch.zeros(qd_width),
        articulation_start=torch.tensor([0, 1]),
        up_axis=2,
    )

    layout = _infer_joint_layout(model)

    assert layout.quaternion_q_starts == (q_width - quaternion_width,)
    assert layout.delta_q_dim == q_width - 1
    assert layout.prediction_dim == q_width + qd_width - 1


def test_joint_layout_rejects_invalid_newton_quaternion_widths() -> None:
    free_type = _require_newton().JointType.FREE
    qd_width, q_width = free_type.dof_count(0)
    model = SimpleNamespace(
        world_count=1,
        articulation_count=1,
        joint_count=1,
        joint_world=torch.tensor([0]),
        joint_type=torch.tensor([int(free_type)]),
        joint_q_start=torch.tensor([0, q_width + 1]),
        joint_qd_start=torch.tensor([0, qd_width]),
        joint_limit_lower=torch.zeros(qd_width),
        joint_limit_upper=torch.zeros(qd_width),
        articulation_start=torch.tensor([0, 1]),
        up_axis=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"FREE joint metadata requires {q_width} position and "
            rf"{qd_width} velocity"
        ),
    ):
        _infer_joint_layout(model)


def test_joint_codec_uses_rotation_vector_deltas_for_quaternions() -> None:
    layout = _fake_ball_joint_layout()
    codec = NeRDJointStateCodec(layout=layout, robot_centric=False)
    current = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]])
    equivalent = torch.tensor([[0.0, 0.0, 0.0, -1.0, 1.0, 2.0, 3.0]])

    equivalent_delta = codec.state_to_delta(current, equivalent)
    assert equivalent_delta.shape == (1, 6)
    assert torch.allclose(equivalent_delta, torch.zeros_like(equivalent_delta))
    assert torch.equal(
        codec.encode_state(equivalent)[0, :4],
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    angle = 0.4
    next_state = torch.tensor(
        [
            [
                0.0,
                0.0,
                np.sin(angle / 2.0),
                np.cos(angle / 2.0),
                1.5,
                1.5,
                4.0,
            ]
        ],
        dtype=torch.float32,
    )
    delta = codec.state_to_delta(current, next_state)
    assert torch.allclose(
        delta, torch.tensor([[0.0, 0.0, angle, 0.5, -0.5, 1.0]]), atol=1.0e-6
    )

    recovered = codec.delta_to_state(current, delta)
    assert torch.allclose(recovered[..., 4:], next_state[..., 4:], atol=1.0e-6)
    quaternion_dot = (recovered[..., :4] * next_state[..., :4]).sum(-1).abs()
    assert torch.allclose(quaternion_dot, torch.ones_like(quaternion_dot), atol=1.0e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(recovered[..., :4], dim=-1),
        torch.ones(1),
        atol=1.0e-6,
    )

    descriptor = _codec_descriptor(codec)
    restored = _codec_from_descriptor(descriptor)
    assert restored.layout.quaternion_q_starts == (0,)
    assert restored.prediction_shape == (6,)
    descriptor.pop("delta_schema")
    with pytest.raises(ValueError, match="obsolete joint-delta schema"):
        _codec_from_descriptor(descriptor)


def test_joint_codec_packs_multiple_quaternions_with_scalar_coordinates() -> None:
    layout = JointLayout(
        world_count=1,
        dof_q=10,
        dof_qd=2,
        continuous_q_mask=torch.tensor(
            [True, False, False, False, False, False, False, False, False, False]
        ),
        base_translation_mask=torch.zeros(10, dtype=torch.bool),
        root_is_free=False,
        up_axis_index=2,
        quaternion_q_starts=(1, 6),
    )
    codec = NeRDJointStateCodec(layout=layout, robot_centric=False)
    current = torch.tensor(
        [
            [
                torch.pi - 0.1,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.5,
                -0.5,
            ]
        ]
    )
    next_state = torch.tensor(
        [
            [
                -torch.pi + 0.1,
                np.sin(0.1),
                0.0,
                0.0,
                np.cos(0.1),
                3.0,
                0.0,
                -np.sin(0.15),
                0.0,
                np.cos(0.15),
                1.0,
                -1.5,
            ]
        ],
        dtype=torch.float32,
    )

    delta = codec.state_to_delta(current, next_state)

    assert delta.shape == (1, 10)
    assert torch.allclose(
        delta,
        torch.tensor([[0.2, 0.2, 0.0, 0.0, 1.0, 0.0, -0.3, 0.0, 0.5, -1.0]]),
        atol=1.0e-6,
    )
    recovered = codec.delta_to_state(current, delta)
    assert torch.allclose(recovered, next_state, atol=1.0e-6)


def test_joint_codec_groups_multiple_articulations_by_newton_world() -> None:
    model = SimpleNamespace(
        world_count=2,
        articulation_count=5,
        joint_count=5,
        joint_world=torch.tensor([0, 0, 1, 1, -1]),
        joint_type=torch.tensor([0, 1, 0, 1, 0]),
        joint_q_start=torch.tensor([0, 1, 2, 3, 4, 5]),
        joint_qd_start=torch.tensor([0, 1, 2, 3, 4, 5]),
        joint_limit_lower=torch.tensor([-100.0, -10.0, -100.0, -10.0, -1.0]),
        joint_limit_upper=torch.tensor([100.0, 10.0, 100.0, 10.0, 1.0]),
        articulation_start=torch.tensor([0, 1, 2, 3, 4, 5]),
        up_axis=2,
    )
    layout = _infer_joint_layout(model)
    assert layout.world_count == 2
    assert layout.dof_q == layout.dof_qd == 2
    assert layout.q_indices.tolist() == [[0, 1], [2, 3]]
    assert layout.continuous_q_mask.tolist() == [False, True]
    assert layout.base_translation_mask.tolist() == [True, False]

    codec = NeRDJointStateCodec(layout=layout, robot_centric=False)
    state = SimpleNamespace(
        joint_q=torch.tensor([10.0, 11.0, 20.0, 21.0, 99.0]),
        joint_qd=torch.tensor([1.0, 2.0, 3.0, 4.0, 99.0]),
    )
    value = codec.read(state)
    assert value.tolist() == [[10.0, 11.0, 1.0, 2.0], [20.0, 21.0, 3.0, 4.0]]
    codec.write(state, torch.zeros_like(value))
    assert state.joint_q.tolist() == [0.0, 0.0, 0.0, 0.0, 99.0]
    assert state.joint_qd.tolist() == [0.0, 0.0, 0.0, 0.0, 99.0]


def test_joint_codec_rejects_inconsistent_world_semantics() -> None:
    model = SimpleNamespace(
        world_count=2,
        articulation_count=4,
        joint_count=4,
        joint_world=torch.tensor([0, 0, 1, 1]),
        joint_type=torch.tensor([0, 1, 0, 1]),
        joint_q_start=torch.tensor([0, 1, 2, 3, 4]),
        joint_qd_start=torch.tensor([0, 1, 2, 3, 4]),
        joint_limit_lower=torch.tensor([-100.0, -10.0, -100.0, -10.0]),
        joint_limit_upper=torch.tensor([100.0, 10.0, 100.0, 10.0]),
        articulation_start=torch.tensor([0, 1, 2, 3, 4]),
        joint_label=[
            "world_0/root",
            "world_0/hinge",
            "world_1/root",
            "world_1/wrong_hinge",
        ],
        body_label=[
            "world_0/base",
            "world_0/tool",
            "world_1/base",
            "world_1/tool",
        ],
        joint_parent=torch.tensor([-1, 0, -1, 2]),
        joint_child=torch.tensor([0, 1, 2, 3]),
        up_axis=2,
    )

    with pytest.raises(ValueError, match="ordered joint semantics"):
        _infer_joint_layout(model)


def test_joint_codec_write_supports_leaf_state_tensors() -> None:
    codec = NeRDJointStateCodec(layout=_fake_joint_layout(), robot_centric=False)
    state = SimpleNamespace(
        joint_q=torch.zeros(2, requires_grad=True),
        joint_qd=torch.zeros(2, requires_grad=True),
    )
    codec.write(state, torch.ones(1, 4))
    assert torch.equal(state.joint_q, torch.ones(2))
    assert torch.equal(state.joint_qd, torch.ones(2))


def test_free_body_frame_transform_round_trip_uses_newton_velocity_order() -> None:
    position = torch.tensor([[1.0, 2.0, 3.0]])
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    linear = torch.tensor([[0.5, -0.2, 0.1]])
    angular = torch.tensor([[0.1, 0.2, 0.3]])
    frame_position = torch.tensor([[0.2, 0.3, 0.4]])
    frame_quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    local = _base_frame_transform(
        position, quaternion, linear, angular, frame_position, frame_quaternion
    )
    recovered = _world_frame_transform(*local, frame_position, frame_quaternion)
    for actual, expected in zip(recovered, (position, quaternion, linear, angular)):
        assert torch.allclose(actual, expected, atol=1.0e-6)
    assert torch.equal(
        local[2], linear
    )  # frame translation does not shift COM velocity


def test_body_delta_round_trip_and_live_token_pack() -> None:
    # Local seed so the round-trip tolerance does not silently depend on the
    # conftest SEED constant; the quat<->rotvec round trip can lose up to a few
    # milliradians near 180-degree rotations.
    torch.manual_seed(0)
    current = torch.randn(3, 4, 13)
    next_state = torch.randn(3, 4, 13)
    current[..., 3:7] = torch.nn.functional.normalize(current[..., 3:7], dim=-1)
    next_state[..., 3:7] = torch.nn.functional.normalize(next_state[..., 3:7], dim=-1)
    recovered = _delta_to_body_state(current, _body_state_to_delta(current, next_state))
    assert torch.allclose(recovered[..., :3], next_state[..., :3], atol=1.0e-5)
    assert torch.allclose(recovered[..., 7:], next_state[..., 7:], atol=1.0e-5)

    def _quat_angular_distance(q1, q2):
        dot = (q1 * q2).sum(-1).abs().clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    # Compare quaternions by angular distance: the quat<->rotvec round trip
    # canonicalizes onto the w>=0 hemisphere, so allclose on raw components fails.
    assert (
        _quat_angular_distance(recovered[..., 3:7], next_state[..., 3:7]) < 5.0e-3
    ).all()
    state = SimpleNamespace(
        body_q=torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, -2.0]]),
        body_qd=torch.arange(6, dtype=torch.float32).reshape(1, 6),
    )
    tokens = _body_state_tokens(state)
    assert torch.equal(tokens[0, 7:], state.body_qd[0])
    assert torch.equal(tokens[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_codec_factory_builds_explicit_particle_body_composite() -> None:
    codec = nerd_state_codec(_model(), ("particle", "body"))
    assert isinstance(codec, NeRDCompositeStateCodec)
    assert codec.batch_size == 2
    assert codec.state_shape == (2 * 6 + 2 * 13,)
    state = _State()
    value = codec.read(state)
    value[:, 0] = 3.0
    codec.write(state, value)
    assert torch.equal(state.particle_q[[0, 2], 0], torch.tensor([3.0, 3.0]))


def test_composite_codec_rejects_overlapping_state_ownership() -> None:
    model = SimpleNamespace(
        world_count=1,
        body_count=1,
        body_world=torch.tensor([0]),
        body_label=["root"],
    )
    body = NeRDBodyStateCodec(model)
    joint = NeRDJointStateCodec(layout=_fake_joint_layout())
    with pytest.raises(ValueError, match="'body_q'.*owned by both"):
        NeRDCompositeStateCodec(joint, body)


def test_composite_codec_requires_explicit_state_ownership() -> None:
    class UndeclaredCodec(NeRDStateCodec):
        name = "undeclared"
        batch_size = 1
        state_shape = (1,)
        prediction_shape = (1,)

        def read(self, state):
            return state

        def write(self, state, value):
            state.copy_(value)

    with pytest.raises(ValueError, match="must declare state_fields"):
        NeRDCompositeStateCodec(UndeclaredCodec())


def test_entity_codec_compatibility_rejects_permuted_semantics() -> None:
    model = _model()
    body = NeRDBodyStateCodec(model, indices=torch.tensor(((0, 1), (2, 3))))
    swapped_body = NeRDBodyStateCodec(
        model,
        indices=torch.tensor(((1, 0), (3, 2))),
    )
    particle = NeRDParticleStateCodec(
        model,
        indices=torch.tensor(((0, 1), (2, 3))),
    )
    swapped_particle = NeRDParticleStateCodec(
        model,
        indices=torch.tensor(((1, 0), (3, 2))),
    )

    assert body.compatibility_signature() != swapped_body.compatibility_signature()
    assert (
        particle.compatibility_signature() != swapped_particle.compatibility_signature()
    )


def test_codec_factory_builds_explicit_particle_state() -> None:
    model = SimpleNamespace(
        world_count=2,
        particle_count=4,
        particle_world=torch.tensor([0, 0, 1, 1]),
        body_count=1,
        body_world=torch.tensor([-1]),
        articulation_count=0,
        joint_count=0,
    )
    codec = nerd_state_codec(model, "particle")
    assert isinstance(codec, NeRDParticleStateCodec)
    with pytest.raises(ValueError, match="representation must be"):
        nerd_state_codec(model, "auto")


def test_body_codec_uses_rotation_vector_delta_and_writes_selected_entities() -> None:
    state = _State()
    codec = NeRDBodyStateCodec(_model())
    current = codec.read(state)
    next_state = current.clone()
    next_state[..., 0] += 0.25
    delta = codec.state_to_delta(current, next_state)
    assert delta.shape == (2, 2, 12)
    recovered = codec.delta_to_state(current, delta)
    codec.write(state, recovered)
    assert torch.allclose(state.body_q[:, 0], torch.full((4,), 0.25))


def test_body_heading_frame_is_invariant_and_invertible() -> None:
    codec = NeRDBodyStateCodec(
        _model(),
        reference_frame=NeRDBodyHeadingFrame(body="base", up_axis=2),
    )
    current = torch.zeros(2, 2, 13)
    current[0, 0, :3] = torch.tensor([1.0, 2.0, 0.5])
    current[0, 1, :3] = torch.tensor([2.0, 2.0, 0.5])
    tilt = torch.tensor([np.sin(0.2), 0.0, 0.0, np.cos(0.2)])
    child_relative = torch.tensor([0.0, np.sin(0.15), 0.0, np.cos(0.15)])
    current[0, 0, 3:7] = tilt
    current[0, 1, 3:7] = _quat_mul(tilt, child_relative)
    current[0, :, 7:10] = torch.tensor([1.0, 0.2, 0.1])
    current[0, :, 10:13] = torch.tensor([0.3, -0.1, 0.2])

    half = 2.0**-0.5
    yaw = torch.tensor([0.0, 0.0, half, half])
    current[1, :, 3:7] = _quat_mul(yaw, current[0, :, 3:7])
    current[1, 0, :3] = torch.tensor([-3.0, 4.0, 0.5])
    current[1, 1, :3] = torch.tensor([-3.0, 5.0, 0.5])
    current[1, :, 7:10] = torch.tensor([-0.2, 1.0, 0.1])
    current[1, :, 10:13] = torch.tensor([0.1, 0.3, 0.2])

    encoded = codec.encode_state(current)
    assert torch.allclose(encoded[0], encoded[1], atol=1.0e-6)
    world_commands = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    local_commands = codec.world_vectors_to_model_frame(current, world_commands)
    assert torch.allclose(local_commands[0], local_commands[1], atol=1.0e-6)

    next_state = current.clone()
    next_state[0, :, :3] += torch.tensor([0.02, -0.01, 0.03])
    next_state[0, :, 7:10] += torch.tensor([0.05, -0.02, 0.01])
    next_state[0, :, 10:13] += torch.tensor([0.01, 0.03, -0.02])
    rotation_delta = torch.tensor([0.0, 0.0, np.sin(0.05), np.cos(0.05)])
    next_state[0, :, 3:7] = _quat_mul(
        rotation_delta.expand(2, 4), next_state[0, :, 3:7]
    )

    def rotate_yaw(vector: torch.Tensor) -> torch.Tensor:
        return torch.stack((-vector[..., 1], vector[..., 0], vector[..., 2]), dim=-1)

    next_state[1, :, :3] = (
        rotate_yaw(next_state[0, :, :3] - current[0, 0, :3]) + current[1, 0, :3]
    )
    next_state[1, :, 3:7] = _quat_mul(yaw.expand(2, 4), next_state[0, :, 3:7])
    next_state[1, :, 7:10] = rotate_yaw(next_state[0, :, 7:10])
    next_state[1, :, 10:13] = rotate_yaw(next_state[0, :, 10:13])
    delta = codec.state_to_delta(current, next_state)
    assert torch.allclose(delta[0], delta[1], atol=1.0e-6)
    recovered = codec.delta_to_state(current, delta)
    assert torch.allclose(recovered[..., :3], next_state[..., :3], atol=1.0e-6)
    assert torch.allclose(recovered[..., 7:], next_state[..., 7:], atol=1.0e-6)
    assert torch.allclose(
        (recovered[..., 3:7] * next_state[..., 3:7]).sum(-1).abs(),
        torch.ones(2, 2),
        atol=1.0e-6,
    )


def test_contact_normals_use_the_body_heading_frame() -> None:
    model = _model()
    model.shape_body = torch.tensor([0, 1, 2, 3])
    codec = NeRDBodyStateCodec(
        model,
        reference_frame=NeRDBodyHeadingFrame(body="base", up_axis=2),
    )
    state = _State()
    half = 2.0**-0.5
    state.body_q[2:, 3:7] = torch.tensor([0.0, 0.0, half, half])
    contacts = SimpleNamespace(
        rigid_contact_count=torch.tensor([1]),
        rigid_contact_shape0=torch.tensor([3]),
        rigid_contact_shape1=torch.tensor([0]),
        rigid_contact_normal=torch.tensor([[0.0, 1.0, 0.0]]),
        rigid_contact_point0=torch.zeros(1, 3),
        rigid_contact_point1=torch.zeros(1, 3),
    )
    features = NeRDRigidContactInput(model, codec).read(state, contacts)
    assert torch.allclose(
        features[1, 1, 1:4], torch.tensor([1.0, 0.0, 0.0]), atol=1.0e-6
    )


def test_body_heading_frame_checkpoint_descriptor_round_trip() -> None:
    codec = NeRDBodyStateCodec(
        _model(),
        reference_frame=NeRDBodyHeadingFrame(body="tool", up_axis=1),
    )
    restored = _codec_from_descriptor(_codec_descriptor(codec))
    assert isinstance(restored, NeRDBodyStateCodec)
    assert restored.reference_frame == NeRDBodyHeadingFrame(body=1, up_axis=1)
    assert restored.compatibility_signature() == codec.compatibility_signature()


def test_implicit_body_frame_checkpoint_descriptor_is_rejected() -> None:
    codec = NeRDBodyStateCodec(_model())
    with pytest.raises(ValueError, match="obsolete implicit body-frame schema"):
        _codec_from_descriptor(
            {
                "type": "body",
                "indices": codec.indices,
                "semantic_ids": codec.semantic_ids,
                "robot_centric": True,
                "root_body": 1,
                "up_axis": 2,
            }
        )


@pytest.mark.parametrize("body", [True, 0.5, "", None])
def test_body_heading_frame_rejects_invalid_body_reference(body: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        NeRDBodyHeadingFrame(body=body)  # type: ignore[arg-type]


def test_body_codec_factory_accepts_an_explicit_heading_frame() -> None:
    codec = nerd_state_codec(
        _model(),
        "body",
        body_reference_frame=NeRDBodyHeadingFrame(body="tool", up_axis=1),
    )
    assert isinstance(codec, NeRDBodyStateCodec)
    assert codec.reference_frame == NeRDBodyHeadingFrame(body=1, up_axis=1)


def test_fixed_body_frame_preserves_position_relative_to_environment() -> None:
    half = 2.0**-0.5
    frame = NeRDFixedFrame(
        position=(10.0, -2.0, 0.5),
        quaternion=(0.0, 0.0, half, half),
    )
    codec = NeRDBodyStateCodec(_model(), reference_frame=frame)
    current = torch.zeros(2, 2, 13)
    current[..., 6] = 1.0
    current[0, 0, :3] = torch.tensor([10.0, -1.0, 0.5])
    current[0, 1, :3] = torch.tensor([9.0, -2.0, 0.5])
    current[1] = current[0]
    current[1, :, 0] += 1.0

    encoded = codec.encode_state(current)
    assert torch.allclose(encoded[0, 0, :3], torch.tensor([1.0, 0.0, 0.0]), atol=1.0e-6)
    assert not torch.allclose(encoded[0], encoded[1])

    next_state = current.clone()
    next_state[..., :3] += torch.tensor([0.02, -0.01, 0.03])
    delta = codec.state_to_delta(current, next_state)
    recovered = codec.delta_to_state(current, delta)
    assert torch.allclose(recovered, next_state, atol=1.0e-6)

    world_vector = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    local_vector = codec.world_vectors_to_model_frame(current, world_vector)
    assert torch.allclose(
        local_vector,
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atol=1.0e-6,
    )
    world_point = torch.tensor([[10.0, -1.0, 0.5], [10.0, -1.0, 0.5]])
    local_point = codec.world_points_to_model_frame(current, world_point)
    assert torch.allclose(
        local_point,
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atol=1.0e-6,
    )


def test_fixed_body_frame_checkpoint_descriptor_round_trip() -> None:
    frame = NeRDFixedFrame(
        position=(1.0, 2.0, 3.0),
        quaternion=(0.0, 0.0, 0.0, 2.0),
    )
    codec = NeRDBodyStateCodec(_model(), reference_frame=frame)
    restored = _codec_from_descriptor(_codec_descriptor(codec))
    assert isinstance(restored, NeRDBodyStateCodec)
    assert restored.reference_frame == NeRDFixedFrame(position=(1.0, 2.0, 3.0))
    assert restored.compatibility_signature() == codec.compatibility_signature()


def test_body_codec_accepts_unreplicated_newton_scene_world_ids() -> None:
    model = SimpleNamespace(
        world_count=1,
        body_count=4,
        body_world=torch.full((4,), -1),
    )
    codec = NeRDBodyStateCodec(model)
    assert codec.indices.tolist() == [[0, 1, 2, 3]]


def test_problem_runner_collects_any_existing_newton_run() -> None:
    model = _model()
    state = _State()
    codec = NeRDParticleStateCodec(model)

    def reset(rng: np.random.Generator) -> None:
        state.particle_q.zero_()
        state.particle_qd.zero_()
        state.particle_qd[:, 0] = torch.as_tensor(
            rng.uniform(0.5, 1.0, 4), dtype=torch.float32
        )

    def advance(inputs: torch.Tensor, frame: int) -> None:
        del frame
        per_particle = inputs.repeat_interleave(2, dim=0)
        state.particle_qd[:, :1] += per_particle
        state.particle_q += state.particle_qd

    problem = NeRDProblem(
        codec=codec,
        get_state=lambda: state,
        reset=reset,
        advance=advance,
        sample_inputs=lambda rng, batch, frame: np.full(
            (batch, 1), 0.1 * (frame + 1), np.float32
        ),
    )
    trajectories = collect_nerd_trajectories(
        problem,
        num_trajectories=3,
        steps=4,
        device="cpu",
        log=lambda _message: None,
    )
    assert trajectories.states.shape == (3, 5, 2, 6)
    assert trajectories.inputs.shape == (3, 4, 1)
    assert torch.isfinite(trajectories.states).all()


def test_problem_runner_preserves_entity_aligned_inputs() -> None:
    model = _model()
    state = _State()
    codec = NeRDParticleStateCodec(model)

    def reset(rng: np.random.Generator) -> None:
        del rng
        state.particle_q.zero_()
        state.particle_qd.zero_()

    def advance(inputs: torch.Tensor, frame: int) -> None:
        del frame
        state.particle_qd[:, 0] += inputs[..., 0].reshape(-1)
        state.particle_q += state.particle_qd

    problem = NeRDProblem(
        codec=codec,
        get_state=lambda: state,
        reset=reset,
        advance=advance,
        sample_inputs=lambda rng, batch, frame: rng.uniform(
            0.0, 0.1 * (frame + 1), (batch, 2, 3)
        ),
    )
    trajectories = collect_nerd_trajectories(
        problem,
        num_trajectories=2,
        steps=3,
        device="cpu",
        log=lambda _message: None,
    )
    assert trajectories.inputs.shape == (2, 3, 2, 3)
    assert trajectories.external_input_shape == (2, 3)


def test_problem_runner_separates_teacher_controls_from_model_observations() -> None:
    model = _model()
    state = _State()
    codec = NeRDParticleStateCodec(model)
    controls_seen: list[torch.Tensor] = []

    def reset(rng: np.random.Generator) -> None:
        del rng
        state.particle_q.zero_()
        state.particle_qd.zero_()

    def advance(controls: torch.Tensor, frame: int) -> None:
        del frame
        controls_seen.append(controls.clone())
        per_particle = controls.repeat_interleave(2, dim=0)
        state.particle_qd[:, :1] += per_particle
        state.particle_q += state.particle_qd

    def observe(controls: torch.Tensor, frame: int) -> torch.Tensor:
        del frame
        positions = codec.read(state)[..., :1]
        return concatenate_nerd_inputs(
            controls,
            positions,
            entity_shape=codec.state_shape[:-1],
        )

    problem = NeRDProblem(
        codec=codec,
        get_state=lambda: state,
        reset=reset,
        advance=advance,
        sample_inputs=lambda rng, batch, frame: np.full(
            (batch, 1), 0.1 * (frame + 1), np.float32
        ),
        observe_inputs=observe,
    )
    trajectories = collect_nerd_trajectories(
        problem,
        num_trajectories=2,
        steps=2,
        device="cpu",
        log=lambda _message: None,
    )

    assert trajectories.inputs.shape == (2, 2, 2, 2)
    assert torch.allclose(trajectories.inputs[:, 0, :, 0], torch.full((2, 2), 0.1))
    assert torch.equal(trajectories.inputs[:, 0, :, 1], torch.zeros(2, 2))
    assert torch.allclose(trajectories.inputs[:, 1, :, 1], torch.full((2, 2), 0.1))
    assert [tuple(value.shape) for value in controls_seen] == [(2, 1), (2, 1)]


def _free_particle_nerd_problem(state: _State) -> NeRDProblem:
    """Inline NeRD problem fixture mirroring the collection-path tests."""
    codec = NeRDParticleStateCodec(_model())

    def reset(rng: np.random.Generator) -> None:
        state.particle_q.zero_()
        state.particle_qd.zero_()
        state.particle_qd[:, 0] = torch.as_tensor(
            rng.uniform(0.5, 1.0, 4), dtype=torch.float32
        )

    def advance(inputs: torch.Tensor, frame: int) -> None:
        del frame
        per_particle = inputs.repeat_interleave(2, dim=0)
        state.particle_qd[:, :1] += per_particle
        state.particle_q += state.particle_qd

    return NeRDProblem(
        codec=codec,
        get_state=lambda: state,
        reset=reset,
        advance=advance,
        sample_inputs=lambda rng, batch, frame: np.full(
            (batch, 1), 0.1 * (frame + 1), np.float32
        ),
    )


def test_fit_nerd_composes_collection_and_training_on_cpu() -> None:
    problem = _free_particle_nerd_problem(_State())
    config = NeRDTrainingConfig(
        context_frames=2,
        epochs=1,
        steps_per_epoch=2,
        batch_size=2,
    )
    trained = fit_nerd(
        problem,
        num_trajectories=3,
        steps=4,
        config=config,
        dynamics_model="FullyConnected",
        model_kwargs={"layer_size": 8, "num_layers": 1},
        device="cpu",
        log=lambda _message: None,
    )
    assert isinstance(trained, TrainedNeRDModel)
    # The collection codec is carried through into the trained model, and the
    # per-frame ``(batch, 1)`` inputs collapse to a one-element external shape.
    assert (
        trained.codec.compatibility_signature()
        == problem.codec.compatibility_signature()
    )
    assert trained.external_input_shape == (1,)


def test_fit_nerd_forwards_trajectory_filter_into_collection() -> None:
    # A filter rejecting every trajectory must reach collection through
    # fit_nerd; if it were dropped, collection would succeed and training
    # would not raise. The filter must return one flag per collected world.
    seen: list[tuple[int, ...]] = []

    def reject_everything(states: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        del inputs
        seen.append(tuple(states.shape))
        return torch.zeros(states.shape[0], dtype=torch.bool)

    with pytest.raises(RuntimeError, match="every collected trajectory was rejected"):
        fit_nerd(
            _free_particle_nerd_problem(_State()),
            num_trajectories=3,
            steps=4,
            config=NeRDTrainingConfig(
                context_frames=2,
                epochs=1,
                steps_per_epoch=2,
                batch_size=2,
            ),
            dynamics_model="FullyConnected",
            model_kwargs={"layer_size": 8, "num_layers": 1},
            device="cpu",
            trajectory_filter=reject_everything,
            log=lambda _message: None,
        )
    assert seen, "trajectory_filter was not forwarded into collection"


def test_training_supervises_every_causal_prefix() -> None:
    class ScalarCodec(NeRDStateCodec):
        name = "scalar"
        batch_size = 1
        state_shape = (1,)
        prediction_shape = (1,)

        def read(self, state):
            return state

        def write(self, state, value):
            state.copy_(value)

    class PerPositionModel(torch.nn.Module):
        def __init__(self, context_frames: int) -> None:
            super().__init__()
            self.delta = torch.nn.Parameter(torch.zeros(context_frames))

        def forward(self, inputs):
            batch, time = inputs.shape[:2]
            return self.delta[:time].view(1, time, 1).expand(batch, -1, -1)

    context_frames = 3
    model = PerPositionModel(context_frames)
    states = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0]).view(1, 5, 1)
    states = states.expand(4, -1, -1)
    trained = train_nerd(
        NeRDDataset(states=states, codec=ScalarCodec()),
        NeRDTrainingConfig(
            context_frames=context_frames,
            epochs=1,
            steps_per_epoch=1,
            batch_size=4,
        ),
        dynamics_model=model,
        device="cpu",
        log=lambda _message: None,
    )
    assert bool((trained.model.delta != 0.0).all())


def test_train_nerd_learns_linear_dynamics_better_than_barely_trained() -> None:
    # Deterministic linear dynamics q_{t+1} = q_t + input. The particle codec's
    # delta is plain subtraction, so the learnable target is exactly the input
    # and a trained model must drive rollout error far below a barely-trained
    # baseline -- proving the optimizer actually fits rather than relying on a
    # lucky initialization.
    torch.manual_seed(0)
    codec = NeRDParticleStateCodec(_model())
    trajectories, frames = 8, 6
    inputs = 0.05 * torch.randn(trajectories, frames - 1, 2, 6)
    states = torch.zeros(trajectories, frames, 2, 6)
    for frame in range(frames - 1):
        states[:, frame + 1] = states[:, frame] + inputs[:, frame]
    dataset = NeRDDataset(states=states, inputs=inputs, codec=codec, frame_dt=0.1)
    config_kwargs = {
        "context_frames": 2,
        "batch_size": 4,
    }
    common = {
        "dynamics_model": "FullyConnected",
        "model_kwargs": {"layer_size": 64, "num_layers": 3},
        "device": "cpu",
        "seed": 0,
        "log": lambda _message: None,
    }
    barely_trained = train_nerd(
        dataset,
        NeRDTrainingConfig(epochs=1, steps_per_epoch=1, **config_kwargs),
        **common,
    )
    well_trained = train_nerd(
        dataset,
        NeRDTrainingConfig(epochs=150, steps_per_epoch=8, **config_kwargs),
        **common,
    )
    baseline = evaluate_nerd(barely_trained, dataset, device="cpu")
    learned = evaluate_nerd(well_trained, dataset, device="cpu")
    assert learned.finite_trajectory_fraction == 1.0
    assert learned.mean_error < 0.25 * baseline.mean_error
    assert learned.mean_error < 1.0e-2


def test_generic_training_and_runtime_use_registry_model_name() -> None:
    # End-to-end lifecycle of a registry-named NeRD model, exercised as a
    # sequence of stages: dataset -> train -> step-model + frame_dt guard ->
    # evaluate/predict -> deployment codec compatibility -> save/load.

    # Stage 1: dataset and training from a registry model name.
    model = _model()
    codec = NeRDParticleStateCodec(model)
    states = torch.zeros(4, 6, 2, 6)
    for frame in range(states.shape[1]):
        states[:, frame, :, 0] = frame * 0.1
    trajectories = NeRDDataset(
        states=states,
        codec=codec,
        frame_dt=0.1,
    )
    assert trajectories.inputs.shape == (4, 5, 0)
    trained = train_nerd(
        trajectories,
        NeRDTrainingConfig(
            context_frames=2,
            epochs=1,
            steps_per_epoch=2,
            batch_size=2,
        ),
        dynamics_model="FullyConnected",
        model_kwargs={"layer_size": 8, "num_layers": 1},
        device="cpu",
        log=lambda _message: None,
    )
    assert isinstance(trained.model, FullyConnected)

    # Stage 2: step-model construction, callability, and frame_dt guard.
    runtime = trained.as_step_model(device="cpu")
    assert callable(runtime)
    state_in, state_out = _State(), _State()
    state_in.body_qd.fill_(3.0)
    state_out.body_qd.fill_(-1.0)
    runtime(state_in, state_out, None, None, 0.1)
    assert torch.isfinite(state_out.particle_q).all()
    assert torch.equal(state_out.body_qd, state_in.body_qd)
    with pytest.raises(ValueError, match="frame_dt"):
        runtime(state_in, state_out, None, None, 0.2)

    # Stage 3: evaluation and single-step prediction.
    evaluation = evaluate_nerd(trained, trajectories, device="cpu")
    assert evaluation.error_by_frame.shape == (5,)
    assert evaluation.finite_trajectory_fraction == 1.0
    next_state = trained.predict_next(states[:, :2], device="cpu")
    assert next_state.shape == states[:, 0].shape

    # Stage 4: deployment codec compatibility (single-world build + rejection).
    single_world = SimpleNamespace(
        device="cpu",
        world_count=1,
        particle_count=2,
        particle_world=torch.tensor([0, 0]),
    )
    deployed = trained.as_step_model(newton_model=single_world)
    assert deployed.codec.batch_size == 1

    class LookalikeCodec(NeRDStateCodec):
        name = "lookalike"
        batch_size = 1
        state_shape = codec.state_shape
        prediction_shape = codec.prediction_shape

        def read(self, state):
            return torch.zeros(1, *self.state_shape)

        def write(self, state, value):
            pass

    with pytest.raises(ValueError, match="semantically incompatible"):
        trained.as_step_model(device="cpu", state_codec=LookalikeCodec())

    # Stage 5: save/load round-trip preserves model type, codec, and frame_dt.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "particle_nerd.pt"
        trained.save(path)
        loaded = type(trained).load(path)
    assert type(loaded.model) is type(trained.model)
    assert loaded.codec.compatibility_signature() == codec.compatibility_signature()
    assert loaded.frame_dt == pytest.approx(0.1)


def test_compiled_inference_wiring_and_checkpoint(monkeypatch) -> None:
    codec = NeRDParticleStateCodec(_model())
    model = FullyConnected(
        in_features=6,
        out_features=6,
        layer_size=8,
        num_layers=1,
    )
    normalizers = NeRDNormalizers(
        input_mean=torch.zeros(2, 6),
        input_std=torch.ones(2, 6),
        target_mean=torch.zeros(2, 6),
        target_std=torch.ones(2, 6),
    )
    trained = TrainedNeRDModel(
        model=model,
        normalizers=normalizers,
        codec=codec,
        config=NeRDTrainingConfig(context_frames=2),
        external_input_shape=(0,),
        active_delta_mask=torch.ones(2, 6),
    )
    compile_args = {}

    def fake_compile(module, **kwargs):
        compile_args.update(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    runtime = trained.compile_for_inference(device="cpu")
    initial = torch.zeros(2, 2, 6)
    assert torch.equal(
        runtime.rollout(initial, steps=2, device="cpu"),
        trained.rollout(initial, steps=2, device="cpu"),
    )
    assert compile_args == {"mode": "reduce-overhead", "fullgraph": True}
    assert runtime.metadata["inference_compile_mode"] == "reduce-overhead"

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "compiled_nerd.pt"
        runtime.save(path)
        loaded = TrainedNeRDModel.load(path)
    assert isinstance(loaded.model, FullyConnected)
    assert "inference_compile_mode" not in loaded.metadata


def test_checkpoint_model_descriptor_rejects_arbitrary_imports() -> None:
    with pytest.raises(ValueError, match="cannot be reconstructed safely"):
        _model_from_descriptor(
            {"module": "os", "qualname": "system", "args": {"command": "false"}}
        )


def test_conditioned_model_requires_inputs_at_rollout_and_deployment() -> None:
    codec = NeRDParticleStateCodec(_model())
    states = torch.zeros(2, 4, 2, 6)
    states[:, 1:, :, 0] = torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1)
    trajectories = NeRDDataset(
        states=states,
        inputs=torch.zeros(2, 3, 2),
        codec=codec,
    )
    trained = train_nerd(
        trajectories,
        NeRDTrainingConfig(
            context_frames=2,
            epochs=1,
            steps_per_epoch=1,
            batch_size=2,
        ),
        dynamics_model="FullyConnected",
        model_kwargs={"layer_size": 8, "num_layers": 1},
        device="cpu",
        log=lambda _message: None,
    )
    with pytest.raises(ValueError, match="inputs are required"):
        trained.rollout(states[:, 0], steps=1, device="cpu")
    with pytest.raises(ValueError, match="input_history is required"):
        trained.predict_next(states[:, :2], device="cpu")
    with pytest.raises(ValueError, match="input_from_step is required"):
        trained.as_step_model(device="cpu")(_State(), _State(), None, None, 0.1)
    deployed = trained.as_step_model(
        device="cpu",
        input_from_step=lambda state, control, contacts, dt: control,
    )
    assert isinstance(deployed, NeRDStepModel)
    with pytest.raises(TypeError, match="not a SimpleNamespace"):
        deployed(_State(), _State(), SimpleNamespace(), None, 0.1)
    state_out = _State()
    deployed.step_with_inputs(
        _State(),
        state_out,
        torch.zeros(2, 2),
        dt=0.1,
    )
    assert torch.isfinite(state_out.particle_q).all()


def test_entity_model_registry_dimensions_are_inferred() -> None:
    spec = NeRDModelSpec(
        input_dim=14,
        prediction_dim=12,
        context_frames=3,
        input_shape=(5, 14),
        prediction_shape=(5, 12),
    )
    model = _resolve_nerd_model(
        "NeRDEntityTransformer",
        spec,
        device="cpu",
        model_kwargs={
            "hidden_size": 8,
            "entity_depth": 1,
            "temporal_depth": 1,
            "num_heads": 2,
            "head_hidden": 8,
            "head_layers": 1,
        },
    )
    assert isinstance(model, NeRDEntityTransformer)
    assert model.num_entities == 5
    assert model.feature_dim == 14


def test_dynamics_model_auto_selection_is_rejected() -> None:
    spec = NeRDModelSpec(
        input_dim=4,
        prediction_dim=4,
        context_frames=2,
        input_shape=(4,),
        prediction_shape=(4,),
    )
    with pytest.raises(ValueError, match="choose.*NeRDTransformer"):
        _resolve_nerd_model("auto", spec, device="cpu")


def test_entity_aligned_inputs_train_with_entity_model() -> None:
    codec = NeRDParticleStateCodec(_model())
    states = torch.zeros(2, 4, 2, 6)
    states[:, 1:, :, 0] = torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1)
    inputs = torch.randn(2, 3, 2, 3)
    trained = train_nerd(
        NeRDDataset(states=states, inputs=inputs, codec=codec),
        NeRDTrainingConfig(
            context_frames=2,
            epochs=1,
            steps_per_epoch=1,
            batch_size=2,
        ),
        dynamics_model="NeRDEntityTransformer",
        model_kwargs={
            "hidden_size": 8,
            "entity_depth": 1,
            "temporal_depth": 1,
            "num_heads": 2,
            "head_hidden": 8,
            "head_layers": 1,
        },
        device="cpu",
        log=lambda _message: None,
    )
    assert trained.external_input_shape == (2, 3)
    assert trained.model.feature_dim == 9
    prediction = trained.rollout(states[:, 0], inputs, device="cpu")
    assert prediction.shape == states.shape


def test_custom_codec_inputs_and_model_builder_define_a_new_problem() -> None:
    # End-to-end lifecycle of a user-defined problem (custom codec + model
    # builder), exercised as a sequence of stages: spec capture -> rollout
    # invariants -> save -> custom-codec load rejection -> codec-keyed reload.

    # Stage 1: a custom codec and builder define the problem.
    class CustomCodec(NeRDStateCodec):
        name = "custom"
        batch_size = 2
        state_shape = (3,)
        prediction_shape = (2,)

        def read(self, state):
            return state.clone()

        def write(self, state, value):
            state.copy_(value)

        def state_to_delta(self, current, next_state):
            return next_state[..., :2] - current[..., :2]

        def delta_to_state(self, current, delta):
            next_state = current.clone()
            next_state[..., :2] += delta
            return next_state

    codec = CustomCodec()
    states = torch.zeros(4, 5, 3)
    states[:, :, 0] = torch.arange(5)
    states[:, :, 1] = 2.0 * torch.arange(5)
    states[:, :, 2] = torch.arange(4).view(-1, 1)
    inputs = torch.randn(4, 4, 2)
    captured = {}

    def model_builder(spec: NeRDModelSpec) -> torch.nn.Module:
        captured["spec"] = spec
        return FullyConnected(
            in_features=spec.input_dim,
            out_features=spec.prediction_dim,
            layer_size=8,
            num_layers=1,
        )

    trained = train_nerd(
        NeRDDataset(states=states, inputs=inputs, codec=codec),
        NeRDTrainingConfig(
            context_frames=2,
            epochs=1,
            steps_per_epoch=1,
            batch_size=2,
        ),
        dynamics_model=model_builder,
        log=lambda _message: None,
    )
    # Stage 2: the builder receives the inferred spec.
    assert captured["spec"].input_shape == (5,)
    assert captured["spec"].prediction_shape == (2,)

    # Stage 3: rollout obeys the codec invariants (channel 2 is held constant).
    prediction = trained.rollout(states[:, 0], inputs)
    assert prediction.shape == states.shape
    assert torch.equal(prediction[..., 2], states[:, :1, 2].expand(-1, 5))

    # Stage 4: save, reject codeless load, reload with the supplied codec.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "custom_nerd.pt"
        trained.save(path)
        with pytest.raises(ValueError, match="custom NeRD codec"):
            type(trained).load(path)
        loaded = type(trained).load(path, codec=codec)
    assert loaded.codec is codec


def test_training_config_and_zero_active_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="epochs"):
        NeRDTrainingConfig(epochs=0)
    codec = NeRDParticleStateCodec(_model())
    with pytest.raises(ValueError, match="no active"):
        train_nerd(
            NeRDDataset(states=torch.zeros(2, 4, 2, 6), codec=codec),
            NeRDTrainingConfig(
                context_frames=2,
                epochs=1,
                steps_per_epoch=1,
                batch_size=2,
            ),
            dynamics_model="FullyConnected",
            model_kwargs={"layer_size": 8, "num_layers": 1},
            device="cpu",
            log=lambda _message: None,
        )


def test_dataset_validates_trajectory_contract() -> None:
    codec = NeRDParticleStateCodec(_model())
    with pytest.raises(ValueError, match="two frames"):
        NeRDDataset(states=torch.zeros(2, 1, 2, 6), codec=codec)
    with pytest.raises(ValueError, match="one frame for every state transition"):
        NeRDDataset(
            states=torch.zeros(2, 3, 2, 6),
            inputs=torch.zeros(2, 1, 0),
            codec=codec,
        )


def test_joint_step_model_requires_a_live_deployment_model() -> None:
    codec = NeRDJointStateCodec(layout=_fake_joint_layout(), robot_centric=False)
    trained = TrainedNeRDModel(
        model=torch.nn.Linear(4, 4),
        normalizers=NeRDNormalizers(
            input_mean=torch.zeros(1, 1, 4),
            input_std=torch.ones(1, 1, 4),
            target_mean=torch.zeros(1, 1, 4),
            target_std=torch.ones(1, 1, 4),
        ),
        codec=codec,
        config=NeRDTrainingConfig(context_frames=1),
        external_input_shape=(0,),
        active_delta_mask=torch.ones(4),
        frame_dt=0.1,
    )
    with pytest.raises(ValueError, match="live state_codec"):
        trained.as_step_model(device="cpu")


def test_joint_codec_finalize_refreshes_body_state() -> None:
    _require_newton()
    from newton.examples.robot.example_robot_cartpole import Example as CartpoleScene

    scene = load_example_scene(CartpoleScene, device="cpu")
    state = scene.model.state()
    codec = NeRDJointStateCodec(scene.model, robot_centric=False)
    before = torch.clone(torch.as_tensor(state.body_q.numpy()))
    value = codec.read(state)
    value[:, 0] += 0.25
    codec.write(state, value)
    codec.finalize_state(state)
    after = torch.as_tensor(state.body_q.numpy())
    assert not torch.allclose(before, after)


def test_joint_codec_ball_joint_uses_newton_quaternion_layout() -> None:
    newton = _require_newton()
    builder = newton.ModelBuilder()
    anchor = builder.add_link(xform=wp.transform_identity(), label="anchor")
    body = builder.add_link(xform=wp.transform_identity(), label="body")
    fixed = builder.add_joint_fixed(parent=-1, child=anchor, label="fixed")
    ball = builder.add_joint_ball(parent=anchor, child=body, label="ball")
    builder.add_articulation([fixed, ball])
    model = builder.finalize(device="cpu")
    codec = NeRDJointStateCodec(model, robot_centric=False)

    assert codec.layout.quaternion_q_starts == (0,)
    assert codec.state_shape == (7,)
    assert codec.prediction_shape == (6,)

    state = model.state()
    current = codec.read(state)
    delta = torch.zeros(1, 6)
    delta[0, 2] = 0.25
    next_state = codec.delta_to_state(current, delta)
    codec.write(state, next_state)

    quaternion = torch.as_tensor(state.joint_q.numpy())
    assert float(torch.linalg.vector_norm(quaternion)) == pytest.approx(1.0)
