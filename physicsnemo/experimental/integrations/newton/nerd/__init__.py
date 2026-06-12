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

"""Reusable Neural Robot Dynamics tools for Newton simulations.

The Newton-specific part of a learned dynamics model is not the neural network.
It is the state representation and the simulation lifecycle:

* which Newton fields form one model state,
* how relative dynamics are encoded and integrated,
* how a particular application resets and advances one learned frame, and
* how a prediction is written back into live Newton state.

This module makes those boundaries explicit and reusable:

* :class:`NeRDStateCodec` implementations cover generalized joints, rigid
  bodies, particles, and composites of those representations.
* :class:`NeRDProblem` adapts either :class:`NewtonEnv` or an arbitrary existing
  Newton run to the small reset/state/advance interface needed for collection.
* :func:`fit_nerd` performs distributed teacher collection and training.
* :meth:`TrainedNeRDModel.as_step_model` deploys the trained model through the
  same :class:`~physicsnemo.experimental.integrations.newton.NewtonStepModel`
  contract used by :class:`NewtonEnv`.

Within an explicitly selected state representation, the library derives entity
layouts, model dimensions, normalization, causal windows, distributed sharding,
and deployment metadata. It intentionally does not try to infer
application-defined domain randomization, controllers, or scene-specific step
logic; users provide those through the problem runner.

The implementation is split into cohesive submodules (rotation/frame helpers,
codecs, spec/config, problem/data, model construction, checkpointing, training,
and deployment); this package re-exports the full public surface so the import
path ``physicsnemo.experimental.integrations.newton.nerd.<name>`` is unchanged.
"""

from __future__ import annotations

from physicsnemo.experimental.integrations.newton.nerd.checkpoint import (
    _codec_descriptor,
    _codec_from_descriptor,
    _deployment_codec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs import (
    JointLayout,
    NeRDBodyHeadingFrame,
    NeRDBodyStateCodec,
    NeRDCompositeStateCodec,
    NeRDFixedFrame,
    NeRDJointStateCodec,
    NeRDParticleStateCodec,
    NeRDStateCodec,
    nerd_state_codec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.base import (
    _body_label,
    _entity_semantic_ids,
    _model_labels,
    _normalized_label,
    _optional_numpy_field,
    _relative_index_signature,
    _uniform_world_indices,
    _validate_state_value,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.entity import (
    _body_state_to_delta,
    _body_state_tokens,
    _canonicalize_body_state,
    _delta_to_body_state,
    _NeRDEntityStateCodec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs.joint import (
    _anchor_base,
    _delta_to_next_state,
    _joint_coordinate_indices,
    _joint_layout,
    _next_state_to_delta,
    _state_to_input,
)
from physicsnemo.experimental.integrations.newton.nerd.deploy import (
    NeRDRolloutEvaluation,
    NeRDStepModel,
    TrainedNeRDModel,
    _NeRDStepModel,
    evaluate_nerd,
)
from physicsnemo.experimental.integrations.newton.nerd.inputs import (
    NeRDControlInput,
    NeRDRigidContactInput,
    concatenate_nerd_inputs,
)
from physicsnemo.experimental.integrations.newton.nerd.model_builders import (
    _checkpoint_model,
    _checkpoint_model_name,
    _CompiledNeRDModel,
    _model_descriptor,
    _model_from_descriptor,
    _registered_model_dimensions,
    _resolve_nerd_model,
    _supported_checkpoint_models,
)
from physicsnemo.experimental.integrations.newton.nerd.problem import (
    NeRDDataset,
    NeRDProblem,
    collect_nerd_trajectories,
)
from physicsnemo.experimental.integrations.newton.nerd.rotation import (
    _base_frame_transform,
    _canonicalize_quat,
    _normalize_quat,
    _quat_inverse,
    _quat_mul,
    _quat_rotate_inverse,
    _quat_to_rotvec,
    _rotvec_to_quat,
    _world_frame_transform,
    _wrap_to_pi,
)
from physicsnemo.experimental.integrations.newton.nerd.runtime import (
    NeRDNormalizers,
    _append_inputs,
    _global_abs_max,
    _global_mean_std,
    _input_tensor,
    _input_width,
    _model_class_name,
    _model_for_device,
    _model_prediction,
    _predict_delta,
    _runtime_device,
    _state_rmse,
)
from physicsnemo.experimental.integrations.newton.nerd.spec import (
    NeRDModelSpec,
    NeRDTrainingConfig,
)
from physicsnemo.experimental.integrations.newton.nerd.training import (
    fit_nerd,
    train_nerd,
)

__all__ = [
    "JointLayout",
    "NeRDBodyHeadingFrame",
    "NeRDBodyStateCodec",
    "NeRDCompositeStateCodec",
    "NeRDControlInput",
    "NeRDDataset",
    "NeRDJointStateCodec",
    "NeRDModelSpec",
    "NeRDNormalizers",
    "NeRDParticleStateCodec",
    "NeRDFixedFrame",
    "NeRDProblem",
    "NeRDRigidContactInput",
    "NeRDRolloutEvaluation",
    "NeRDStateCodec",
    "NeRDStepModel",
    "NeRDTrainingConfig",
    "TrainedNeRDModel",
    "collect_nerd_trajectories",
    "concatenate_nerd_inputs",
    "evaluate_nerd",
    "fit_nerd",
    "nerd_state_codec",
    "train_nerd",
]
