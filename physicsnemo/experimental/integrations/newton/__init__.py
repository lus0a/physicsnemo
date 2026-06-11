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

"""PhysicsNeMo adapters for Newton simulation, learning, and deployment.

The optional integration provides zero-copy Warp/Torch state views, a reusable
:class:`NewtonEnv` lifecycle, differentiable and simulator-in-the-loop learning
workflows, and NeRD learned solver replacements.

Exports are loaded lazily so importing this package does not import Newton or
optional PhysicsNeMo model/datapipe stacks until a feature is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Sorted alphabetically by key to match the public surface (``__all__`` below
# is ``sorted(_EXPORTS)``) and keep additions as localized diffs.
_EXPORTS = {
    "BPTTSurrogate": "surrogate",
    "DesignRegularizer": "design_space",
    "DesignResult": "design",
    "DesignSpace": "design_space",
    "DesignSurrogate": "design",
    "DesignVariable": "design_space",
    "DifferentiableRollout": "components",
    "GroupedDesignResult": "codesign",
    "NeRDBodyHeadingFrame": "nerd",
    "NeRDControlInput": "nerd",
    "NeRDFixedFrame": "nerd",
    "NeRDProblem": "nerd",
    "NeRDRigidContactInput": "nerd",
    "NeRDStepModel": "nerd",
    "NeRDTrainingConfig": "nerd",
    "NewtonComponents": "components",
    "NewtonEnv": "env",
    "NewtonMesh": "geometry",
    "NewtonPrimitive": "geometry",
    "NewtonRigidObject": "geometry",
    "NewtonRollout": "components",
    "NewtonStepModel": "step_model",
    "ResidualDynamics": "surrogate",
    "SimilarityConstraint": "design_space",
    "SmoothnessConstraint": "design_space",
    "TeacherBatch": "surrogate",
    "TeacherSample": "surrogate",
    "TrainedNeRDModel": "nerd",
    "VerifiedDesignSelection": "design_space",
    "add_rigid_object_shapes": "geometry",
    "bodies": "state",
    "collect_teacher_batch": "surrogate",
    "concatenate_nerd_inputs": "nerd",
    "differentiable_rollout": "adjoint",
    "field_to_torch": "data",
    "fit_nerd": "nerd",
    "grouped_candidate_ranking_loss": "codesign",
    "is_main_process": "distributed",
    "joints": "state",
    "load_example_scene": "scene",
    "optimize_design": "design",
    "optimize_field_in_newton": "adjoint",
    "optimize_field_in_newton_multistart": "adjoint",
    "optimize_grouped_design": "codesign",
    "particles": "state",
    "resolve_device": "distributed",
    "rigid_object_fingerprint": "geometry",
    "select_diverse_designs": "design_space",
    "select_verified_design": "design_space",
    "shortlist_grouped_candidates": "codesign",
    "trajectory_dataset": "trajectory",
    "write_state_fields": "step_model",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public integration feature when first requested."""

    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
