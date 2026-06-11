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

import numpy as np
import pytest
import torch

from physicsnemo.experimental.integrations.newton import (
    NewtonMesh,
    NewtonPrimitive,
    NewtonRigidObject,
    add_rigid_object_shapes,
    rigid_object_fingerprint,
)
from physicsnemo.experimental.integrations.newton.geometry import _allocate_counts
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import tetrahedron_surface

# Newton is an optional extra; skip the whole module (rather than erroring at
# collection) when it is absent. The integration itself never imports Newton at
# import time, so the first-party imports above stay unguarded.
newton = pytest.importorskip("newton")


def test_compound_rigid_object_sampling_is_deterministic() -> None:
    rigid_object = NewtonRigidObject(
        "compound",
        density=400.0,
        parts=(
            NewtonPrimitive("box", (0.03, 0.02, 0.04)),
            NewtonPrimitive(
                "sphere",
                (0.015,),
                position=(0.035, 0.0, 0.02),
            ),
        ),
        tags=("train",),
    )
    first = rigid_object.sample_surface(128, seed=9)
    second = rigid_object.sample_surface(128, seed=9)
    assert first.shape == (128, 3)
    np.testing.assert_array_equal(first, second)
    assert rigid_object.approximate_mass > 0.0
    lower, upper = rigid_object.bounds
    assert np.all(lower < upper)
    assert len(rigid_object.fingerprint) == 64


def test_physicsnemo_mesh_is_shared_with_newton_adapter() -> None:
    mesh = tetrahedron_surface.load(side_length=0.08)
    rigid_object = NewtonRigidObject.from_mesh(
        "mesh-object",
        mesh,
        density=350.0,
        tags=("holdout",),
    )
    points = rigid_object.sample_surface(96, seed=5)
    assert points.shape == (96, 3)
    assert np.isfinite(points).all()

    builder = newton.ModelBuilder()
    body = builder.add_body()
    shape_ids = add_rigid_object_shapes(
        builder,
        body,
        rigid_object,
        newton.ModelBuilder.ShapeConfig(density=rigid_object.density),
    )
    assert len(shape_ids) == 1


def test_rigid_object_fingerprint_tracks_geometry() -> None:
    first = NewtonRigidObject(
        "sphere",
        density=300.0,
        parts=(NewtonPrimitive("sphere", (0.04,)),),
    )
    second = NewtonRigidObject(
        "sphere",
        density=300.0,
        parts=(NewtonPrimitive("sphere", (0.05,)),),
    )
    assert rigid_object_fingerprint((first,)) != rigid_object_fingerprint((second,))


def test_rigid_object_fingerprint_canonicalizes_quaternion_sign() -> None:
    positive = NewtonRigidObject(
        "box",
        density=300.0,
        parts=(NewtonPrimitive("box", (0.04, 0.03, 0.02), quaternion=(1, 2, 3, 4)),),
    )
    negative = NewtonRigidObject(
        "box",
        density=300.0,
        parts=(
            NewtonPrimitive("box", (0.04, 0.03, 0.02), quaternion=(-1, -2, -3, -4)),
        ),
    )
    assert positive.fingerprint == negative.fingerprint


def test_surface_count_allocation_terminates_with_minimum_counts() -> None:
    np.testing.assert_array_equal(
        _allocate_counts(np.asarray((0.9, 0.05, 0.05)), 3),
        np.ones(3, dtype=np.int64),
    )


def test_newton_mesh_rejects_non_surface_and_open_meshes() -> None:
    with pytest.raises(ValueError, match="surface"):
        NewtonMesh(
            Mesh(
                points=torch.zeros((4, 3)),
                cells=torch.tensor(((0, 1, 2, 3),)),
            )
        )
    with pytest.raises(ValueError, match="closed"):
        NewtonMesh(
            Mesh(
                points=torch.tensor(
                    (
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    )
                ),
                cells=torch.tensor(((1, 2, 3),)),
            )
        )


def test_newton_mesh_rejects_inconsistent_winding() -> None:
    mesh = tetrahedron_surface.load(side_length=0.08)
    cells = mesh.cells.clone()
    cells[0, :2] = cells[0, :2].flip(0)
    with pytest.raises(ValueError, match="winding"):
        NewtonMesh(Mesh(points=mesh.points, cells=cells))


def test_newton_shape_adapter_enforces_density() -> None:
    rigid_object = NewtonRigidObject(
        "sphere",
        density=300.0,
        parts=(NewtonPrimitive("sphere", (0.04,)),),
    )
    builder = newton.ModelBuilder()
    body = builder.add_body()
    with pytest.raises(ValueError, match="density"):
        add_rigid_object_shapes(
            builder,
            body,
            rigid_object,
            newton.ModelBuilder.ShapeConfig(density=200.0),
        )
