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

"""Tests for mesh point normalization."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.transformations import normalize_points


def test_normalize_points_preserves_mesh_context(device):
    points = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [2.0, 4.0, 3.0]],
        device=device,
    )
    mesh = Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2]], device=device),
        point_data={"value": torch.arange(3.0, device=device)},
    )
    original_area = mesh.cell_areas

    normalized, centroid, radius = normalize_points(mesh)

    assert normalized.points.shape == points.shape
    assert centroid.shape == (3,)
    assert radius.shape == ()
    assert torch.equal(normalized.cells, mesh.cells)
    assert torch.equal(normalized.point_data["value"], mesh.point_data["value"])
    torch.testing.assert_close(
        normalized.points.mean(dim=0),
        torch.zeros_like(centroid),
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized.points, dim=-1).amax(),
        torch.ones((), device=device),
    )
    torch.testing.assert_close(normalized.points * radius + centroid, points)
    torch.testing.assert_close(normalized.cell_areas, original_area / radius.square())


def test_normalize_points_method_handles_degenerate_mesh():
    points = torch.full((7, 3), 4.0)
    mesh = Mesh(points=points)

    normalized, centroid, radius = mesh.normalize_points()

    assert torch.equal(normalized.points, torch.zeros_like(points))
    assert torch.equal(centroid, torch.full((3,), 4.0))
    assert radius == pytest.approx(1.0e-8)


@pytest.mark.parametrize("eps", [0.0, -1.0])
def test_normalize_points_requires_positive_epsilon(eps):
    with pytest.raises(ValueError, match="eps must be positive"):
        normalize_points(Mesh(points=torch.zeros((1, 3))), eps=eps)
