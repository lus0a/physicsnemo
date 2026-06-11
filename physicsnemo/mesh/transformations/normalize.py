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

"""Normalization transformations for meshes."""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from physicsnemo.mesh.transformations.geometric import scale, translate

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def normalize_points(
    mesh: "Mesh",
    *,
    eps: float = 1.0e-8,
) -> tuple[
    "Mesh",
    Float[torch.Tensor, " n_spatial_dims"],
    Float[torch.Tensor, ""],
]:
    """Center and isotropically scale the points of a mesh.

    The arithmetic mean of the mesh vertices is translated to the origin, then
    all coordinates are divided by the maximum vertex distance from that mean.
    The resulting points therefore lie within the unit ball. Connectivity,
    attached data, and compatible cached geometry are preserved by the standard
    mesh transformation machinery.

    Parameters
    ----------
    mesh : Mesh
        Mesh whose point coordinates will be normalized. The operation supports
        meshes of any manifold or spatial dimension.
    eps : float, optional
        Positive lower bound for the returned radius. This avoids division by
        zero when all mesh points coincide.

    Returns
    -------
    Mesh
        New mesh with centered, isotropically scaled point coordinates.
    Float[torch.Tensor, " n_spatial_dims"]
        Arithmetic mean of the original point coordinates.
    Float[torch.Tensor, ""]
        Maximum distance from the centroid, clamped to at least ``eps``. The
        original coordinates can be reconstructed as
        ``normalized.points * radius + centroid``.

    Raises
    ------
    ValueError
        If ``eps`` is not positive, the mesh has no points or spatial
        dimensions, or its point coordinates are non-finite.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(radius=2.0).translate([1.0, 0.0, 0.0])
    >>> normalized, centroid, radius = normalize_points(mesh)
    >>> reconstructed = normalized.points * radius + centroid
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not torch.compiler.is_compiling():
        if mesh.n_points == 0 or mesh.n_spatial_dims == 0:
            raise ValueError("mesh must contain points and spatial dimensions")
        if not torch.isfinite(mesh.points).all():
            raise ValueError("mesh points must be finite")

    centroid = mesh.points.mean(dim=0)
    centered = translate(mesh, -centroid)
    radius = torch.linalg.vector_norm(centered.points, dim=-1).amax().clamp_min(eps)
    normalized = scale(centered, 1.0 / radius, assume_invertible=True)
    return normalized, centroid, radius
