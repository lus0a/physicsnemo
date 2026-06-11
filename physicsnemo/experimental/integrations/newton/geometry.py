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

"""Declarative rigid geometry shared by PhysicsNeMo models and Newton."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from physicsnemo.mesh import Mesh

PrimitiveKind = Literal["box", "sphere", "cylinder", "capsule"]
MeshCollision = Literal["convex_hull", "mesh"]


def _validated_transform(
    position: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    translation = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(quaternion, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("position must contain three finite values")
    if rotation.shape != (4,) or not np.isfinite(rotation).all():
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(rotation))
    if norm <= 1.0e-12:
        raise ValueError("quaternion must have nonzero norm")
    rotation /= norm
    # q and -q encode the same rotation. Canonicalizing the sign keeps cache
    # fingerprints stable for physically identical transforms.
    for component in (rotation[3], rotation[0], rotation[1], rotation[2]):
        if abs(component) > 1.0e-15:
            if component < 0.0:
                rotation *= -1.0
            break
    return tuple(translation.tolist()), tuple(rotation.tolist())


def _rotation_matrix(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def _transform_points(
    points: np.ndarray,
    position: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    rotation = _rotation_matrix(quaternion)
    return (
        np.asarray(points, dtype=np.float64) @ rotation.T
        + np.asarray(position, dtype=np.float64)
    ).astype(np.float32)


@dataclass(frozen=True)
class NewtonPrimitive:
    """One analytic Newton collision primitive in an object's local frame.

    ``box`` sizes are half extents. ``sphere`` uses ``(radius,)``.
    ``cylinder`` and ``capsule`` use ``(radius, half_height)`` and are aligned
    with local z before applying ``quaternion``.
    """

    kind: PrimitiveKind
    size: tuple[float, ...]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        expected = {"box": 3, "sphere": 1, "cylinder": 2, "capsule": 2}
        if self.kind not in expected:
            raise ValueError(f"unsupported primitive kind: {self.kind!r}")
        size = tuple(float(value) for value in self.size)
        if len(size) != expected[self.kind] or any(
            not np.isfinite(value) or value <= 0.0 for value in size
        ):
            raise ValueError(
                f"{self.kind} size must contain {expected[self.kind]} "
                "positive finite values"
            )
        position, quaternion = _validated_transform(
            self.position,
            self.quaternion,
        )
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion)

    @property
    def volume(self) -> float:
        """Return the analytic primitive volume."""
        if self.kind == "box":
            return float(8.0 * np.prod(self.size))
        if self.kind == "sphere":
            return 4.0 * np.pi * self.size[0] ** 3 / 3.0
        radius, half_height = self.size
        cylinder = 2.0 * np.pi * radius**2 * half_height
        if self.kind == "cylinder":
            return cylinder
        return cylinder + 4.0 * np.pi * radius**3 / 3.0

    @property
    def surface_area(self) -> float:
        """Return the analytic primitive surface area."""
        if self.kind == "box":
            x, y, z = self.size
            return 8.0 * (x * y + x * z + y * z)
        if self.kind == "sphere":
            return 4.0 * np.pi * self.size[0] ** 2
        radius, half_height = self.size
        cylinder_side = 4.0 * np.pi * radius * half_height
        if self.kind == "cylinder":
            return cylinder_side + 2.0 * np.pi * radius**2
        return cylinder_side + 4.0 * np.pi * radius**2

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return transformed axis-aligned lower and upper bounds."""
        rotation = _rotation_matrix(self.quaternion)
        center = np.asarray(self.position, dtype=np.float64)
        if self.kind == "box":
            extent = np.abs(rotation) @ np.asarray(self.size)
        elif self.kind == "sphere":
            extent = np.full(3, self.size[0])
        else:
            radius, half_height = self.size
            axis = rotation[:, 2]
            axial = half_height * np.abs(axis)
            if self.kind == "capsule":
                # The hemispherical caps make the bound isotropic in radius.
                extent = axial + radius
            else:
                # A cylinder's radial extent projects onto each axis.
                extent = axial + radius * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
        return (center - extent).astype(np.float32), (center + extent).astype(
            np.float32
        )


@dataclass(frozen=True)
class NewtonMesh:
    """PhysicsNeMo triangle surface mesh with a Newton collision policy."""

    mesh: Mesh
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    collision: MeshCollision = "convex_hull"

    def __post_init__(self) -> None:
        if self.mesh.n_manifold_dims != 2 or self.mesh.n_spatial_dims != 3:
            raise ValueError("NewtonMesh requires a triangle surface Mesh[2, 3]")
        if self.mesh.n_cells == 0:
            raise ValueError("NewtonMesh requires at least one triangle")
        if self.collision not in {"convex_hull", "mesh"}:
            raise ValueError("collision must be 'convex_hull' or 'mesh'")
        position, quaternion = _validated_transform(
            self.position,
            self.quaternion,
        )
        if not bool(torch.isfinite(self.mesh.points).all()):
            raise ValueError("mesh points must be finite")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion)
        volume = _closed_mesh_volume(self.mesh)
        if volume <= 1.0e-12:
            raise ValueError("mesh must enclose a nonzero volume")

    @classmethod
    def from_arrays(
        cls,
        vertices: np.ndarray,
        triangles: np.ndarray,
        **kwargs: Any,
    ) -> NewtonMesh:
        """Construct from array geometry produced by any mesh loader."""
        return cls(
            mesh=Mesh(
                points=torch.as_tensor(vertices, dtype=torch.float32),
                cells=torch.as_tensor(triangles, dtype=torch.int64),
            ),
            **kwargs,
        )

    @property
    def volume(self) -> float:
        """Enclosed volume, summed over consistently oriented closed components."""
        return _closed_mesh_volume(self.mesh)

    @property
    def surface_area(self) -> float:
        """Return the triangle-mesh surface area."""
        vertices = self.mesh.points[self.mesh.cells]
        vectors = vertices[:, 1:] - vertices[:, :1]
        from physicsnemo.mesh.geometry import compute_cell_areas  # noqa: PLC0415

        return float(compute_cell_areas(vectors).sum())

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return transformed axis-aligned lower and upper bounds."""
        points = _transform_points(
            self.mesh.points.detach().cpu().numpy(),
            self.position,
            self.quaternion,
        )
        return points.min(axis=0), points.max(axis=0)


GeometryPart = NewtonPrimitive | NewtonMesh


@dataclass(frozen=True)
class NewtonRigidObject:
    """Named compound rigid geometry for simulation and neural features."""

    name: str
    density: float
    parts: tuple[GeometryPart, ...]
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        tags = tuple(self.tags)
        color = tuple(float(value) for value in self.color)
        if not self.name:
            raise ValueError("object name must not be empty")
        if not parts:
            raise ValueError("a rigid object requires at least one geometry part")
        if not np.isfinite(self.density) or self.density <= 0.0:
            raise ValueError("density must be positive and finite")
        if len(color) != 3 or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in color
        ):
            raise ValueError("color must contain three finite values in [0, 1]")
        if len(set(tags)) != len(tags):
            raise ValueError("object tags must be unique")
        object.__setattr__(self, "density", float(self.density))
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "tags", tags)

    @classmethod
    def from_mesh(
        cls,
        name: str,
        mesh: Mesh,
        *,
        density: float,
        color: tuple[float, float, float] = (0.7, 0.7, 0.7),
        tags: tuple[str, ...] = (),
        collision: MeshCollision = "convex_hull",
    ) -> NewtonRigidObject:
        """Create a rigid object from a PhysicsNeMo surface mesh."""
        return cls(
            name=name,
            density=density,
            color=color,
            tags=tags,
            parts=(NewtonMesh(mesh=mesh, collision=collision),),
        )

    @property
    def volume(self) -> float:
        """Additive part volume.

        Overlapping compound parts are counted independently. This matches
        Newton's per-shape inertia accumulation but is not a geometric union.
        """
        return float(sum(part.volume for part in self.parts))

    @property
    def approximate_mass(self) -> float:
        """Density times additive part volume."""
        return self.density * self.volume

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return union bounds over every geometry part."""
        bounds = tuple(part.bounds for part in self.parts)
        return (
            np.min(np.stack([lower for lower, _ in bounds]), axis=0),
            np.max(np.stack([upper for _, upper in bounds]), axis=0),
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable geometry and material fingerprint."""
        return rigid_object_fingerprint((self,))

    def sample_surface(
        self,
        num_points: int,
        *,
        seed: int = 0,
    ) -> np.ndarray:
        """Return an area-uniform XYZ point cloud over all geometry parts."""
        if num_points < len(self.parts):
            raise ValueError("num_points must be at least the number of parts")
        rng = np.random.default_rng(seed)
        weights = np.asarray(
            [part.surface_area for part in self.parts],
            dtype=np.float64,
        )
        counts = _allocate_counts(weights, num_points)
        samples = []
        for part, count in zip(self.parts, counts):
            if isinstance(part, NewtonPrimitive):
                local = _sample_primitive(part, int(count), rng)
            else:
                generator = torch.Generator(device=part.mesh.points.device)
                generator.manual_seed(int(rng.integers(0, np.iinfo(np.int64).max)))
                local = (
                    part.mesh.sample_random_points(
                        int(count),
                        generator=generator,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            samples.append(_transform_points(local, part.position, part.quaternion))
        return np.concatenate(samples, axis=0)


def rigid_object_fingerprint(objects: tuple[NewtonRigidObject, ...]) -> str:
    """Hash object geometry and physical metadata for cache compatibility."""
    digest = hashlib.sha256()
    for rigid_object in objects:
        header = {
            "name": rigid_object.name,
            "density": rigid_object.density,
            "color": rigid_object.color,
            "tags": rigid_object.tags,
        }
        digest.update(json.dumps(header, sort_keys=True).encode())
        for part in rigid_object.parts:
            transform = {
                "position": part.position,
                "quaternion": part.quaternion,
            }
            if isinstance(part, NewtonPrimitive):
                transform.update({"kind": part.kind, "size": part.size})
                digest.update(json.dumps(transform, sort_keys=True).encode())
            else:
                transform["collision"] = part.collision
                digest.update(json.dumps(transform, sort_keys=True).encode())
                for array in (part.mesh.points, part.mesh.cells):
                    blob = (
                        array.detach().to(device="cpu").contiguous().numpy().tobytes()
                    )
                    # Length-prefix each variable-length mesh blob so the
                    # points/cells boundary is unambiguous and two distinct
                    # geometries cannot collide by re-splitting the byte stream.
                    digest.update(struct.pack("<Q", len(blob)))
                    digest.update(blob)
    return digest.hexdigest()


def add_rigid_object_shapes(
    builder: Any,
    body: int,
    rigid_object: NewtonRigidObject,
    cfg: Any | None = None,
    *,
    label_prefix: str | None = None,
) -> tuple[int, ...]:
    """Add a declarative rigid object's shapes to a Newton model builder.

    Newton and Warp are imported only when this adapter is called. When ``cfg``
    is omitted, a Newton shape configuration using ``rigid_object.density`` is
    created. A supplied configuration must use the same density so simulation
    mass cannot silently disagree with the declarative object.
    """
    import warp as wp  # noqa: PLC0415

    from physicsnemo.experimental.integrations.newton.dependencies import require_newton

    newton = require_newton()
    if cfg is None:
        cfg = newton.ModelBuilder.ShapeConfig(density=rigid_object.density)
    cfg_density = getattr(cfg, "density", None)
    if cfg_density is None:
        raise TypeError("cfg must expose a density attribute")
    if not np.isclose(
        float(cfg_density),
        rigid_object.density,
        rtol=1.0e-7,
        atol=0.0,
    ):
        raise ValueError(
            "cfg density must match rigid_object.density; "
            f"got {float(cfg_density)} and {rigid_object.density}"
        )

    prefix = label_prefix or rigid_object.name
    shape_ids = []
    for index, part in enumerate(rigid_object.parts):
        xform = wp.transform(
            wp.vec3(*part.position),
            wp.quat(*part.quaternion),
        )
        kwargs = {
            "body": body,
            "xform": xform,
            "cfg": cfg,
            "color": rigid_object.color,
            "label": f"{prefix}_{index}",
        }
        if isinstance(part, NewtonPrimitive):
            if part.kind == "box":
                shape = builder.add_shape_box(
                    hx=part.size[0],
                    hy=part.size[1],
                    hz=part.size[2],
                    **kwargs,
                )
            elif part.kind == "sphere":
                shape = builder.add_shape_sphere(radius=part.size[0], **kwargs)
            elif part.kind == "cylinder":
                shape = builder.add_shape_cylinder(
                    radius=part.size[0],
                    half_height=part.size[1],
                    **kwargs,
                )
            else:
                shape = builder.add_shape_capsule(
                    radius=part.size[0],
                    half_height=part.size[1],
                    **kwargs,
                )
        else:
            mesh = newton.Mesh(
                vertices=part.mesh.points.detach().cpu().numpy(),
                indices=part.mesh.cells.detach().cpu().numpy().reshape(-1),
            )
            add_mesh = (
                builder.add_shape_convex_hull
                if part.collision == "convex_hull"
                else builder.add_shape_mesh
            )
            shape = add_mesh(mesh=mesh, **kwargs)
        shape_ids.append(shape)
    return tuple(shape_ids)


def _allocate_counts(weights: np.ndarray, total: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("surface areas must be a non-empty one-dimensional array")
    if total < len(weights):
        raise ValueError("total must be at least the number of surface parts")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("surface areas must be positive and finite")
    raw = total * weights / weights.sum()
    counts = np.maximum(1, np.floor(raw).astype(np.int64))
    while counts.sum() < total:
        counts[int(np.argmax(raw - counts))] += 1
    while counts.sum() > total:
        eligible = counts > 1
        if not bool(eligible.any()):
            raise RuntimeError(
                "could not allocate at least one sample per surface part"
            )
        excess = np.where(eligible, counts - raw, -np.inf)
        counts[int(np.argmax(excess))] -= 1
    return counts


def _closed_mesh_volume(mesh: Mesh) -> float:
    """Validate a closed oriented triangle complex and return its volume."""
    cells = mesh.cells.detach().to(device="cpu", dtype=torch.int64).numpy()
    points = mesh.points.detach().to(device="cpu", dtype=torch.float64).numpy()
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("NewtonMesh requires triangular cells")

    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face, triangle in enumerate(cells):
        if len(set(int(index) for index in triangle)) != 3:
            raise ValueError("mesh triangles must contain three distinct vertices")
        vertices = points[triangle]
        area_vector = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        if float(np.linalg.norm(area_vector)) <= 1.0e-12:
            raise ValueError("mesh triangles must have nonzero area")
        for start, end in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            edge = (min(start, end), max(start, end))
            direction = 1 if (start, end) == edge else -1
            edge_faces.setdefault(edge, []).append((face, direction))

    adjacency: list[list[int]] = [[] for _ in range(len(cells))]
    for incident in edge_faces.values():
        if len(incident) != 2:
            raise ValueError(
                "mesh must be closed: every edge must belong to exactly two triangles"
            )
        (first, first_direction), (second, second_direction) = incident
        if first_direction == second_direction:
            raise ValueError("mesh triangles must use consistent winding")
        adjacency[first].append(second)
        adjacency[second].append(first)

    triangle_volume = (
        np.einsum(
            "ij,ij->i",
            points[cells[:, 0]],
            np.cross(points[cells[:, 1]], points[cells[:, 2]]),
        )
        / 6.0
    )
    visited = np.zeros(len(cells), dtype=bool)
    volume = 0.0
    for first_face in range(len(cells)):
        if visited[first_face]:
            continue
        stack = [first_face]
        visited[first_face] = True
        component_faces = []
        while stack:
            face = stack.pop()
            component_faces.append(face)
            for neighbor in adjacency[face]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        volume += abs(float(triangle_volume[component_faces].sum()))
    return volume


def _sample_primitive(
    primitive: NewtonPrimitive,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if primitive.kind == "box":
        half = np.asarray(primitive.size, dtype=np.float32)
        points = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32) * half
        areas = np.asarray(
            (half[1] * half[2], half[0] * half[2], half[0] * half[1]),
        )
        faces = rng.choice(3, size=count, p=areas / areas.sum())
        signs = rng.choice(np.asarray((-1.0, 1.0), dtype=np.float32), size=count)
        points[np.arange(count), faces] = signs * half[faces]
    elif primitive.kind == "sphere":
        directions = rng.normal(size=(count, 3)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = directions * primitive.size[0]
    elif primitive.kind == "cylinder":
        radius, half_height = primitive.size
        side_area = 4.0 * np.pi * radius * half_height
        cap_area = 2.0 * np.pi * radius**2
        side = rng.random(count) < side_area / (side_area + cap_area)
        angles = rng.uniform(0.0, 2.0 * np.pi, count)
        points = np.empty((count, 3), dtype=np.float32)
        points[:, 0] = radius * np.cos(angles)
        points[:, 1] = radius * np.sin(angles)
        points[:, 2] = rng.uniform(-half_height, half_height, count)
        cap = ~side
        radial = np.sqrt(rng.random(cap.sum()))
        points[cap, 0] *= radial
        points[cap, 1] *= radial
        points[cap, 2] = rng.choice((-half_height, half_height), cap.sum())
    else:
        radius, half_height = primitive.size
        side_area = 4.0 * np.pi * radius * half_height
        sphere_area = 4.0 * np.pi * radius**2
        cylinder = rng.random(count) < side_area / (side_area + sphere_area)
        angles = rng.uniform(0.0, 2.0 * np.pi, count)
        points = np.empty((count, 3), dtype=np.float32)
        points[cylinder, 0] = radius * np.cos(angles[cylinder])
        points[cylinder, 1] = radius * np.sin(angles[cylinder])
        points[cylinder, 2] = rng.uniform(
            -half_height,
            half_height,
            cylinder.sum(),
        )
        directions = rng.normal(size=((~cylinder).sum(), 3)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points[~cylinder] = radius * directions
        points[~cylinder, 2] += np.where(
            points[~cylinder, 2] >= 0.0,
            half_height,
            -half_height,
        )
    return points


__all__ = [
    "GeometryPart",
    "MeshCollision",
    "NewtonMesh",
    "NewtonPrimitive",
    "NewtonRigidObject",
    "PrimitiveKind",
    "add_rigid_object_shapes",
    "rigid_object_fingerprint",
]
