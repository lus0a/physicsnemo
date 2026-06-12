# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Headless, replicated Newton RJ45 scene used for NeRD data generation.

Newton's interactive RJ45 example finalizes one model internally. This module
keeps the same physical construction but separates a reusable single-world
blueprint from the batched runtime, allowing VBD to advance many independent
worlds in one solver call.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import newton
import newton.examples
import newton.utils
import numpy as np
import torch
import warp as wp
from newton.math import quat_between_vectors_robust
from newton.solvers import SolverVBD

from physicsnemo.experimental.integrations.newton.data import torch_warp_stream

CONTACT_KE = 1.0e5
CONTACT_KD = 0.0
SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    mu=0.0,
    ke=CONTACT_KE,
    kd=CONTACT_KD,
    gap=0.002,
    density=1.0e6,
    mu_torsional=0.0,
    mu_rolling=0.0,
)
MESH_SDF_MAX_RESOLUTION = 128
MESH_SDF_NARROW_BAND_RANGE = (-2.0 * SHAPE_CFG.gap, 2.0 * SHAPE_CFG.gap)
PLUG_Y_OFFSET = -0.025
CABLE_RADIUS = 0.00325
CABLE_KINEMATIC_COUNT = 4
CABLE_MU = 2.0
LATCH_LIMIT_LOWER = -0.2
LATCH_LIMIT_UPPER = 0.3
LATCH_SPRING_KE = 0.15
LATCH_SPRING_KD = 0.2
LATCH_LIMIT_KD = 1.0e-4


def smoothstep(value: float) -> float:
    """Ramp ``value`` from zero to one with zero slope at both ends."""
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def insertion_command(
    insertion_distance: float,
    frame: int,
    frames: int,
    previous: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target displacement and frame-to-frame command."""
    phase = frame / max(frames - 1, 1)
    delta = np.asarray(
        (0.0, insertion_distance * smoothstep(min(phase * 1.25, 1.0)), 0.0),
        dtype=np.float32,
    )
    return delta, np.concatenate((delta, delta - previous))


@dataclass(frozen=True)
class RJ45WorldLayout:
    """Indices and anchor data local to one RJ45 blueprint."""

    plug_body: int
    latch_body: int
    socket_position: tuple[float, float, float]
    rest_position: tuple[float, float, float]
    anchor_bodies: tuple[int, ...]
    anchor_offsets: tuple[tuple[float, float, float], ...]
    anchor_rotations: tuple[tuple[float, float, float, float], ...]
    align_bodies: tuple[int, ...]
    align_next: tuple[int, ...]


@wp.kernel
def _apply_targets(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    targets: wp.array[wp.vec3],
    plug_bodies: wp.array[int],
    latch_bodies: wp.array[int],
    gravity: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
):
    world = wp.tid()
    plug = plug_bodies[world]
    latch = latch_bodies[world]
    world_gravity = gravity[world]

    anti_plug = -world_gravity * body_mass[plug]
    anti_latch = -world_gravity * body_mass[latch]
    wp.atomic_add(body_f, plug, wp.spatial_vector(anti_plug, wp.vec3(0.0)))
    wp.atomic_add(body_f, latch, wp.spatial_vector(anti_latch, wp.vec3(0.0)))

    target = targets[world]
    plug_position = wp.transform_get_translation(body_q[plug])
    plug_velocity = wp.spatial_top(body_qd[plug])
    plug_mass = body_mass[plug]
    plug_multiplier = 10.0 + plug_mass
    plug_force = plug_multiplier * (
        stiffness * (target - plug_position) - damping * plug_velocity
    )
    wp.atomic_add(body_f, plug, wp.spatial_vector(plug_force, wp.vec3(0.0)))

    latch_velocity = wp.spatial_top(body_qd[latch])
    latch_mass = body_mass[latch]
    spring_acceleration = (target - plug_position) * (
        plug_multiplier * stiffness / plug_mass
    )
    latch_force = spring_acceleration * latch_mass - latch_velocity * (
        (10.0 + latch_mass) * damping
    )
    wp.atomic_add(body_f, latch, wp.spatial_vector(latch_force, wp.vec3(0.0)))


@wp.kernel
def _sync_cable_anchors(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    plug_bodies: wp.array[int],
    anchor_bodies: wp.array[int],
    anchor_offsets: wp.array[wp.vec3],
    anchor_rotations: wp.array[wp.quat],
    anchors_per_world: int,
):
    index = wp.tid()
    world = index // anchors_per_world
    plug_transform = body_q[plug_bodies[world]]
    plug_position = wp.transform_get_translation(plug_transform)
    plug_rotation = wp.transform_get_rotation(plug_transform)
    anchor_position = plug_position + wp.quat_rotate(
        plug_rotation, anchor_offsets[index]
    )
    anchor_rotation = wp.normalize(wp.mul(plug_rotation, anchor_rotations[index]))
    body = anchor_bodies[index]
    body_q[body] = wp.transform(anchor_position, anchor_rotation)
    body_qd[body] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _align_cable_orientations(
    body_q: wp.array[wp.transform],
    cable_bodies: wp.array[int],
    cable_next: wp.array[int],
):
    index = wp.tid()
    body = cable_bodies[index]
    next_body = cable_next[index]
    transform = body_q[body]
    position = wp.transform_get_translation(transform)
    rotation = wp.transform_get_rotation(transform)
    segment = wp.transform_get_translation(body_q[next_body]) - position
    segment_length = wp.length(segment)
    if segment_length < 1.0e-10:
        return
    direction = segment / segment_length
    current_direction = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    swing = quat_between_vectors_robust(current_direction, direction)
    body_q[body] = wp.transform(position, wp.normalize(wp.mul(swing, rotation)))


def _load_mesh(
    stage: Any, usd_module: Any, prim_path: str
) -> tuple[newton.Mesh, wp.vec3]:
    prim = stage.GetPrimAtPath(prim_path)
    usd_mesh = usd_module.get_mesh(prim, load_normals=True)
    position = wp.transform_get_translation(usd_module.get_transform(prim, local=False))
    normals = (
        np.asarray(usd_mesh.normals, dtype=np.float32)
        if usd_mesh.normals is not None
        else None
    )
    mesh = newton.Mesh(
        np.asarray(usd_mesh.vertices, dtype=np.float32),
        np.asarray(usd_mesh.indices, dtype=np.int32),
        normals=normals,
    )
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=SHAPE_CFG.gap,
    )
    return mesh, position


def _load_cable_centerline(stage: Any, usd_module: Any) -> tuple[wp.vec3, ...]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath("/World/CableCurve")
    points = UsdGeom.BasisCurves(prim).GetPointsAttr().Get()
    position = wp.transform_get_translation(usd_module.get_transform(prim, local=False))
    return tuple(
        wp.vec3(
            float(point[0]) + float(position[0]),
            float(point[1]) + float(position[1]) + PLUG_Y_OFFSET,
            float(point[2]) + float(position[2]),
        )
        for point in points
    )


def build_rj45_world() -> tuple[newton.ModelBuilder, RJ45WorldLayout]:
    """Build one world blueprint without finalizing a Newton model."""
    try:
        from pxr import Usd
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "The RJ45 example requires Pixar USD Python bindings. Install the "
            "PhysicsNeMo 'newton' extra."
        ) from error
    from newton import usd as newton_usd

    stage = Usd.Stage.Open(newton.examples.get_asset("rj45_plug.usd"))
    socket_mesh, socket_position = _load_mesh(stage, newton_usd, "/World/Socket")
    plug_mesh, plug_center = _load_mesh(stage, newton_usd, "/World/Plug")
    latch_mesh, latch_center = _load_mesh(stage, newton_usd, "/World/Latch")

    builder = newton.ModelBuilder(gravity=-9.81)
    builder.rigid_gap = 0.005
    builder.add_ground_plane()
    socket_shape = builder.add_shape_mesh(
        -1,
        mesh=socket_mesh,
        xform=wp.transform(socket_position, wp.quat_identity()),
        cfg=SHAPE_CFG,
        label="socket",
    )

    plug_position = wp.vec3(
        plug_center[0],
        plug_center[1] + PLUG_Y_OFFSET,
        plug_center[2],
    )
    plug_body = builder.add_link(
        xform=wp.transform(plug_position, wp.quat_identity()),
        label="plug",
    )
    plug_shape = builder.add_shape_mesh(plug_body, mesh=plug_mesh, cfg=SHAPE_CFG)

    latch_position = wp.vec3(
        latch_center[0],
        latch_center[1] + PLUG_Y_OFFSET,
        latch_center[2],
    )
    latch_body = builder.add_link(
        xform=wp.transform(latch_position, wp.quat_identity()),
        label="latch",
    )
    latch_shape = builder.add_shape_mesh(
        latch_body,
        mesh=latch_mesh,
        cfg=SHAPE_CFG,
    )
    connector_shapes = (socket_shape, plug_shape, latch_shape)

    joint_dof = newton.ModelBuilder.JointDofConfig
    plug_joint = builder.add_joint_d6(
        parent=-1,
        child=plug_body,
        linear_axes=(
            joint_dof(axis=(1.0, 0.0, 0.0)),
            joint_dof(axis=(0.0, 1.0, 0.0)),
            joint_dof(axis=(0.0, 0.0, 1.0)),
        ),
        angular_axes=None,
        parent_xform=wp.transform(plug_position, wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    latch_joint = builder.add_joint_revolute(
        parent=plug_body,
        child=latch_body,
        axis=(-1.0, 0.0, 0.0),
        parent_xform=wp.transform(latch_center - plug_center, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        target_ke=LATCH_SPRING_KE,
        target_kd=LATCH_SPRING_KD,
        limit_lower=LATCH_LIMIT_LOWER,
        limit_upper=LATCH_LIMIT_UPPER,
        limit_kd=LATCH_LIMIT_KD,
        collision_filter_parent=True,
    )
    builder.add_articulation([plug_joint, latch_joint])

    cable_points = _load_cable_centerline(stage, newton_usd)
    cable_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
        cable_points
    )
    rod_bodies, _ = builder.add_rod(
        positions=cable_points,
        quaternions=cable_quaternions,
        radius=CABLE_RADIUS,
        cfg=dataclasses.replace(
            builder.default_shape_cfg,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=CABLE_MU,
        ),
        bend_stiffness=1.0e1,
        bend_damping=1.0e-1,
        label="cable",
    )
    for body in rod_bodies[:CABLE_KINEMATIC_COUNT]:
        for cable_shape in builder.body_shapes[body]:
            for connector_shape in connector_shapes:
                builder.add_shape_collision_filter_pair(cable_shape, connector_shape)
    for body in (*rod_bodies[:CABLE_KINEMATIC_COUNT], rod_bodies[-1]):
        builder.body_mass[body] = 0.0
        builder.body_inv_mass[body] = 0.0
        builder.body_inertia[body] = wp.mat33(0.0)
        builder.body_inv_inertia[body] = wp.mat33(0.0)

    align_start = max(CABLE_KINEMATIC_COUNT - 1, 0)
    return builder, RJ45WorldLayout(
        plug_body=plug_body,
        latch_body=latch_body,
        socket_position=tuple(float(value) for value in socket_position),
        rest_position=tuple(float(value) for value in plug_position),
        anchor_bodies=tuple(rod_bodies[:CABLE_KINEMATIC_COUNT]),
        anchor_offsets=tuple(
            tuple(float(value) for value in cable_points[index] - plug_position)
            for index in range(CABLE_KINEMATIC_COUNT)
        ),
        anchor_rotations=tuple(
            tuple(float(value) for value in cable_quaternions[index])
            for index in range(CABLE_KINEMATIC_COUNT)
        ),
        align_bodies=tuple(rod_bodies[align_start:-1]),
        align_next=tuple(rod_bodies[align_start + 1 :]),
    )


class BatchedRJ45Scene:
    """Replicated, headless RJ45 VBD runtime.

    Parameters
    ----------
    world_count : int
        Number of independent RJ45 worlds advanced together.
    device : str
        Warp/Torch device used by the finalized Newton model.
    """

    def __init__(self, world_count: int, device: str) -> None:
        if world_count <= 0:
            raise ValueError("world_count must be positive")
        with wp.ScopedDevice(device):
            blueprint, layout = build_rj45_world()
            builder = newton.ModelBuilder(gravity=-9.81)
            builder.rigid_gap = blueprint.rigid_gap
            builder.replicate(blueprint, world_count)
            builder.color()
            self.model = builder.finalize()
        self.world_count = world_count
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.pick_stiffness = 50.0
        self.pick_damping = 10.0

        body_world = torch.as_tensor(self.model.body_world.numpy())
        body_groups = tuple(
            torch.nonzero(body_world == world, as_tuple=False).flatten()
            for world in range(world_count)
        )
        if len({int(group.numel()) for group in body_groups}) != 1:
            raise ValueError("replicated RJ45 worlds have inconsistent body counts")
        world_bodies = torch.stack(body_groups)
        self.plug_body_index = layout.plug_body
        self.latch_body_index = layout.latch_body
        self.socket_position = layout.socket_position
        self.plug_bodies = world_bodies[:, layout.plug_body]
        self.latch_bodies = world_bodies[:, layout.latch_body]
        self.rest_positions = torch.tensor(
            layout.rest_position,
            dtype=torch.float32,
            device=device,
        ).expand(world_count, 3)
        self._rest_targets = wp.array(
            (layout.rest_position,) * world_count,
            dtype=wp.vec3,
            device=self.model.device,
        )

        def replicated_body_indices(local_indices: tuple[int, ...]) -> np.ndarray:
            return (
                world_bodies[:, torch.tensor(local_indices)].reshape(-1).cpu().numpy()
            )

        self._plug_bodies = wp.array(
            self.plug_bodies.cpu().numpy(),
            dtype=int,
            device=self.model.device,
        )
        self._latch_bodies = wp.array(
            self.latch_bodies.cpu().numpy(),
            dtype=int,
            device=self.model.device,
        )
        self._targets = wp.clone(self._rest_targets)
        self._target_tensor = wp.to_torch(self._targets)
        self._anchor_bodies = wp.array(
            replicated_body_indices(layout.anchor_bodies),
            dtype=int,
            device=self.model.device,
        )
        self._anchor_offsets = wp.array(
            layout.anchor_offsets * world_count,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self._anchor_rotations = wp.array(
            layout.anchor_rotations * world_count,
            dtype=wp.quat,
            device=self.model.device,
        )
        self._anchors_per_world = len(layout.anchor_bodies)
        self._align_bodies = wp.array(
            replicated_body_indices(layout.align_bodies),
            dtype=int,
            device=self.model.device,
        )
        self._align_next = wp.array(
            replicated_body_indices(layout.align_next),
            dtype=int,
            device=self.model.device,
        )
        self._align_count = len(layout.align_bodies) * world_count

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._initial_body_q = wp.clone(self.state_0.body_q)
        self._initial_body_qd = wp.clone(self.state_0.body_qd)
        self.solver = self._new_solver()
        self.graph = None
        self.capture()
        wp.synchronize()
        self._restore_state()

    def _new_solver(self) -> SolverVBD:
        """Construct a VBD solver with the stock RJ45 constraint settings."""
        solver = SolverVBD(
            self.model,
            iterations=12,
            rigid_contact_hard=False,
            rigid_body_contact_buffer_size=256,
        )
        for joint in range(self.model.joint_count):
            solver.set_joint_constraint_mode(joint, False)
        return solver

    def _restore_state(self) -> None:
        """Restore state buffers after graph capture or a completed episode."""
        with torch_warp_stream(self.model.device):
            wp.copy(self.state_0.body_q, self._initial_body_q)
            wp.copy(self.state_0.body_qd, self._initial_body_qd)
            wp.copy(self.state_1.body_q, self._initial_body_q)
            wp.copy(self.state_1.body_qd, self._initial_body_qd)
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.contacts.clear()

    def set_targets(self, targets: torch.Tensor) -> None:
        """Set one world-space plug target per replicated world."""
        if targets.shape != (self.world_count, 3):
            raise ValueError(
                f"targets must have shape ({self.world_count}, 3), "
                f"got {tuple(targets.shape)}"
            )
        with torch_warp_stream(self.model.device), torch.no_grad():
            self._target_tensor.copy_(
                targets.to(
                    device=self._target_tensor.device,
                    dtype=self._target_tensor.dtype,
                )
            )

    def reset(self) -> None:
        """Start independent episodes with fresh state and VBD solver history."""
        wp.copy(self._targets, self._rest_targets)
        # SolverVBD intentionally carries penalty/constraint state and optional
        # warm-start history across timesteps. Independent trajectories must
        # rebuild it; resetting body state alone would leak solver history.
        self.graph = None
        self.solver = self._new_solver()
        self.capture()
        wp.synchronize()
        self._restore_state()

    def collide(self) -> None:
        """Refresh rigid contacts for the current batched state."""
        with torch_warp_stream(self.model.device):
            self.model.collide(self.state_0, self.contacts)

    def seat_error_mm(self, world: int = 0) -> float:
        """Return the selected world's plug seating error in millimeters."""
        if not 0 <= world < self.world_count:
            raise IndexError(f"world must be in [0, {self.world_count})")
        seated = self.rest_positions[world].detach().cpu().numpy().copy()
        seated[1] -= PLUG_Y_OFFSET
        plug = self.state_0.body_q.numpy()[int(self.plug_bodies[world]), :3]
        return float(np.linalg.norm(plug - seated) * 1000.0)

    def capture(self) -> None:
        """Capture one batched frame when graph capture is available."""
        if wp.get_device(self.model.device).is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        """Advance every replicated world by one frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                kernel=_apply_targets,
                dim=self.world_count,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.model.body_mass,
                    self._targets,
                    self._plug_bodies,
                    self._latch_bodies,
                    self.model.gravity,
                    self.pick_stiffness,
                    self.pick_damping,
                ),
                device=self.model.device,
            )
            wp.launch(
                kernel=_sync_cable_anchors,
                dim=self.world_count * self._anchors_per_world,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._plug_bodies,
                    self._anchor_bodies,
                    self._anchor_offsets,
                    self._anchor_rotations,
                    self._anchors_per_world,
                ),
                device=self.model.device,
            )
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(
                kernel=_align_cable_orientations,
                dim=self._align_count,
                inputs=(
                    self.state_0.body_q,
                    self._align_bodies,
                    self._align_next,
                ),
                device=self.model.device,
            )

    def step(self) -> None:
        """Advance one captured or eager batched frame."""
        with torch_warp_stream(self.model.device):
            if self.graph is None:
                self.simulate()
            else:
                wp.capture_launch(self.graph)
