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

"""Newton scene for offline neural co-design of a tendon-driven gripper.

The gripper has two identical opposing five-segment fingers. Revolute joints act as
compliant flexures, and a shared fixed tendon command applies a routed torque
distribution. The learned design controls geometry only: segment dimensions,
pre-curvature, palm size, and contact-pad shape. Candidate grasp poses vary
object offset, yaw, and grasp height. Grasp objects may be compound Newton
primitives or application-provided closed triangle meshes; the same geometry is
sampled for the PointNet input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import torch
import warp as wp

from physicsnemo.experimental.integrations.newton import (
    DesignSpace,
    DesignVariable,
    NewtonPrimitive,
    NewtonRigidObject,
    add_rigid_object_shapes,
    field_to_torch,
    rigid_object_fingerprint,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.procedural import lumpy_sphere
from physicsnemo.mesh.primitives.surfaces import octahedron_surface

SEGMENTS_PER_FINGER = 5
FINGER_COUNT = 2
SCENE_VERSION = 10
LENGTH_SLICE = slice(0, SEGMENTS_PER_FINGER)
RADIUS_SLICE = slice(LENGTH_SLICE.stop, LENGTH_SLICE.stop + SEGMENTS_PER_FINGER)
REST_ANGLE_SLICE = slice(RADIUS_SLICE.stop, RADIUS_SLICE.stop + SEGMENTS_PER_FINGER)
FINGER_SCALE_INDEX = REST_ANGLE_SLICE.stop
ROOT_HALF_WIDTH_INDEX = FINGER_SCALE_INDEX + 1
PAD_PROTRUSION_INDEX = ROOT_HALF_WIDTH_INDEX + 1
PAD_HALF_WIDTH_INDEX = PAD_PROTRUSION_INDEX + 1
GRIPPER_DESIGN_SPACE = DesignSpace(
    (
        *(
            DesignVariable(
                f"segment_length_{index}",
                0.042,
                0.062,
                tags=("geometry", "finger", "length_profile"),
            )
            for index in range(SEGMENTS_PER_FINGER)
        ),
        *(
            DesignVariable(
                f"segment_radius_{index}",
                0.0075,
                0.0155,
                tags=("geometry", "finger", "radius_profile"),
            )
            for index in range(SEGMENTS_PER_FINGER)
        ),
        *(
            DesignVariable(
                f"segment_rest_angle_{index}",
                -0.050,
                0.110,
                tags=("geometry", "finger", "curvature_profile"),
            )
            for index in range(SEGMENTS_PER_FINGER)
        ),
        DesignVariable(
            "finger_length_scale",
            0.88,
            1.12,
            tags=("geometry", "finger", "shared"),
        ),
        DesignVariable("root_radius", 0.052, 0.085, tags=("geometry", "palm")),
        DesignVariable("pad_protrusion", 0.004, 0.018, tags=("geometry", "contact")),
        DesignVariable("pad_half_width", 0.014, 0.040, tags=("geometry", "contact")),
    )
)
DESIGN_NAMES = GRIPPER_DESIGN_SPACE.names
DESIGN_DIM = GRIPPER_DESIGN_SPACE.dimension
FIXED_FLEXURE_STIFFNESS = np.asarray(
    (0.48, 0.52, 0.58, 0.66, 0.76),
    dtype=np.float32,
)
FIXED_JAW_TRAVEL = 0.040
FIXED_TENDON_TORQUE = 0.180
FIXED_ROUTING_EXPONENT = 1.15
PALM_HALF_HEIGHT = 0.006
FINGERTIP_CLEARANCE = 0.006
NOMINAL_SEGMENT_LENGTH = 0.052
NOMINAL_PALM_CENTER_Z = (
    PALM_HALF_HEIGHT
    + SEGMENTS_PER_FINGER * NOMINAL_SEGMENT_LENGTH
    + FINGERTIP_CLEARANCE
)
POINT_CLOUD_ENVELOPE_HALF_WIDTH = 0.090
POINT_CLOUD_ENVELOPE_Z = (-0.300, -0.100)

# Single source of truth for the continuous grasp loss and its companions.
# The torch counterpart ``gripper_model.outcome_loss_torch`` and the robust
# aggregation in ``gripper_model._robust_grouped_objective`` are kept separate
# for differentiability and must stay in sync with these values.
LOSS_LIFT_COEFFICIENT = 1.8
LOSS_SLIP_COEFFICIENT = 0.65
LOSS_ROTATION_COEFFICIENT = 0.30
LOSS_SLIP_SCALE = 0.08
LOSS_ROTATION_SCALE = 1.25
LOSS_SUCCESS_PENALTY = 0.80
SUCCESS_LIFT_THRESHOLD = 0.72
SUCCESS_SLIP_THRESHOLD = 0.075
ROBUST_MEAN_WEIGHT = 0.65
ROBUST_MAX_WEIGHT = 0.35


def _lofted_surface(
    heights: tuple[float, ...],
    radii_x: tuple[float, ...],
    radii_y: tuple[float, ...] | None = None,
    *,
    centers_x: tuple[float, ...] | None = None,
    lobes: int = 0,
    lobe_amplitude: float = 0.0,
    twist: float = 0.0,
    resolution: int = 36,
) -> Mesh:
    """Build a closed irregular object from elliptical cross sections."""
    z = np.asarray(heights, dtype=np.float32)
    rx = np.asarray(radii_x, dtype=np.float32)
    ry = rx if radii_y is None else np.asarray(radii_y, dtype=np.float32)
    cx = (
        np.zeros_like(z)
        if centers_x is None
        else np.asarray(centers_x, dtype=np.float32)
    )
    if not (len(z) == len(rx) == len(ry) == len(cx)) or len(z) < 2:
        raise ValueError("loft profiles must contain matching rows")
    if resolution < 8 or np.any(rx <= 0.0) or np.any(ry <= 0.0):
        raise ValueError("loft radii must be positive and resolution at least 8")

    theta = 2.0 * np.pi * np.arange(resolution, dtype=np.float32) / resolution
    progress = np.linspace(0.0, 1.0, len(z), dtype=np.float32)
    rings = []
    for row, fraction in enumerate(progress):
        phase = theta + twist * fraction
        modulation = (
            1.0 + lobe_amplitude * np.cos(lobes * phase)
            if lobes > 0
            else np.ones_like(theta)
        )
        rings.append(
            np.stack(
                (
                    cx[row] + rx[row] * modulation * np.cos(phase),
                    ry[row] * modulation * np.sin(phase),
                    np.full_like(theta, z[row]),
                ),
                axis=-1,
            )
        )
    points = np.concatenate(
        (
            np.concatenate(rings, axis=0),
            np.asarray(((cx[0], 0.0, z[0]), (cx[-1], 0.0, z[-1]))),
        ),
        axis=0,
    )
    cells: list[tuple[int, int, int]] = []
    for row in range(len(z) - 1):
        lower = row * resolution
        upper = (row + 1) * resolution
        for column in range(resolution):
            next_column = (column + 1) % resolution
            a = lower + column
            b = lower + next_column
            c = upper + column
            d = upper + next_column
            cells.extend(((a, b, c), (b, d, c)))
    bottom = len(z) * resolution
    top = bottom + 1
    last_ring = (len(z) - 1) * resolution
    for column in range(resolution):
        next_column = (column + 1) % resolution
        cells.append((bottom, next_column, column))
        cells.append((top, last_ring + column, last_ring + next_column))
    return Mesh(
        points=torch.as_tensor(points, dtype=torch.float32),
        cells=torch.as_tensor(cells, dtype=torch.int64),
    )


def _anisotropic_lumpy_mesh(
    *,
    radius: float,
    scale: tuple[float, float, float],
    noise: float,
    seed: int,
) -> Mesh:
    """Create a PhysicsNeMo procedural mesh with a non-spherical silhouette."""
    mesh = lumpy_sphere.load(
        radius=radius,
        subdivisions=2,
        noise_amplitude=noise,
        seed=seed,
    )
    return mesh.scale(torch.as_tensor(scale), assume_invertible=True)


def default_object_library() -> tuple[NewtonRigidObject, ...]:
    """Return household-like geometry plus unseen held-out shape families."""
    bottle_mesh = _lofted_surface(
        (-0.060, -0.052, -0.020, 0.025, 0.043, 0.052, 0.061),
        (0.029, 0.036, 0.039, 0.038, 0.032, 0.020, 0.018),
        (0.025, 0.031, 0.034, 0.033, 0.029, 0.018, 0.016),
        centers_x=(0.0, 0.0, 0.002, 0.004, 0.005, 0.005, 0.005),
        lobes=4,
        lobe_amplitude=0.045,
    )
    pear_mesh = _lofted_surface(
        (-0.056, -0.049, -0.028, -0.004, 0.020, 0.041, 0.057),
        (0.024, 0.039, 0.050, 0.052, 0.043, 0.026, 0.012),
        (0.022, 0.036, 0.046, 0.048, 0.040, 0.024, 0.011),
        centers_x=(-0.004, -0.003, -0.001, 0.002, 0.006, 0.010, 0.012),
        lobes=3,
        lobe_amplitude=0.035,
        twist=0.25,
    )
    ribbed_vessel_mesh = _lofted_surface(
        (-0.057, -0.048, -0.026, 0.000, 0.025, 0.046, 0.058),
        (0.034, 0.046, 0.051, 0.045, 0.052, 0.035, 0.030),
        (0.030, 0.040, 0.045, 0.040, 0.046, 0.031, 0.027),
        lobes=7,
        lobe_amplitude=0.085,
        twist=0.55,
    )
    star_mesh = _lofted_surface(
        (-0.052, -0.044, -0.016, 0.016, 0.044, 0.052),
        (0.030, 0.046, 0.050, 0.047, 0.038, 0.027),
        (0.027, 0.041, 0.045, 0.043, 0.034, 0.024),
        lobes=5,
        lobe_amplitude=0.16,
        twist=1.15,
    )
    organic_mesh = _anisotropic_lumpy_mesh(
        radius=1.0,
        scale=(0.052, 0.043, 0.058),
        noise=0.16,
        seed=41,
    )
    unseen_organic_mesh = _anisotropic_lumpy_mesh(
        radius=1.0,
        scale=(0.044, 0.054, 0.061),
        noise=0.21,
        seed=73,
    )
    gem_mesh = octahedron_surface.load(size=0.052)
    asymmetric_bottle_mesh = _lofted_surface(
        (-0.061, -0.053, -0.030, 0.002, 0.030, 0.048, 0.060),
        (0.027, 0.042, 0.047, 0.043, 0.036, 0.025, 0.016),
        (0.024, 0.037, 0.041, 0.038, 0.032, 0.022, 0.014),
        centers_x=(-0.007, -0.006, -0.002, 0.004, 0.010, 0.013, 0.015),
        lobes=4,
        lobe_amplitude=0.07,
        twist=-0.45,
    )
    quarter_turn = float(np.sqrt(0.5))
    return (
        NewtonRigidObject.from_mesh(
            name="faceted_bottle",
            mesh=bottle_mesh,
            density=320.0,
            color=(0.92, 0.43, 0.12),
            tags=("train",),
        ),
        NewtonRigidObject.from_mesh(
            name="asymmetric_pear",
            mesh=pear_mesh,
            density=360.0,
            color=(0.76, 0.58, 0.10),
            tags=("train",),
        ),
        NewtonRigidObject.from_mesh(
            name="ribbed_vessel",
            mesh=ribbed_vessel_mesh,
            density=330.0,
            color=(0.94, 0.33, 0.16),
            tags=("train",),
        ),
        NewtonRigidObject(
            name="handled_cup",
            density=300.0,
            color=(0.18, 0.58, 0.78),
            parts=(
                NewtonPrimitive("cylinder", (0.037, 0.050)),
                NewtonPrimitive(
                    "capsule",
                    (0.007, 0.024),
                    position=(0.055, 0.0, 0.026),
                    quaternion=(0.0, quarter_turn, 0.0, quarter_turn),
                ),
                NewtonPrimitive(
                    "capsule",
                    (0.007, 0.024),
                    position=(0.055, 0.0, -0.026),
                    quaternion=(0.0, quarter_turn, 0.0, quarter_turn),
                ),
                NewtonPrimitive(
                    "capsule",
                    (0.007, 0.024),
                    position=(0.079, 0.0, 0.0),
                ),
            ),
            tags=("train",),
        ),
        NewtonRigidObject(
            name="offset_power_tool",
            density=390.0,
            color=(0.95, 0.57, 0.08),
            parts=(
                NewtonPrimitive(
                    "box", (0.043, 0.028, 0.032), position=(0.0, 0.0, 0.026)
                ),
                NewtonPrimitive(
                    "box", (0.018, 0.023, 0.045), position=(0.020, 0.0, -0.035)
                ),
                NewtonPrimitive(
                    "cylinder",
                    (0.018, 0.030),
                    position=(-0.052, 0.0, 0.030),
                    quaternion=(0.0, quarter_turn, 0.0, quarter_turn),
                ),
            ),
            tags=("train",),
        ),
        NewtonRigidObject.from_mesh(
            name="organic_lobed_mesh",
            mesh=organic_mesh,
            density=380.0,
            color=(0.45, 0.31, 0.78),
            tags=("train",),
        ),
        NewtonRigidObject.from_mesh(
            name="twisted_star_mesh",
            mesh=star_mesh,
            density=350.0,
            color=(0.16, 0.62, 0.50),
            tags=("train",),
        ),
        NewtonRigidObject(
            name="sphere",
            density=410.0,
            color=(0.82, 0.24, 0.12),
            parts=(NewtonPrimitive("sphere", (0.046,)),),
            tags=("train",),
        ),
        NewtonRigidObject.from_mesh(
            name="faceted_gem",
            mesh=gem_mesh,
            density=420.0,
            color=(0.64, 0.24, 0.70),
            tags=("validation",),
        ),
        NewtonRigidObject.from_mesh(
            name="unseen_organic_mesh",
            mesh=unseen_organic_mesh,
            density=390.0,
            color=(0.38, 0.31, 0.78),
            tags=("validation",),
        ),
        NewtonRigidObject(
            name="forked_household_tool",
            density=370.0,
            color=(0.16, 0.55, 0.78),
            parts=(
                NewtonPrimitive("capsule", (0.014, 0.055)),
                NewtonPrimitive(
                    "box",
                    (0.050, 0.018, 0.012),
                    position=(0.0, 0.0, 0.054),
                ),
                NewtonPrimitive(
                    "capsule",
                    (0.009, 0.025),
                    position=(0.038, 0.0, 0.075),
                    quaternion=(0.0, quarter_turn, 0.0, quarter_turn),
                ),
                NewtonPrimitive(
                    "capsule",
                    (0.009, 0.025),
                    position=(-0.038, 0.0, 0.075),
                    quaternion=(0.0, quarter_turn, 0.0, quarter_turn),
                ),
            ),
            tags=("validation",),
        ),
        NewtonRigidObject(
            name="double_lobed_container",
            density=340.0,
            color=(0.24, 0.66, 0.38),
            parts=(
                NewtonPrimitive("sphere", (0.038,), position=(-0.026, 0.0, 0.0)),
                NewtonPrimitive("sphere", (0.031,), position=(0.034, 0.0, 0.020)),
                NewtonPrimitive("capsule", (0.018, 0.030), position=(0.0, 0.0, -0.030)),
            ),
            tags=("validation",),
        ),
        NewtonRigidObject.from_mesh(
            name="unseen_asymmetric_bottle",
            mesh=asymmetric_bottle_mesh,
            density=355.0,
            color=(0.86, 0.30, 0.24),
            tags=("test",),
        ),
    )


OBJECTS = default_object_library()
TRAIN_OBJECT_INDICES = tuple(
    index for index, grasp_object in enumerate(OBJECTS) if "train" in grasp_object.tags
)
VALIDATION_OBJECT_INDICES = tuple(
    index
    for index, grasp_object in enumerate(OBJECTS)
    if "validation" in grasp_object.tags
)
HOLDOUT_OBJECT_INDICES = tuple(
    index for index, grasp_object in enumerate(OBJECTS) if "test" in grasp_object.tags
)
DISPLAY_OBJECT_INDICES = (7, 3, 6, 12)
TRAIN_OBJECT_LIBRARY_FINGERPRINT = rigid_object_fingerprint(
    tuple(OBJECTS[index] for index in TRAIN_OBJECT_INDICES)
)
OBJECT_LIBRARY_FINGERPRINT = rigid_object_fingerprint(OBJECTS)


def object_split(grasp_object: NewtonRigidObject) -> str:
    """Return the example's train, validation, or held-out test role."""
    if "validation" in grasp_object.tags:
        return "validation"
    return "test" if "test" in grasp_object.tags else "train"


def object_initial_height(grasp_object: NewtonRigidObject) -> float:
    """Place the object's lowest geometry point just above the ground."""
    lower, _ = grasp_object.bounds
    return -float(lower[2]) + 0.001


@dataclass(frozen=True)
class GraspPose:
    """Object pose and closure command relative to the gripper."""

    offset_x: float
    offset_y: float
    yaw: float
    root_height: float
    closure_scale: float = 1.0

    def as_array(self) -> np.ndarray:
        """Return the pose fields in surrogate feature order."""
        return np.asarray(
            (
                self.offset_x,
                self.offset_y,
                self.yaw,
                self.root_height,
                self.closure_scale,
            ),
            dtype=np.float32,
        )


@dataclass(frozen=True)
class SimulationOptions:
    """Timing, solver, and test-load settings."""

    fps: float = 60.0
    substeps: int = 8
    sim_time: float = 2.0
    solver_iterations: int = 36
    close_start: float = 0.18
    close_end: float = 0.88
    lift_start: float = 1.02
    lift_end: float = 1.70
    lift_height: float = 0.20
    disturbance_start: float = 1.73
    disturbance_end: float = 1.84
    disturbance_force: float = 0.8
    world_spacing: float = 0.42

    @property
    def dt(self) -> float:
        """Return the solver substep duration in seconds."""
        return 1.0 / (self.fps * self.substeps)

    @property
    def frame_count(self) -> int:
        """Return the number of displayed simulation frames."""
        return int(round(self.sim_time * self.fps))


@dataclass(frozen=True)
class WorldRecord:
    """Indices and initial conditions for one isolated Newton world."""

    sample_index: int
    object_index: int
    pose_index: int
    moving_ids: tuple[int, ...]
    object_id: int
    segment_ids: tuple[int, ...]
    torque_dof_ids: tuple[int, ...]
    origin: np.ndarray
    moving_positions: np.ndarray
    moving_directions: np.ndarray
    jaw_travel: float
    object_position: np.ndarray
    object_rotation: np.ndarray


@dataclass
class GripperBatch:
    """Live batched Newton state plus tensorized actuation metadata."""

    model: Any
    solver: Any
    state_0: Any
    state_1: Any
    control: Any
    contacts: Any
    designs: np.ndarray
    object_indices: np.ndarray
    pose_indices: np.ndarray
    poses: tuple[tuple[GraspPose, ...], ...]
    records: list[WorldRecord]
    options: SimulationOptions
    moving_ids: torch.Tensor
    moving_initial_positions: torch.Tensor
    moving_directions: torch.Tensor
    jaw_travel: torch.Tensor
    object_ids: torch.Tensor
    torque_dof_ids: torch.Tensor
    torque_values: torch.Tensor


def unit_to_design(normalized: np.ndarray) -> np.ndarray:
    """Map normalized vectors to physical geometry values."""
    return np.asarray(
        GRIPPER_DESIGN_SPACE.decode(
            np.asarray(normalized, dtype=np.float32),
        ),
        dtype=np.float32,
    )


def design_to_unit(designs: np.ndarray) -> np.ndarray:
    """Map physical designs back to normalized coordinates."""
    return np.asarray(
        GRIPPER_DESIGN_SPACE.encode(np.asarray(designs, dtype=np.float32)),
        dtype=np.float32,
    )


def baseline_design() -> np.ndarray:
    """Return a straight, uniform two-finger geometry reference."""
    return np.asarray(
        [
            *([0.052] * SEGMENTS_PER_FINGER),
            0.0130,
            0.0120,
            0.0110,
            0.0100,
            0.0090,
            *([0.0] * SEGMENTS_PER_FINGER),
            1.0,
            0.070,
            0.008,
            0.024,
        ],
        dtype=np.float32,
    )


def generate_pose_candidates(
    count: int = 12,
    *,
    seed: int = 17,
    object_count: int = len(OBJECTS),
) -> tuple[tuple[GraspPose, ...], ...]:
    """Generate deterministic candidate grasp poses for every object."""
    if count <= 0:
        raise ValueError("count must be positive")
    templates = (
        GraspPose(0.0, 0.0, 0.0, 0.008, 0.72),
        GraspPose(0.0, 0.0, 0.0, 0.008, 0.88),
        GraspPose(0.0, 0.0, 0.0, 0.008, 1.05),
        GraspPose(0.0, 0.0, 0.0, 0.008, 1.22),
        GraspPose(0.0, 0.0, 0.0, 0.000, 0.95),
        GraspPose(0.0, 0.0, 0.0, 0.022, 0.95),
        GraspPose(0.0, 0.0, 0.0, 0.034, 0.95),
        GraspPose(0.008, 0.0, 0.0, 0.008, 0.95),
        GraspPose(-0.008, 0.0, 0.0, 0.008, 0.95),
        GraspPose(0.0, 0.0, 0.25 * np.pi, 0.008, 0.95),
        GraspPose(0.0, 0.0, 0.50 * np.pi, 0.008, 0.95),
        GraspPose(0.0, 0.0, -0.50 * np.pi, 0.008, 0.95),
    )
    all_poses = []
    for object_index in range(object_count):
        poses = list(templates[:count])
        extra = count - len(poses)
        if extra > 0:
            samples = (
                torch.quasirandom.SobolEngine(
                    5, scramble=True, seed=seed + object_index
                )
                .draw(extra)
                .numpy()
            )
            poses.extend(
                GraspPose(
                    offset_x=float(-0.010 + 0.020 * sample[0]),
                    offset_y=float(-0.012 + 0.024 * sample[1]),
                    yaw=float(-1.55 + 3.10 * sample[2]),
                    root_height=float(0.002 + 0.034 * sample[3]),
                    closure_scale=float(0.70 + 0.55 * sample[4]),
                )
                for sample in samples
            )
        all_poses.append(tuple(poses))
    return tuple(all_poses)


def object_point_cloud(
    object_index: int,
    pose: GraspPose,
    *,
    objects: tuple[NewtonRigidObject, ...] = OBJECTS,
    num_points: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Sample the candidate-local object surface inside the gripper envelope."""
    grasp_object = objects[object_index]
    oversample_count = max(8 * num_points, 1024)
    points = grasp_object.sample_surface(
        oversample_count,
        seed=seed + 97 * object_index,
    )
    cosine, sine = np.cos(pose.yaw), np.sin(pose.yaw)
    rotation = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    points = points @ rotation.T
    points[:, 0] += pose.offset_x
    points[:, 1] += pose.offset_y
    points[:, 2] += (
        object_initial_height(grasp_object) - NOMINAL_PALM_CENTER_Z - pose.root_height
    )
    lower_z, upper_z = POINT_CLOUD_ENVELOPE_Z
    inside = (
        (np.abs(points[:, 0]) <= POINT_CLOUD_ENVELOPE_HALF_WIDTH)
        & (np.abs(points[:, 1]) <= POINT_CLOUD_ENVELOPE_HALF_WIDTH)
        & (points[:, 2] >= lower_z)
        & (points[:, 2] <= upper_z)
    )
    local_points = points[inside]
    if len(local_points) < max(32, num_points // 4):
        local_points = points
    rng = np.random.default_rng(seed + 7919 * object_index)
    selected = rng.choice(
        len(local_points),
        size=num_points,
        replace=len(local_points) < num_points,
    )
    return local_points[selected].astype(np.float32)


def point_cloud_table(
    poses: tuple[tuple[GraspPose, ...], ...],
    *,
    objects: tuple[NewtonRigidObject, ...] = OBJECTS,
    num_points: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Return PointNet inputs for any primitive/mesh object collection."""
    if len(poses) != len(objects):
        raise ValueError("poses and objects must contain the same number of entries")
    pose_count = len(poses[0])
    if any(len(group) != pose_count for group in poses):
        raise ValueError("every object must have the same number of pose candidates")
    return np.stack(
        [
            np.stack(
                [
                    object_point_cloud(
                        object_index,
                        pose,
                        objects=objects,
                        num_points=num_points,
                        seed=seed,
                    )
                    for pose in object_poses
                ]
            )
            for object_index, object_poses in enumerate(poses)
        ]
    ).astype(np.float32)


def surrogate_feature_table(
    poses: tuple[tuple[GraspPose, ...], ...],
    *,
    objects: tuple[NewtonRigidObject, ...] = OBJECTS,
    num_points: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized point sets and explicit global context features.

    Surface shape and orientation stay in the normalized point cloud. The
    accompanying context stores translation, physical scale, object mass, and
    an explicit candidate-pose encoding, so normalization does not erase
    information needed by the physics surrogate. This representation works
    unchanged for primitive, compound, and mesh-backed
    :class:`NewtonRigidObject` instances.
    """
    raw_points = point_cloud_table(
        poses,
        objects=objects,
        num_points=num_points,
        seed=seed,
    )
    point_tensor = torch.as_tensor(raw_points)
    centroids = point_tensor.mean(dim=-2)
    centered = point_tensor - centroids[..., None, :]
    radii = torch.linalg.vector_norm(centered, dim=-1).amax(dim=-1).clamp_min(1.0e-8)
    normalized = centered / radii[..., None, None]
    pose_context = np.asarray(
        [
            [
                (
                    pose.offset_x / 0.012,
                    pose.offset_y / 0.012,
                    np.sin(pose.yaw),
                    np.cos(pose.yaw),
                    pose.root_height / 0.034,
                    (pose.closure_scale - 0.70) / 0.55,
                )
                for pose in object_poses
            ]
            for object_poses in poses
        ],
        dtype=np.float32,
    )
    masses = object_mass_features(objects)[:, None, None]
    mass_context = np.broadcast_to(masses, (*pose_context.shape[:2], 1))
    context = np.concatenate(
        (
            centroids.numpy() / 0.16,
            np.log(radii.numpy().clip(1.0e-6) / 0.05)[..., None],
            mass_context,
            pose_context,
        ),
        axis=-1,
    )
    return normalized.numpy().astype(np.float32), context.astype(np.float32)


def object_mass_features(
    objects: tuple[NewtonRigidObject, ...] = OBJECTS,
) -> np.ndarray:
    """Return log-mass features with a fixed 100 g reference scale."""
    masses = np.asarray(
        [grasp_object.approximate_mass for grasp_object in objects],
        dtype=np.float32,
    )
    return np.log(masses.clip(1.0e-6) / 0.1).astype(np.float32)


def build_gripper_batch(
    designs: np.ndarray,
    object_indices: np.ndarray,
    pose_indices: np.ndarray,
    *,
    poses: tuple[tuple[GraspPose, ...], ...],
    objects: tuple[NewtonRigidObject, ...] = OBJECTS,
    options: SimulationOptions | None = None,
    device: str = "cuda",
    world_origins: np.ndarray | None = None,
) -> GripperBatch:
    """Build collision-isolated articulated worlds for a simulation batch."""
    designs = np.asarray(designs, dtype=np.float32)
    object_indices = np.asarray(object_indices, dtype=np.int64).reshape(-1)
    pose_indices = np.asarray(pose_indices, dtype=np.int64).reshape(-1)
    if designs.ndim != 2 or designs.shape[1] != DESIGN_DIM:
        raise ValueError(f"designs must have shape (batch, {DESIGN_DIM})")
    if len(designs) != len(object_indices) or len(designs) != len(pose_indices):
        raise ValueError("designs, object_indices, and pose_indices must align")
    if len(poses) != len(objects):
        raise ValueError("poses and objects must contain the same number of entries")
    opts = options or SimulationOptions()
    if world_origins is not None:
        world_origins = np.asarray(world_origins, dtype=np.float32)
        if world_origins.shape != (len(designs), 3):
            raise ValueError("world_origins must have shape (batch, 3)")

    scene = newton.ModelBuilder()
    scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.85, margin=0.0005))
    records: list[WorldRecord] = []
    torque_rows = []
    for sample_index, design in enumerate(designs):
        object_index = int(object_indices[sample_index])
        pose_index = int(pose_indices[sample_index])
        pose = poses[object_index][pose_index]
        grasp_object = objects[object_index]
        world, local = _world_builder(design, grasp_object, pose)
        body_offset = scene.body_count
        dof_offset = scene.joint_dof_count
        origin = (
            world_origins[sample_index]
            if world_origins is not None
            else np.zeros(3, dtype=np.float32)
        )
        scene.add_world(
            world,
            xform=wp.transform(wp.vec3(*origin.tolist()), wp.quat_identity()),
            label_prefix=f"sample_{sample_index}_{grasp_object.name}",
        )
        torque_rows.append(local["torque_values"])
        records.append(
            WorldRecord(
                sample_index=sample_index,
                object_index=object_index,
                pose_index=pose_index,
                moving_ids=tuple(body_offset + index for index in local["moving_ids"]),
                object_id=body_offset + local["object_id"],
                segment_ids=tuple(
                    body_offset + index for index in local["segment_ids"]
                ),
                torque_dof_ids=tuple(
                    dof_offset + index for index in local["torque_dof_ids"]
                ),
                origin=origin,
                moving_positions=origin + local["moving_positions"],
                moving_directions=local["moving_directions"],
                jaw_travel=float(local["jaw_travel"]),
                object_position=origin + local["object_position"],
                object_rotation=local["object_rotation"],
            )
        )

    model = scene.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    contacts = model.contacts()
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=opts.solver_iterations,
        rigid_contact_relaxation=0.72,
    )
    tensor_device = field_to_torch(state_0.body_q).device
    return GripperBatch(
        model=model,
        solver=solver,
        state_0=state_0,
        state_1=state_1,
        control=control,
        contacts=contacts,
        designs=designs,
        object_indices=object_indices,
        pose_indices=pose_indices,
        poses=poses,
        records=records,
        options=opts,
        moving_ids=torch.as_tensor(
            [record.moving_ids for record in records],
            dtype=torch.long,
            device=tensor_device,
        ),
        moving_initial_positions=torch.as_tensor(
            np.stack([record.moving_positions for record in records]),
            dtype=torch.float32,
            device=tensor_device,
        ),
        moving_directions=torch.as_tensor(
            np.stack([record.moving_directions for record in records]),
            dtype=torch.float32,
            device=tensor_device,
        ),
        jaw_travel=torch.as_tensor(
            [record.jaw_travel for record in records],
            dtype=torch.float32,
            device=tensor_device,
        ),
        object_ids=torch.as_tensor(
            [record.object_id for record in records],
            dtype=torch.long,
            device=tensor_device,
        ),
        torque_dof_ids=torch.as_tensor(
            [record.torque_dof_ids for record in records],
            dtype=torch.long,
            device=tensor_device,
        ),
        torque_values=torch.as_tensor(
            np.stack(torque_rows),
            dtype=torch.float32,
            device=tensor_device,
        ),
    )


def _world_builder(
    design: np.ndarray,
    grasp_object: NewtonRigidObject,
    pose: GraspPose,
) -> tuple[newton.ModelBuilder, dict[str, Any]]:
    segment_lengths = np.asarray(design[LENGTH_SLICE], dtype=np.float32)
    segment_radii = np.asarray(design[RADIUS_SLICE], dtype=np.float32)
    rest_angles = np.asarray(design[REST_ANGLE_SLICE], dtype=np.float32)
    finger_scale = float(design[FINGER_SCALE_INDEX])
    root_half_width = float(design[ROOT_HALF_WIDTH_INDEX])
    pad_protrusion = float(design[PAD_PROTRUSION_INDEX])
    pad_half_width = float(design[PAD_HALF_WIDTH_INDEX])
    profile_lengths = segment_lengths * finger_scale

    builder = newton.ModelBuilder()
    finger_cfg = newton.ModelBuilder.ShapeConfig(
        density=620.0,
        mu=1.05,
        margin=0.0005,
        collision_group=-2,
    )
    pad_cfg = newton.ModelBuilder.ShapeConfig(
        density=680.0,
        mu=1.65,
        mu_torsional=0.012,
        margin=0.0007,
        collision_group=-2,
    )
    palm_visual_cfg = newton.ModelBuilder.ShapeConfig(
        density=1.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    palm_position = np.asarray(
        (
            0.0,
            0.0,
            PALM_HALF_HEIGHT
            + float(profile_lengths.sum())
            + FINGERTIP_CLEARANCE
            + pose.root_height,
        ),
        dtype=np.float32,
    )
    active_angles = 2.0 * np.pi * np.arange(FINGER_COUNT) / FINGER_COUNT
    palm_id = builder.add_body(
        xform=wp.transform(wp.vec3(*palm_position.tolist()), wp.quat_identity()),
        mass=1.0,
        is_kinematic=True,
        label="palm",
    )
    builder.add_shape_sphere(
        palm_id,
        radius=0.020,
        cfg=palm_visual_cfg,
        color=(0.16, 0.20, 0.24),
        label="palm_hub",
    )
    for finger_index, angle in enumerate(active_angles):
        outward = np.asarray((np.cos(angle), np.sin(angle), 0.0), dtype=np.float32)
        yaw = wp.quat(0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle))
        builder.add_shape_box(
            palm_id,
            xform=wp.transform(
                wp.vec3(*(0.5 * root_half_width * outward).tolist()),
                yaw,
            ),
            hx=0.5 * root_half_width + 0.012,
            hy=0.013,
            hz=PALM_HALF_HEIGHT,
            cfg=palm_visual_cfg,
            color=(0.16, 0.20, 0.24),
            label=f"palm_spoke_{finger_index}",
        )

    moving_ids = [palm_id]
    moving_positions = [palm_position]
    moving_directions = [np.zeros(3, dtype=np.float32)]
    segment_ids: list[int] = []
    torque_dof_ids: list[int] = []
    torque_values: list[float] = []
    for finger_index, angle_value in enumerate(active_angles):
        angle = float(angle_value)
        outward = np.asarray((np.cos(angle), np.sin(angle), 0.0), dtype=np.float32)
        tangent = np.asarray((-np.sin(angle), np.cos(angle), 0.0), dtype=np.float32)
        base_position = palm_position + root_half_width * outward
        finger_base = builder.add_body(
            xform=wp.transform(wp.vec3(*base_position.tolist()), wp.quat_identity()),
            mass=0.25,
            is_kinematic=True,
            label=f"finger_carriage_{finger_index}",
        )
        yaw = wp.quat(0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle))
        builder.add_shape_box(
            finger_base,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), yaw),
            hx=0.017,
            hy=0.035,
            hz=PALM_HALF_HEIGHT,
            cfg=palm_visual_cfg,
            color=(0.24, 0.29, 0.34),
            label=f"finger_carriage_{finger_index}",
        )
        moving_ids.append(finger_base)
        moving_positions.append(base_position)
        moving_directions.append(outward)
        parent = finger_base
        articulation = []
        previous_length = 0.0
        for segment_index in range(SEGMENTS_PER_FINGER):
            segment_length = float(segment_lengths[segment_index] * finger_scale)
            radius = float(segment_radii[segment_index])
            link = builder.add_link(label=f"finger_{finger_index}_{segment_index}")
            builder.add_shape_capsule(
                link,
                radius=radius,
                half_height=max(0.001, 0.5 * segment_length - radius),
                cfg=(
                    pad_cfg if segment_index >= SEGMENTS_PER_FINGER - 2 else finger_cfg
                ),
                color=_curvature_color(float(rest_angles[segment_index])),
                label=f"phalange_{segment_index}",
            )
            if segment_index >= 1:
                builder.add_shape_box(
                    link,
                    xform=wp.transform(
                        wp.vec3(*(-pad_protrusion * outward).tolist()),
                        yaw,
                    ),
                    hx=0.72 * radius,
                    hy=pad_half_width,
                    hz=0.34 * segment_length,
                    cfg=pad_cfg,
                    color=(
                        (0.13, 0.32, 0.78)
                        if segment_index >= SEGMENTS_PER_FINGER - 2
                        else (0.16, 0.39, 0.68)
                    ),
                    label=f"contact_pad_{segment_index}",
                )
            joint = builder.add_joint_revolute(
                parent,
                link,
                parent_xform=wp.transform(
                    wp.vec3(
                        0.0,
                        0.0,
                        -PALM_HALF_HEIGHT
                        if segment_index == 0
                        else -0.5 * previous_length,
                    ),
                    wp.quat_identity(),
                ),
                child_xform=wp.transform(
                    wp.vec3(0.0, 0.0, 0.5 * segment_length),
                    wp.quat_identity(),
                ),
                axis=wp.vec3(*tangent.tolist()),
                target_pos=float(rest_angles[segment_index]),
                target_ke=float(FIXED_FLEXURE_STIFFNESS[segment_index]),
                target_kd=0.060
                + 0.080 * np.sqrt(FIXED_FLEXURE_STIFFNESS[segment_index]),
                limit_lower=-0.10,
                limit_upper=0.42,
                limit_ke=120.0,
                limit_kd=6.0,
                label=f"flexure_{finger_index}_{segment_index}",
            )
            route_fraction = segment_index / (SEGMENTS_PER_FINGER - 1)
            route_weight = (1.0 - 0.45 * route_fraction) ** FIXED_ROUTING_EXPONENT
            torque_dof_ids.append(builder.joint_qd_start[joint])
            torque_values.append(FIXED_TENDON_TORQUE * route_weight)
            segment_ids.append(link)
            articulation.append(joint)
            parent = link
            previous_length = segment_length
        builder.add_articulation(articulation)

    object_cfg = newton.ModelBuilder.ShapeConfig(
        density=grasp_object.density,
        mu=0.82,
        mu_torsional=0.007,
        margin=0.0005,
    )
    object_position = np.asarray(
        (pose.offset_x, pose.offset_y, object_initial_height(grasp_object)),
        dtype=np.float32,
    )
    object_rotation = np.asarray(
        (0.0, 0.0, np.sin(0.5 * pose.yaw), np.cos(0.5 * pose.yaw)),
        dtype=np.float32,
    )
    object_id = builder.add_body(
        xform=wp.transform(
            wp.vec3(*object_position.tolist()),
            wp.quat(*object_rotation.tolist()),
        ),
        label=grasp_object.name,
    )
    add_rigid_object_shapes(
        builder,
        object_id,
        grasp_object,
        object_cfg,
    )
    return builder, {
        "moving_ids": tuple(moving_ids),
        "object_id": object_id,
        "segment_ids": tuple(segment_ids),
        "torque_dof_ids": tuple(torque_dof_ids),
        "torque_values": np.asarray(torque_values, dtype=np.float32),
        "moving_positions": np.stack(moving_positions),
        "moving_directions": np.stack(moving_directions),
        "jaw_travel": FIXED_JAW_TRAVEL * pose.closure_scale,
        "object_position": object_position,
        "object_rotation": object_rotation,
    }


def _ramp(time: float, start: float, end: float) -> float:
    return float(np.clip((time - start) / max(end - start, 1.0e-8), 0.0, 1.0))


def apply_gripper_motion(batch: GripperBatch, state: Any, time: float) -> None:
    """Apply tendon tension and the prescribed palm lift."""
    close = _ramp(time, batch.options.close_start, batch.options.close_end)
    lift = (
        _ramp(time, batch.options.lift_start, batch.options.lift_end)
        * batch.options.lift_height
    )
    transforms = field_to_torch(state.body_q)
    velocities = field_to_torch(state.body_qd)
    old_positions = transforms[batch.moving_ids, :3].clone()
    new_positions = batch.moving_initial_positions.clone()
    new_positions -= close * batch.jaw_travel[:, None, None] * batch.moving_directions
    new_positions[:, :, 2] += lift
    transforms[batch.moving_ids, :3] = new_positions
    transforms[batch.moving_ids, 3:7] = transforms.new_tensor([0.0, 0.0, 0.0, 1.0])
    velocities[batch.moving_ids, :3] = (
        new_positions - old_positions
    ) / batch.options.dt
    velocities[batch.moving_ids, 3:6] = 0.0

    joint_forces = field_to_torch(batch.control.joint_f)
    joint_forces[batch.torque_dof_ids] = close * batch.torque_values


def step_gripper_batch(batch: GripperBatch, time: float) -> float:
    """Advance one Newton substep."""
    apply_gripper_motion(batch, batch.state_0, time)
    batch.state_0.clear_forces()
    if batch.options.disturbance_start <= time < batch.options.disturbance_end:
        body_forces = field_to_torch(batch.state_0.body_f)
        body_forces[batch.object_ids, 1] = batch.options.disturbance_force
    batch.model.collide(batch.state_0, batch.contacts)
    batch.solver.step(
        batch.state_0,
        batch.state_1,
        batch.control,
        batch.contacts,
        batch.options.dt,
    )
    batch.state_0, batch.state_1 = batch.state_1, batch.state_0
    return time + batch.options.dt


def run_gripper_batch(batch: GripperBatch) -> dict[str, np.ndarray]:
    """Run close, lift, and disturbance phases and return physical outcomes."""
    time = 0.0
    for _ in range(batch.options.frame_count):
        for _ in range(batch.options.substeps):
            time = step_gripper_batch(batch, time)
    return measure_batch(batch)


def measure_batch(batch: GripperBatch) -> dict[str, np.ndarray]:
    """Measure lift, lateral slip, rotation, and retained-grasp success."""
    transforms = field_to_torch(batch.state_0.body_q).detach().cpu().numpy()
    lift_fraction = np.empty(len(batch.records), dtype=np.float32)
    lateral_slip = np.empty_like(lift_fraction)
    rotation_error = np.empty_like(lift_fraction)
    success = np.empty_like(lift_fraction)
    final_positions = np.empty((len(batch.records), 3), dtype=np.float32)
    for record in batch.records:
        transform = transforms[record.object_id]
        index = record.sample_index
        if not np.isfinite(transform).all():
            lift_fraction[index] = 0.0
            lateral_slip[index] = 0.16
            rotation_error[index] = np.pi
            success[index] = 0.0
            final_positions[index] = record.object_position
            continue
        displacement = transform[:3] - record.object_position
        lift = np.clip(
            displacement[2] / batch.options.lift_height,
            0.0,
            1.15,
        )
        slip = float(np.linalg.norm(displacement[:2]))
        quaternion_dot = abs(float(np.dot(transform[3:7], record.object_rotation)))
        angle = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
        lift_fraction[index] = lift
        lateral_slip[index] = slip
        rotation_error[index] = angle
        success[index] = float(
            lift >= SUCCESS_LIFT_THRESHOLD and slip <= SUCCESS_SLIP_THRESHOLD
        )
        final_positions[index] = transform[:3]
    return {
        "lift_fraction": lift_fraction,
        "lateral_slip": lateral_slip,
        "rotation_error": rotation_error,
        "success": success,
        "final_position": final_positions,
    }


def outcome_loss(
    lift_fraction: np.ndarray,
    lateral_slip: np.ndarray,
    rotation_error: np.ndarray,
) -> np.ndarray:
    """Continuous grasp loss used for training diagnostics and validation."""
    lift = np.asarray(lift_fraction, dtype=np.float32)
    slip = np.asarray(lateral_slip, dtype=np.float32)
    rotation = np.asarray(rotation_error, dtype=np.float32)
    return (
        LOSS_LIFT_COEFFICIENT * np.square(1.0 - np.clip(lift, 0.0, 1.0))
        + LOSS_SLIP_COEFFICIENT * np.square(slip / LOSS_SLIP_SCALE)
        + LOSS_ROTATION_COEFFICIENT * np.square(rotation / LOSS_ROTATION_SCALE)
    ).astype(np.float32)


def retained_grasp_success(
    lift_fraction: np.ndarray,
    lateral_slip: np.ndarray,
) -> np.ndarray:
    """Return whether an object remains lifted and laterally retained."""
    lift = np.asarray(lift_fraction, dtype=np.float32)
    slip = np.asarray(lateral_slip, dtype=np.float32)
    return ((lift >= SUCCESS_LIFT_THRESHOLD) & (slip <= SUCCESS_SLIP_THRESHOLD)).astype(
        np.float32
    )


def evaluate_designs(
    designs: np.ndarray,
    object_indices: np.ndarray,
    pose_indices: np.ndarray,
    *,
    poses: tuple[tuple[GraspPose, ...], ...],
    objects: tuple[NewtonRigidObject, ...] = OBJECTS,
    options: SimulationOptions | None = None,
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Evaluate aligned design/object/pose rows in bounded Newton batches."""
    designs = np.asarray(designs, dtype=np.float32)
    object_indices = np.asarray(object_indices, dtype=np.int64)
    pose_indices = np.asarray(pose_indices, dtype=np.int64)
    collected: dict[str, list[np.ndarray]] = {
        "lift_fraction": [],
        "lateral_slip": [],
        "rotation_error": [],
        "success": [],
        "final_position": [],
    }
    for start in range(0, len(designs), batch_size):
        stop = min(len(designs), start + batch_size)
        batch = build_gripper_batch(
            designs[start:stop],
            object_indices[start:stop],
            pose_indices[start:stop],
            poses=poses,
            objects=objects,
            options=options,
            device=device,
        )
        metrics = run_gripper_batch(batch)
        for key in collected:
            collected[key].append(metrics[key])
    result = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    result["loss"] = outcome_loss(
        result["lift_fraction"],
        result["lateral_slip"],
        result["rotation_error"],
    ) + LOSS_SUCCESS_PENALTY * (1.0 - result["success"])
    return result


def _curvature_color(rest_angle: float) -> tuple[float, float, float]:
    unit = np.clip((rest_angle + 0.06) / 0.26, 0.0, 1.0)
    outward = np.asarray((0.12, 0.48, 0.92))
    inward = np.asarray((0.96, 0.34, 0.08))
    return tuple(((1.0 - unit) * outward + unit * inward).tolist())
