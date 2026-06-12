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

"""Newton MPM "nozzle" scene: geometry, build, and a flow-quality objective.

This is the physics half of the nozzle design example, ported from Newton's
``example_mpm_nozzle_design.py``. Six interpretable parameters describe a
monotone engineering nozzle (three radii, a metering height fraction, two
curvature exponents); the geometry is built as a surface-of-revolution mesh,
filled with a fixed lumpy "messy" inlet charge, and simulated as a batch of
parallel worlds with the implicit-MPM solver. The objective rewards a design
that turns the messy inlet into a steady, on-target downstream mass flow.

Nothing here is Newton-specific machinery a user would reuse across scenes -- it
is the scene -- so it lives next to the example rather than in the integration
library. It uses only Newton's public API (``ModelBuilder``, ``Mesh``,
``SolverImplicitMPM``) and changes nothing in Newton.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

DESIGN_NAMES = (
    "r_top",
    "r_meter",
    "r_aperture",
    "z_meter_frac",
    "lower_exponent",
    "upper_exponent",
)
DESIGN_DIM = len(DESIGN_NAMES)

# Fixed messy inlet charge -- the same lumpy upstream material for every design,
# so the optimizer must improve the geometry, not the feed.
NOISY_INPUT_RADIUS = 0.047
NOISY_INPUT_HEIGHT = 0.190
NOISY_INPUT_BOTTOM_GAP = 0.006
NOISY_INPUT_PARTICLES_PER_CELL = 2.20
NOISY_INPUT_PULSE_COUNT = 13
NOISY_INPUT_DROPLET_COUNT = 9

NOZZLE_WALL_THICKNESS = 0.018
NOZZLE_COLLISION_MARGIN = 0.0015
NOZZLE_PROFILE_POINTS = 40
NOZZLE_RADIAL_SEGMENTS = 64

# Single source of truth for the messy-inlet RNG seed, shared by the particle
# emitter and SimulationOptions so the value cannot drift between them.
DEFAULT_PARTICLE_SEED = 422


@dataclass
class DesignBounds:
    """Allowable ranges for the 6 engineering design parameters."""

    r_top_min: float = 0.052
    r_top_max: float = 0.075
    r_meter_min: float = 0.018
    r_meter_max: float = 0.056
    r_aperture_min: float = 0.015
    r_aperture_max: float = 0.026
    z_meter_min: float = 0.18
    z_meter_max: float = 0.58
    exp_min: float = 0.60
    exp_max: float = 1.90

    def as_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(lower, upper)`` design-bound vectors as float32 arrays."""
        lo = np.array(
            [
                self.r_top_min,
                self.r_meter_min,
                self.r_aperture_min,
                self.z_meter_min,
                self.exp_min,
                self.exp_min,
            ],
            dtype=np.float32,
        )
        hi = np.array(
            [
                self.r_top_max,
                self.r_meter_max,
                self.r_aperture_max,
                self.z_meter_max,
                self.exp_max,
                self.exp_max,
            ],
            dtype=np.float32,
        )
        return lo, hi


def unit_to_design(unit: np.ndarray, bounds: DesignBounds) -> np.ndarray:
    """Map a unit-cube point to a valid design (aperture <= meter <= inlet)."""
    lo, hi = bounds.as_array()
    u = np.clip(np.asarray(unit, dtype=np.float32), 0.0, 1.0)
    out = lo + (hi - lo) * u
    meter_lo = np.maximum(lo[1], out[..., 2] * 1.02)
    meter_hi = np.minimum(hi[1], out[..., 0] * 0.96)
    meter_hi = np.maximum(meter_hi, meter_lo + 1.0e-5)
    out[..., 1] = meter_lo + (meter_hi - meter_lo) * u[..., 1]
    return out.astype(np.float32)


def design_to_unit(design: np.ndarray, bounds: DesignBounds) -> np.ndarray:
    """Inverse of :func:`unit_to_design` for valid or near-valid designs."""
    lo, hi = bounds.as_array()
    d = np.asarray(design, dtype=np.float32)
    u = (d - lo) / (hi - lo)
    meter_lo = np.maximum(lo[1], d[..., 2] * 1.02)
    meter_hi = np.minimum(hi[1], d[..., 0] * 0.96)
    meter_hi = np.maximum(meter_hi, meter_lo + 1.0e-5)
    u[..., 1] = (d[..., 1] - meter_lo) / (meter_hi - meter_lo)
    return np.clip(u, 1.0e-4, 1.0 - 1.0e-4).astype(np.float32)


def repair_design(design: np.ndarray, bounds: DesignBounds | None = None) -> np.ndarray:
    """Project a design back into the valid engineering parameterization."""
    b = bounds or DesignBounds()
    return unit_to_design(design_to_unit(np.asarray(design, dtype=np.float32), b), b)


def design_to_profile(
    design: np.ndarray, num_points: int = NOZZLE_PROFILE_POINTS
) -> np.ndarray:
    """Convert a 6-DOF design to a monotone 1D radius profile r(z), bottom to top."""
    d = repair_design(np.asarray(design, dtype=np.float32)).astype(np.float64)
    r_top, r_meter, r_aperture, z_meter_frac, p_lo, p_hi = d.tolist()
    z_meter_frac = float(np.clip(z_meter_frac, 0.05, 0.95))
    p_lo, p_hi = max(0.1, float(p_lo)), max(0.1, float(p_hi))
    z_frac = np.linspace(0.0, 1.0, num_points)
    radii = np.empty_like(z_frac)
    lower = z_frac <= z_meter_frac
    radii[lower] = r_aperture + (r_meter - r_aperture) * np.power(
        z_frac[lower] / z_meter_frac, p_lo
    )
    s_hi = (z_frac[~lower] - z_meter_frac) / (1.0 - z_meter_frac)
    radii[~lower] = r_meter + (r_top - r_meter) * np.power(s_hi, p_hi)
    return np.maximum.accumulate(radii).astype(np.float32)


def build_nozzle_mesh(
    profile_radii: np.ndarray,
    height: float,
    z_offset: float,
    thickness: float = NOZZLE_WALL_THICKNESS,
    num_segments: int = NOZZLE_RADIAL_SEGMENTS,
):
    """Surface-of-revolution shell mesh from a 1D radius profile."""
    profile = np.asarray(profile_radii, dtype=np.float64)
    k_rings, n = profile.shape[0], num_segments
    if k_rings < 2:
        raise ValueError("profile must have at least 2 rings")
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    inner_rings, outer_rings = [], []
    for k in range(k_rings):
        r_in = float(profile[k])
        r_out = r_in + thickness
        z = z_offset + k * height / (k_rings - 1)
        inner_rings.append(np.column_stack([r_in * cos_t, r_in * sin_t, np.full(n, z)]))
        outer_rings.append(
            np.column_stack([r_out * cos_t, r_out * sin_t, np.full(n, z)])
        )
    inner, outer = np.array(inner_rings), np.array(outer_rings)
    vertices = np.concatenate(
        [inner.reshape(-1, 3), outer.reshape(-1, 3)], axis=0
    ).astype(np.float32)
    in_off, out_off = 0, k_rings * n
    indices: list[int] = []
    for k in range(k_rings - 1):  # side walls (inner faces in, outer faces out)
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = (
                in_off + k * n + i,
                in_off + (k + 1) * n + i,
                in_off + k * n + j,
                in_off + (k + 1) * n + j,
            )
            indices.extend([a, b, c, c, b, d])
            a, b, c, d = (
                out_off + k * n + i,
                out_off + (k + 1) * n + i,
                out_off + k * n + j,
                out_off + (k + 1) * n + j,
            )
            indices.extend([a, c, b, b, c, d])
    for i in range(n):  # top rim
        j = (i + 1) % n
        a, b = in_off + (k_rings - 1) * n + i, in_off + (k_rings - 1) * n + j
        c, d = out_off + (k_rings - 1) * n + i, out_off + (k_rings - 1) * n + j
        indices.extend([a, c, b, b, c, d])
    for i in range(n):  # bottom rim
        j = (i + 1) % n
        a, b, c, d = in_off + i, in_off + j, out_off + i, out_off + j
        indices.extend([a, b, c, c, b, d])
    return vertices, np.array(indices, dtype=np.int32)


def emit_messy_input_particles(
    nozzle_top_z: float,
    voxel_size: float,
    density: float,
    seed: int = DEFAULT_PARTICLE_SEED,
):
    """The fixed lumpy inlet charge (ellipsoid pulses + droplets) above the nozzle."""
    rng = np.random.default_rng(seed)
    z0 = nozzle_top_z + NOISY_INPUT_BOTTOM_GAP
    z1 = z0 + NOISY_INPUT_HEIGHT
    lo = np.array([-NOISY_INPUT_RADIUS, -NOISY_INPUT_RADIUS, z0])
    hi = np.array([NOISY_INPUT_RADIUS, NOISY_INPUT_RADIUS, z1])
    res = np.maximum(
        1,
        np.array(
            np.ceil(NOISY_INPUT_PARTICLES_PER_CELL * (hi - lo) / voxel_size), dtype=int
        ),
    )
    cell = (hi - lo) / res
    radius = float(np.max(cell) * 0.5)
    mass = float(np.prod(cell)) * density
    grid = [np.arange(res[i] + 1) * cell[i] for i in range(3)]
    pts = np.stack(np.meshgrid(*grid, indexing="ij")).reshape(3, -1).T
    pts = pts + (rng.random(pts.shape) - 0.5) * 0.62 * np.max(cell) + lo
    centers_z = np.linspace(
        z0 + 0.012, z1 - 0.012, NOISY_INPUT_PULSE_COUNT
    ) + rng.uniform(-0.0045, 0.0045, NOISY_INPUT_PULSE_COUNT)
    centers_xy = rng.uniform(-0.008, 0.008, size=(NOISY_INPUT_PULSE_COUNT, 2))
    blob_r = rng.uniform(
        [0.021, 0.021, 0.0065],
        [0.038, 0.038, 0.0125],
        size=(NOISY_INPUT_PULSE_COUNT, 3),
    )
    inside = np.zeros(pts.shape[0], dtype=bool)
    for i in range(NOISY_INPUT_PULSE_COUNT):
        d = (
            ((pts[:, 0] - centers_xy[i, 0]) / blob_r[i, 0]) ** 2
            + ((pts[:, 1] - centers_xy[i, 1]) / blob_r[i, 1]) ** 2
            + ((pts[:, 2] - centers_z[i]) / blob_r[i, 2]) ** 2
        )
        inside |= (d < 1.0) & (rng.random(pts.shape[0]) > 0.10)
    for _ in range(NOISY_INPUT_DROPLET_COUNT):
        angle, off = rng.uniform(0.0, 2.0 * np.pi), rng.uniform(0.014, 0.030)
        cx, cy, cz = (
            off * np.cos(angle),
            off * np.sin(angle),
            rng.uniform(z0 + 0.018, z1 - 0.018),
        )
        rr = rng.uniform(0.0045, 0.0090)
        inside |= (
            (pts[:, 0] - cx) ** 2
            + (pts[:, 1] - cy) ** 2
            + ((pts[:, 2] - cz) * 1.35) ** 2
        ) < rr * rr
    inside &= np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2) < NOISY_INPUT_RADIUS
    pts = pts[inside]
    vel = np.zeros_like(pts)
    z_norm = (pts[:, 2] - z0) / max(z1 - z0, 1.0e-6)
    vel[:, 2] = -0.32 - 0.20 * (1.0 - z_norm) + rng.normal(0.0, 0.035, pts.shape[0])
    vel[:, 0:2] = rng.normal(0.0, 0.018, size=(pts.shape[0], 2))
    return pts.astype(np.float32), vel.astype(np.float32), mass, radius


@dataclass
class SimulationOptions:
    """Static MPM/material/timestepping parameters shared by every nozzle world."""

    height: float = 0.2
    z_offset: float = 0.2
    sim_time: float = 1.25
    fps: float = 90.0
    substeps: int = 6
    gravity: tuple[float, float, float] = (0.0, 0.0, -10.0)
    density: float = 1000.0
    viscosity: float = 58.0
    tensile_yield_ratio: float = 1.0
    friction: float = 0.0
    ground_friction: float = 0.7
    funnel_friction: float = 0.0
    voxel_size: float = 0.012
    tolerance: float = 5.0e-7
    max_iterations: int = 220
    world_spacing: float = 0.38
    measurement_drop: float = 0.06
    measurement_window: float = 0.09
    target_flow_rate_norm: float = 0.75
    flow_smoothing_frames: int = 5
    flow_warmup_frames: int = 3
    floor_buffer: float = 0.19
    floor_thickness: float = 0.004
    particle_seed: int = DEFAULT_PARTICLE_SEED


def _build_world(design: np.ndarray, opts: SimulationOptions) -> newton.ModelBuilder:
    profile = design_to_profile(design)
    b = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(b)
    vertices, indices = build_nozzle_mesh(profile, opts.height, opts.z_offset)
    mesh = newton.Mesh(vertices, indices, compute_inertia=False, is_solid=True)
    b.add_shape_mesh(
        body=-1,
        mesh=mesh,
        cfg=newton.ModelBuilder.ShapeConfig(
            mu=opts.funnel_friction, margin=NOZZLE_COLLISION_MARGIN
        ),
        label="nozzle",
    )
    pts, vel, mass, radius = emit_messy_input_particles(
        opts.z_offset + opts.height,
        opts.voxel_size,
        opts.density,
        seed=opts.particle_seed,
    )
    if pts.shape[0] > 0:
        b.add_particles(
            pos=pts.tolist(),
            vel=vel.tolist(),
            mass=[mass] * pts.shape[0],
            radius=[radius] * pts.shape[0],
        )
    return b


def _add_finite_floor(
    scene: newton.ModelBuilder,
    designs: np.ndarray,
    offsets: np.ndarray,
    opts: SimulationOptions,
) -> None:
    max_r = float(
        max(np.max(designs[:, 0]) + NOZZLE_WALL_THICKNESS, NOISY_INPUT_RADIUS)
    )
    lo = offsets.min(0) - max_r - opts.floor_buffer
    hi = offsets.max(0) + max_r + opts.floor_buffer
    center = wp.vec3(
        0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), -opts.floor_thickness
    )
    scene.add_shape_box(
        body=-1,
        xform=wp.transform(center, wp.quat_identity()),
        hx=0.5 * (hi[0] - lo[0]),
        hy=0.5 * (hi[1] - lo[1]),
        hz=opts.floor_thickness,
        cfg=newton.ModelBuilder.ShapeConfig(mu=opts.ground_friction),
        label="floor",
    )


def build_multi_world_model(
    designs: np.ndarray, opts: SimulationOptions, offsets: np.ndarray | None = None
):
    """Build one Newton model holding ``B`` nozzle worlds.

    By default the worlds are laid out in a row along +X (used by the
    active-learning oracle). Pass ``offsets`` (shape ``(B, 3)``) to place them on
    an arbitrary layout such as a grid (used by the render script).
    """
    designs = np.asarray(designs, dtype=np.float32)
    B = designs.shape[0]
    scene = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(scene)
    if offsets is None:
        offsets = np.zeros((B, 3), dtype=np.float32)
        offsets[:, 0] = np.arange(B, dtype=np.float32) * opts.world_spacing
    else:
        offsets = np.asarray(offsets, dtype=np.float32).reshape(B, 3)
    _add_finite_floor(scene, designs, offsets, opts)
    for i in range(B):
        scene.add_world(
            _build_world(designs[i], opts),
            xform=wp.transform(
                wp.vec3(
                    float(offsets[i, 0]), float(offsets[i, 1]), float(offsets[i, 2])
                ),
                wp.quat_identity(),
            ),
        )
    model = scene.finalize()
    model.set_gravity(list(opts.gravity))
    model.mpm.viscosity.fill_(opts.viscosity)
    model.mpm.tensile_yield_ratio.fill_(opts.tensile_yield_ratio)
    model.mpm.friction.fill_(opts.friction)
    return model, offsets


def make_mpm_solver(model: newton.Model, opts: SimulationOptions) -> SolverImplicitMPM:
    """Build the implicit-MPM solver for ``model`` from ``opts`` (Newton's nozzle bases)."""
    cfg = SolverImplicitMPM.Config()
    cfg.voxel_size = opts.voxel_size
    cfg.tolerance = opts.tolerance
    cfg.max_iterations = opts.max_iterations
    # Discretization bases that match Newton's nozzle example.
    cfg.strain_basis = "P0"
    cfg.velocity_basis = "Q1"
    cfg.collider_basis = "S2"
    return SolverImplicitMPM(model, cfg)


def _combined_metric(
    spread, flow_rate_norm, flow_stability, flow_coverage, target
) -> float:
    target = max(float(target), 1.0e-6)
    target_error = abs(float(flow_rate_norm) - target) / target
    low_coverage = max(0.0, 0.65 - float(flow_coverage))
    return float(
        1.25 * target_error
        + 0.80 * flow_stability
        + 0.10 * float(spread) / 0.055
        + 1.75 * low_coverage**2
    )


def _engineering_penalty(design: np.ndarray, opts: SimulationOptions) -> float:
    repaired = repair_design(design)
    profile = design_to_profile(repaired)
    dz = opts.height / max(1, profile.shape[0] - 1)
    slope_excess = max(
        0.0, float(np.max(np.abs(np.diff(profile))) / max(dz, 1.0e-6)) - 0.55
    )
    aperture_floor = max(0.0, 0.016 - float(repaired[2]))
    return float(0.030 * slope_excess**2 + 30.0 * aperture_floor**2)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or window <= 1:
        return values
    k = min(int(window), values.size)
    return np.convolve(values, np.ones(k) / k, mode="valid")


def evaluate_designs(
    designs: np.ndarray, opts: SimulationOptions, device: str | None = None
) -> tuple[np.ndarray, dict]:
    """Simulate a batch of nozzle designs in Newton MPM and return the objective.

    Builds a ``B``-world model, steps the implicit-MPM solver until the inlet
    charge drains, and measures the normalized mass-flow rate, its stability, and
    spread at a plane below each nozzle. Returns ``(metrics, details)`` where
    ``metrics`` is the ``(B,)`` objective to minimize, combining the flow-quality
    score (see :func:`_combined_metric`) with a manufacturability term (see
    :func:`_engineering_penalty`).

    ``device`` selects the Warp device for the model build and MPM solve; when
    ``None`` the solve runs on Warp's default device.
    """
    designs = np.asarray(designs, dtype=np.float32)
    scope = wp.ScopedDevice(device) if device is not None else nullcontext()
    with scope:
        model, offsets = build_multi_world_model(designs, opts)
        solver = make_mpm_solver(model, opts)
        B = offsets.shape[0]
        s0, s1 = model.state(), model.state()
        sim_dt = 1.0 / (opts.fps * opts.substeps)
        n_frames = max(1, int(opts.sim_time * opts.fps))
        pw = model.particle_world.numpy()
        z_meas = opts.z_offset - opts.measurement_drop
        measurement_min_z = z_meas - opts.measurement_window

        spread_accum = np.zeros(B)
        spread_count = np.zeros(B, dtype=np.int64)
        first_exit = np.full(B, -1, dtype=np.int64)
        finish = np.full(B, -1, dtype=np.int64)
        crossed = np.zeros(B, dtype=np.int64)
        failed = np.zeros(B, dtype=bool)
        counts = np.array([(pw == w).sum() for w in range(B)], dtype=np.int64)
        flow_samples: list[list[float]] = [[] for _ in range(B)]
        min_exit = np.maximum(4, np.ceil(0.05 * np.maximum(counts, 1)).astype(np.int64))
        max_above = np.maximum(
            2, np.ceil(0.02 * np.maximum(counts, 1)).astype(np.int64)
        )
        prev_z = s0.particle_q.numpy()[:, 2].copy()

        for frame in range(n_frames):
            for _ in range(opts.substeps):
                solver.step(s0, s1, None, None, sim_dt)
                s0, s1 = s1, s0
            pos = s0.particle_q.numpy()
            finite = np.isfinite(pos).all(axis=1)
            if not finite.all():
                for w in range(B):
                    if np.any((pw == w) & ~finite):
                        failed[w] = True
                        finish[w] = frame
                pos = np.nan_to_num(pos, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
            all_done = True
            for w in range(B):
                if failed[w] or finish[w] >= 0:
                    continue
                m_world = pw == w
                m_cross = (
                    m_world
                    & (prev_z >= z_meas)
                    & (pos[:, 2] < z_meas)
                    & (pos[:, 2] > measurement_min_z)
                )
                new_cross = int(m_cross.sum())
                crossed[w] += new_cross
                if first_exit[w] < 0 and crossed[w] >= int(min_exit[w]):
                    first_exit[w] = frame
                if first_exit[w] >= 0:
                    flow_samples[w].append(
                        new_cross * opts.fps / max(1, int(counts[w]))
                    )
                    m_win = (
                        m_world & (pos[:, 2] < z_meas) & (pos[:, 2] > measurement_min_z)
                    )
                    pq = pos[m_win] - offsets[w]
                    if pq.shape[0] >= 4:
                        spread_accum[w] += float(
                            np.sqrt(np.var(pq[:, 0]) + np.var(pq[:, 1]))
                        )
                        spread_count[w] += 1
                remaining = int((m_world & (pos[:, 2] >= z_meas)).sum())
                if first_exit[w] >= 0 and remaining <= int(max_above[w]):
                    finish[w] = frame
                else:
                    all_done = False
            prev_z = pos[:, 2].copy()
            if all_done:
                break

    spreads = np.zeros(B, dtype=np.float32)
    flow_norm = np.zeros(B, dtype=np.float32)
    flow_stab = np.ones(B, dtype=np.float32)
    coverage = np.zeros(B, dtype=np.float32)
    for w in range(B):
        if failed[w]:
            spreads[w], flow_stab[w] = 0.20, 3.0
            continue
        spreads[w] = (
            float(spread_accum[w] / spread_count[w]) if spread_count[w] > 0 else 0.10
        )
        if counts[w] > 0:
            coverage[w] = crossed[w] / float(counts[w])
        samples = _smooth(np.asarray(flow_samples[w]), opts.flow_smoothing_frames)
        # Always drop the startup-transient warmup frames so the flow metrics are
        # not biased by them, even for short post-exit windows (where the trim may
        # leave no samples and the defaults flow_norm=0 / flow_stab=1 then stand,
        # a penalty for a poor/short design).
        samples = samples[opts.flow_warmup_frames :]
        if samples.size > 0:
            flow_norm[w] = float(np.mean(samples))
            flow_stab[w] = float(np.std(samples) / max(float(np.mean(samples)), 1.0e-6))

    metrics = np.array(
        [
            _combined_metric(
                spreads[i],
                flow_norm[i],
                flow_stab[i],
                coverage[i],
                opts.target_flow_rate_norm,
            )
            + _engineering_penalty(designs[i], opts)
            for i in range(B)
        ],
        dtype=np.float32,
    )
    details = {
        "flow_rate_norm": flow_norm,
        "flow_stability": flow_stab,
        "flow_coverage": coverage,
        "spread": spreads,
    }
    return metrics, details
