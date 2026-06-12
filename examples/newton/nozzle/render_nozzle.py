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

"""Render comparative views of the Newton MPM nozzle scene headlessly.

Companion render script for the nozzle inverse-design example. It builds the
nozzle surface-of-revolution mesh together with the MPM particle jet in a single
Newton model and renders them with the headless GL viewer. The comparative views
draw each nozzle mesh at its world offset and overlay the live particle state.

Two views:

* ``--view worlds``      several designs simulated side by side, the batched
                         worlds laid out in a grid.
* ``--view before-after`` the bootstrap design next to the optimized best in a
                         labeled two-world render.

Designs are read from ``outputs/nozzle/pareto_data.npz``, written by the example
under ``--save-pareto-data``.

Run from the PhysicsNeMo repository root (GPU recommended):
    uv run python examples/newton/nozzle/render_nozzle.py --view before-after
"""

from __future__ import annotations

import argparse
import colorsys
from contextlib import nullcontext
from pathlib import Path

import nozzle_scene as scene
import numpy as np
import warp as wp
from PIL import Image

from physicsnemo.experimental.integrations.newton import resolve_device
from physicsnemo.experimental.integrations.newton.visualization import (
    aim_camera,
    capture_frame,
    draw_text,
    headless_viewer,
    save_gif,
)

MESH_HALF_R = 0.0815  # nozzle inlet radius + wall, from the design bounds


def _newton_device_scope(device: str | None):
    """Warp ``ScopedDevice`` for the MPM solve, or a no-op when ``device`` is None.

    When ``device`` is unset the model/solver run on Warp's default device (the
    GPU when one is visible); otherwise the requested device is honored
    (``resolve_device`` also maps it to this rank's device under DistributedManager).
    """
    if device is None:
        return nullcontext()
    return wp.ScopedDevice(str(resolve_device(device)))


def draw_labels(frame: Image.Image, labels: list[tuple[float, str]]) -> Image.Image:
    """Stamp left-anchored captions at fractional x positions via ``draw_text``."""
    image = frame
    for frac_x, text in labels:
        image = draw_text(
            image,
            text,
            xy=(int(frac_x * image.width) + 4, 8),
            color=(220, 236, 255),
            background=(8, 10, 14, 180),
        )
    return image


def load_designs(
    args: argparse.Namespace, bounds: scene.DesignBounds
) -> dict[str, np.ndarray]:
    """Return the best and bootstrap designs from saved optimization data."""
    npz = Path(args.pareto_data)
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} does not exist; run example_mpm_nozzle_design.py "
            "--save-pareto-data first"
        )
    data = np.load(npz)
    units = data["units"]
    scores = data["scores"]
    boot = int(data["bootstrap"]) if "bootstrap" in data else len(units)
    best_i = (
        int(data["best_index"]) if "best_index" in data else int(np.nanargmin(scores))
    )
    boot_i = int(np.nanargmin(scores[:boot])) if boot > 0 else 0
    return {
        "best": scene.unit_to_design(units[best_i], bounds),
        "bootstrap": scene.unit_to_design(units[boot_i], bounds),
    }


def palette(n: int) -> list[tuple[float, float, float]]:
    """``n`` visually distinct nozzle colors (evenly spaced hues)."""
    return [colorsys.hsv_to_rgb(i / max(n, 1), 0.62, 0.95) for i in range(n)]


def grid_offsets(n: int, spacing: float) -> tuple[np.ndarray, int, int]:
    """Centered ``rows x cols`` grid of world origins in the X-Y plane."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    idx = np.arange(n)
    off = np.zeros((n, 3), dtype=np.float32)
    off[:, 0] = (idx % cols).astype(np.float32) * spacing
    off[:, 1] = (idx // cols).astype(np.float32) * spacing
    off[:, 0] -= 0.5 * (cols - 1) * spacing
    off[:, 1] -= 0.5 * (rows - 1) * spacing
    return off, cols, rows


def _wp_mesh(verts: np.ndarray, indices: np.ndarray) -> tuple[wp.array, wp.array]:
    """Pack (verts, triangle indices) into the warp arrays ``log_mesh`` expects."""
    pts = wp.array(np.ascontiguousarray(verts, dtype=np.float32), dtype=wp.vec3)
    tri = wp.array(
        np.ascontiguousarray(indices, dtype=np.int32).reshape(-1), dtype=wp.int32
    )
    return pts, tri


def _ground_quad(lo: np.ndarray, hi: np.ndarray, z: float) -> tuple[wp.array, wp.array]:
    """A single flat quad (two triangles) covering ``[lo, hi]`` at height ``z``."""
    v = np.array(
        [[lo[0], lo[1], z], [hi[0], lo[1], z], [hi[0], hi[1], z], [lo[0], hi[1], z]],
        dtype=np.float32,
    )
    return _wp_mesh(v, np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))


def render_manual(
    designs: np.ndarray,
    opts: scene.SimulationOptions,
    args: argparse.Namespace,
    out_path: Path,
    *,
    offsets: np.ndarray,
    colors: list[tuple[float, float, float]],
    labels: list[tuple[float, str]] | None = None,
    azim: float = 90.0,
    elev: float = -34.0,
    margin: float = 1.2,
) -> Path:
    """Render several nozzle worlds, drawing each nozzle and the jet explicitly.

    The default multi-world ``log_state`` path misplaces the static nozzle mesh
    shapes (a viewer artifact, the simulation itself is correct). Here every
    nozzle mesh is drawn at its own world offset with its own color via
    ``log_mesh`` and the particles are drawn from the live state via
    ``log_points``, both in world coordinates, so geometry and jet always align.
    """
    designs = np.atleast_2d(np.asarray(designs, dtype=np.float32))
    n = designs.shape[0]
    frames: list[Image.Image] = []
    with _newton_device_scope(args.newton_device):
        model, off = scene.build_multi_world_model(designs, opts, offsets=offsets)
        solver = scene.make_mpm_solver(model, opts)
        s0, s1 = model.state(), model.state()

        # Per-world nozzle meshes, vertices pre-translated into world space.
        meshes = []
        for i in range(n):
            verts, indices = scene.build_nozzle_mesh(
                scene.design_to_profile(designs[i]), opts.height, opts.z_offset
            )
            meshes.append(_wp_mesh(np.asarray(verts, np.float32) + off[i], indices))
        radius = float(
            scene.emit_messy_input_particles(
                opts.z_offset + opts.height,
                opts.voxel_size,
                opts.density,
                seed=opts.particle_seed,
            )[3]
        )

        lo = np.array(
            [
                off[:, 0].min() - MESH_HALF_R,
                off[:, 1].min() - MESH_HALF_R,
                -opts.floor_thickness,
            ],
            dtype=np.float64,
        )
        hi = np.array(
            [
                off[:, 0].max() + MESH_HALF_R,
                off[:, 1].max() + MESH_HALF_R,
                opts.z_offset + opts.height,
            ],
            dtype=np.float64,
        )
        pad = 0.06
        ground = _ground_quad(
            lo - [pad, pad, 0.0], hi + [pad, pad, 0.0], -opts.floor_thickness
        )

        viewer = headless_viewer(args.width, args.height)
        viewer.set_model(
            model
        )  # sets the Z-up camera convention; we draw geometry ourselves
        aim_camera(viewer, lo, hi, azim=azim, elev=elev, margin=margin)

        sim_dt = 1.0 / (opts.fps * opts.substeps)
        n_frames = max(1, int(opts.sim_time * opts.fps))
        stride = max(1, n_frames // args.max_frames)
        sand = (0.86, 0.74, 0.45)
        t = 0.0
        for f in range(n_frames):
            for _ in range(opts.substeps):
                solver.step(s0, s1, None, None, sim_dt)
                s0, s1 = s1, s0
            t += 1.0 / opts.fps
            if f % stride:
                continue
            pq = s0.particle_q.numpy()
            pq = pq[np.isfinite(pq).all(axis=1)]
            pts = wp.array(np.ascontiguousarray(pq, np.float32), dtype=wp.vec3)
            # log_points wants a per-point color array (the GL backend calls .numpy()).
            pt_colors = wp.array(
                np.tile(np.asarray(sand, np.float32), (pq.shape[0], 1)), dtype=wp.vec3
            )
            viewer.begin_frame(t)
            viewer.log_mesh("/ground", ground[0], ground[1], color=(0.32, 0.55, 0.78))
            for i in range(n):
                viewer.log_mesh(
                    f"/nozzle_{i}", meshes[i][0], meshes[i][1], color=colors[i]
                )
            viewer.log_points(
                "/jet",
                pts,
                radii=wp.full(len(pts), radius, dtype=wp.float32, device=viewer.device),
                colors=pt_colors,
            )
            viewer.end_frame()
            frame = capture_frame(viewer)  # RGB PIL.Image, alpha already stripped
            if labels:
                frame = draw_labels(frame, labels)
            frames.append(frame)
        viewer.close()

    if not frames:
        raise RuntimeError("no frames captured")
    save_gif(frames, out_path, fps=args.gif_fps)
    print(f"[render_nozzle] wrote {out_path} ({len(frames)} frames, {n} worlds)")
    return out_path


def make_options(args: argparse.Namespace) -> scene.SimulationOptions:
    """Build nozzle simulation options from command-line arguments."""
    return scene.SimulationOptions(
        sim_time=args.sim_time,
        fps=args.fps,
        substeps=args.substeps,
        max_iterations=args.max_iterations,
        voxel_size=args.voxel_size,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the nozzle renderer."""
    p = argparse.ArgumentParser(
        description="Render comparative Newton MPM nozzle scenes"
    )
    p.add_argument("--view", choices=("worlds", "before-after"), default="before-after")
    here = Path(__file__).parent
    p.add_argument(
        "--pareto-data",
        type=Path,
        default=here / "outputs" / "nozzle" / "pareto_data.npz",
    )
    p.add_argument(
        "--media-dir",
        "--img-dir",
        dest="media_dir",
        type=Path,
        default=here / "outputs" / "nozzle",
        help="directory for generated GIFs",
    )
    p.add_argument(
        "--num-worlds", type=int, default=81, help="designs in the --view worlds grid"
    )
    p.add_argument(
        "--grid-spacing",
        type=float,
        default=0.30,
        help="world spacing in the worlds grid",
    )
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--height", type=int, default=760)
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--gif-fps", type=float, default=24.0)
    p.add_argument("--sim-time", type=float, default=1.25)
    p.add_argument("--fps", type=float, default=90.0)
    p.add_argument("--substeps", type=int, default=6)
    p.add_argument("--max-iterations", type=int, default=220)
    p.add_argument("--voxel-size", type=float, default=0.012)
    p.add_argument(
        "--newton-device",
        default=None,
        help="Warp device for the MPM solve (default: Warp's default device)",
    )
    return p.parse_args()


def early_designs(
    args: argparse.Namespace, bounds: scene.DesignBounds, n: int
) -> np.ndarray:
    """The first ``n`` evaluated designs (append order), i.e. the diverse early rounds."""
    npz = Path(args.pareto_data)
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} does not exist; run example_mpm_nozzle_design.py "
            "--save-pareto-data first"
        )
    units = np.load(npz)["units"]
    take = units[: min(n, len(units))]
    return np.stack([scene.unit_to_design(u, bounds) for u in take], axis=0)


def main() -> int:
    """Render the selected nozzle visualization."""
    args = parse_args()
    bounds = scene.DesignBounds()
    opts = make_options(args)
    media_dir = Path(args.media_dir)
    if args.view == "worlds":
        # The diverse early designs on a centered grid, drawn from a pulled-back
        # elevated front view so every world is visible and the ground spans them.
        designs = early_designs(args, bounds, args.num_worlds)
        offsets, _, _ = grid_offsets(len(designs), args.grid_spacing)
        render_manual(
            designs,
            opts,
            args,
            media_dir / "newton_nozzle_worlds.gif",
            offsets=offsets,
            colors=palette(len(designs)),
            azim=90.0,
            elev=-40.0,
            margin=1.12,
        )
    else:  # before-after: bootstrap (left) vs optimized (right), front view
        d = load_designs(args, bounds)
        pair = np.stack([d["bootstrap"], d["best"]], axis=0)
        offsets, _, _ = grid_offsets(2, max(args.grid_spacing, 0.34))
        render_manual(
            pair,
            opts,
            args,
            media_dir / "newton_nozzle_before_after.gif",
            offsets=offsets,
            colors=[(0.90, 0.45, 0.18), (0.36, 0.74, 0.0)],
            labels=[(0.04, "bootstrap"), (0.55, "optimized")],
            azim=90.0,
            elev=-16.0,
            margin=1.5,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
