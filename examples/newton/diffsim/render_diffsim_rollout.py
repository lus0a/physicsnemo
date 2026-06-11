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

"""Render the ball diffsim rollout as a GIF from the actual Newton viewer.

This writes the viewer frame used in the diffsim README: the ball thrown toward
a held-out endpoint under the safeguarded BPTT/Newton launch, with its flight
trajectory and target plate marked. The plate's near face is one ball radius
from the requested center position, so a zero-miss rollout touches it instead of
overlapping a solid target marker.

The target is held out from training: the surrogate is target-conditioned and was
fit on a distribution of reachable targets, so it proposes a launch for a brand
new target without a solver in its BPTT loop. Newton then refines both the cold
and BPTT starts and keeps the better full-simulation result. The surrogate is a
gradient surrogate, not a replacement for the rendered Newton rollout.

The camera is derived analytically (look-at -> pitch/yaw for a Z-up scene) to
frame the whole trajectory together with the target. GIFs are assembled with PIL
because imageio is not in the Newton venv.

Run from the PhysicsNeMo repository root:
    uv run python examples/newton/diffsim/render_diffsim_rollout.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ball_problem as ball
import newton
import numpy as np
import torch
import warp as wp

from physicsnemo.experimental.integrations.newton import (
    field_to_torch,
    optimize_field_in_newton_multistart,
)
from physicsnemo.experimental.integrations.newton.visualization import (
    aim_camera,
    capture_frame,
    headless_viewer,
    save_gif,
)


def _build_ball(args):
    """Train the target-conditioned ball surrogate, pick one held-out target, then
    run a safeguarded cold/BPTT Newton refinement portfolio.

    Returns the env, the safeguarded launch plan, and the held-out target. The
    target is not used in training: the surrogate is conditioned on the target
    and fit on a distribution of reachable targets, so optimizing a launch for
    this brand new target costs no solver rollouts inside the BPTT loop. Newton
    fully refines both the nominal cold start and the BPTT proposal and renders
    the better full-Newton branch.
    """
    torch.manual_seed(args.seed)
    env = ball.make_env(args)
    surrogate, _train, held_out, _fit, _evaluation = ball.build_surrogate(env, args)

    # A brand-new reachable target, seeded apart from the train/held-out harvests so
    # it is genuinely held out from training.
    rng = np.random.default_rng(args.seed + 500_000)
    target = ball.reachable_target(env, rng, args.steps)

    # Surrogate-optimize the launch for this target (no simulator in the loop), then
    # fully refine both the cold and BPTT starts and keep the better real branch.
    batch = ball.single_target_batch(held_out, target)
    starts = ball.feasible_sobol_launches(args.opt_samples, seed=args.seed + 600_000)
    proposal = surrogate.optimize_multistart(
        batch,
        starts=args.opt_samples,
        initial_params=starts,
        steps=args.opt_steps,
        lr=ball.BPTT_LR,
        seed=args.seed + 600_000,
    )
    plan = optimize_field_in_newton_multistart(
        env,
        loss_fn=ball.ball_loss_fn(env, target),
        field="particle_qd",
        initials=np.stack(
            (
                ball.LAUNCH_MEAN,
                np.asarray(proposal["best_params"], np.float32).reshape(-1),
            )
        ),
        optimization_steps=ball.NEWTON_REFINE_STEPS,
        lr=ball.NEWTON_REFINE_LR,
        steps=args.steps,
    )
    return env, plan, np.asarray(target, np.float32)


def _log_target(viewer, target, approach, ball_radius, color=(0.95, 0.2, 0.55)):
    """Draw a thin target plate tangent to a ball centered at ``target``."""
    direction = 1.0 if approach[2] >= 0.0 else -1.0
    half_extents = np.array((0.2, 0.2, 0.025), dtype=np.float32)
    center = np.asarray(target, np.float32).copy()
    center[2] += direction * (ball_radius + half_extents[2])

    # Build the warp arrays on the viewer's device so they match the device the
    # viewer launches its render kernels on (model.device, set by set_model);
    # arrays left on warp's default device would mismatch and crash log_shapes.
    viewer.log_shapes(
        "/target",
        newton.GeoType.BOX,
        tuple(float(x) for x in half_extents),
        wp.array(
            [wp.transform(tuple(float(x) for x in center), wp.quat_identity())],
            dtype=wp.transform,
            device=viewer.device,
        ),
        wp.array([wp.vec3(*color)], dtype=wp.vec3, device=viewer.device),
    )


def _log_trajectory(viewer, points, color=(0.1, 0.85, 0.3)):
    """Growing trajectory trail. ``log_lines`` only draws a thin screen-space line,
    so the path is drawn as a dense trail of small spheres, which reads clearly."""
    if not points:
        return
    arr = np.asarray(points, np.float32)
    # Match the viewer's device (see _log_target) so log_points' kernel launch and
    # the arrays agree. radii is one float32 value per point.
    viewer.log_points(
        "/trajectory",
        wp.array(arr, dtype=wp.vec3, device=viewer.device),
        radii=wp.full(len(arr), 0.05, dtype=wp.float32, device=viewer.device),
        colors=wp.array(
            [wp.vec3(*color)] * len(arr), dtype=wp.vec3, device=viewer.device
        ),
    )


def _subject_track(states, frame_idx):
    """Per-frame ball position (the single particle, read as the per-particle mean)."""
    return np.array(
        [
            field_to_torch(states[i].particle_q).mean(0).detach().cpu().numpy()
            for i in frame_idx
        ]
    )


def render_newton_gif(env, states, target, path, *, args, stride=None, hold_frames=0):
    """Frames of the real Newton rollout, annotated with the ball's trajectory
    and endpoint target.

    ``hold_frames`` repeats the objective-horizon frame so the arrival reads
    clearly. The camera is fixed to frame the whole trajectory together with the
    target so both stay in view.
    """
    if stride is None:
        # keep_states returns one state per substep; sample one frame per sim frame.
        stride = max(1, getattr(env, "substeps", 1))
    frame_idx = list(range(0, len(states), stride))
    if frame_idx[-1] != len(states) - 1:
        frame_idx.append(len(states) - 1)
    track = _subject_track(states, frame_idx)
    approach = track[-1] - track[-2]
    ball_radius = float(env.model.particle_radius.numpy()[0])

    viewer = headless_viewer(args.width, args.height)
    viewer.set_model(env.model)
    viewer.show_particles = True
    # frame the whole trajectory together with the target so both stay in view
    pts = np.vstack([track, np.asarray(target, np.float32)[None]])
    span = np.maximum(pts.max(0) - pts.min(0), 0.3)
    lo, hi = pts.min(0) - 0.22 * span, pts.max(0) + 0.22 * span
    aim_camera(viewer, lo, hi, azim=args.azim, elev=args.elev)

    frames, accum = [], []
    for k, i in enumerate(frame_idx):
        accum.append(tuple(track[k]))
        viewer.begin_frame(i * env.dt)
        viewer.log_state(states[i])
        _log_trajectory(viewer, accum)
        _log_target(viewer, target, approach, ball_radius)
        viewer.end_frame()
        frames.append(capture_frame(viewer))
    frames.extend([frames[-1]] * hold_frames)
    save_gif(frames, path, fps=args.fps, palette=True, optimize=True)
    return frames


def run(args):
    """Render and save the optimized held-out-target Newton rollout."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env, plan, target = _build_ball(args)
    env.reset(particle_qd=np.asarray(plan["best_params"], np.float32).reshape(1, 3))
    states = env.rollout(args.steps, keep_states=True).states
    if states is None:
        raise RuntimeError("render rollout did not retain Newton states")
    render_newton_gif(
        env,
        states,
        target,
        out_dir / "diffsim_ball_frame.gif",
        args=args,
        hold_frames=16,
    )
    return str(out_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the differentiable-ball renderer."""
    p = argparse.ArgumentParser(description="Render the Newton ball diffsim rollout")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "diffsim",
    )
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--azim", type=float, default=55.0)
    p.add_argument("--elev", type=float, default=-18.0)
    # Match the scorecard config so the target-conditioned surrogate generalizes
    # to the held-out target and supplies a useful BPTT proposal. The safeguarded
    # Newton portfolio renders the better refined full-simulation branch.
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--val-samples", type=int, default=32)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--opt-samples", type=int, default=64)
    p.add_argument("--opt-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=123)
    # Default to CUDA when present: headless GL frame capture is unsafe on CPU
    # in Newton releases before 1.2.2 when CUDA is available.
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--newton-device", default=default_device)
    p.add_argument("--torch-device", default=default_device)
    return p.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
