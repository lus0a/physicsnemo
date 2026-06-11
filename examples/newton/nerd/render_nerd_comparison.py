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

"""Render the NeRD cartpole rollout next to the analytical solver, as a GIF.

This writes the viewer frame used in the NeRD README: the same cartpole released
from one initial state and advanced two ways, the analytical Featherstone solver
on the left and the learned NeRD step on the right. Both start from the identical
state; the right clip is then driven entirely by NeRD (fully autoregressive,
free-running from an empty history). The cartpole is the real Newton scene, so
this renders true physics on the left and the learned dynamics on the right.

A small NeRD model is trained inline (so the script is self-contained), then a
single-world cartpole is replayed under each stepper. GIFs are assembled with PIL
because imageio is not in the Newton venv.

Run from the PhysicsNeMo repository root:
    uv run python \
        examples/newton/nerd/render_nerd_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartpole_problem as cartpole
import newton
import numpy as np
import torch

from physicsnemo.experimental.integrations.newton import (
    NeRDTrainingConfig,
    field_to_torch,
    fit_nerd,
    resolve_device,
)
from physicsnemo.experimental.integrations.newton.visualization import (
    capture_frame,
    draw_text,
    frame_bounding_box,
    headless_viewer,
    save_gif,
    stack_horizontal,
)


def _train_nerd(args, device: str) -> object:
    """Train a NeRD model on the replicated cartpole."""
    torch.manual_seed(args.seed)
    scene = cartpole.CartpoleScene(args.num_worlds, device)
    config = NeRDTrainingConfig(
        context_frames=args.context_frames,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
    )
    log = print if args.verbose else (lambda _message: None)
    dynamics_model, model_kwargs = cartpole.model_selection(args)
    trained = fit_nerd(
        cartpole.make_problem(
            scene,
            cart_band=args.cart_band,
            init_velocity=args.init_velocity,
            force_scale=args.force_scale,
        ),
        num_trajectories=args.num_worlds,
        steps=args.steps,
        config=config,
        dynamics_model=dynamics_model,
        model_kwargs=model_kwargs,
        max_abs_state=1.0e5,
        device=device,
        seed=args.seed,
        log=log,
    )
    return trained


def _roll_teacher(scene, q0, qd0, steps):
    """Roll the analytical solver, returning the per-frame ``joint_q`` trajectory."""
    state_0, state_1, control = (
        scene.model.state(),
        scene.model.state(),
        scene.model.control(),
    )
    field_to_torch(state_0.joint_q).copy_(
        torch.as_tensor(q0, dtype=torch.float32).reshape(-1)
    )
    field_to_torch(state_0.joint_qd).copy_(
        torch.as_tensor(qd0, dtype=torch.float32).reshape(-1)
    )
    newton.eval_fk(scene.model, state_0.joint_q, state_0.joint_qd, state_0)
    traj_q, traj_qd = (
        [field_to_torch(state_0.joint_q).clone()],
        [field_to_torch(state_0.joint_qd).clone()],
    )
    for _ in range(steps):
        for _ in range(scene.substeps):
            state_0.clear_forces()
            scene.solver.step(state_0, state_1, control, None, scene.sim_dt)
            state_0, state_1 = state_1, state_0
        traj_q.append(field_to_torch(state_0.joint_q).clone())
        traj_qd.append(field_to_torch(state_0.joint_qd).clone())
    return torch.stack(traj_q), torch.stack(traj_qd)


def _roll_nerd(scene, trained, q0, qd0, steps, device):
    """Free-run NeRD from the same initial state as the teacher (fully
    autoregressive, history grows from empty); return the per-frame ``joint_q``."""
    learned_step = trained.as_step_model(
        newton_model=scene.model,
        device=device,
    )
    state_0, state_1 = (
        scene.model.state(),
        scene.model.state(),
    )
    field_to_torch(state_0.joint_q).copy_(
        torch.as_tensor(q0, dtype=torch.float32).reshape(-1)
    )
    field_to_torch(state_0.joint_qd).copy_(
        torch.as_tensor(qd0, dtype=torch.float32).reshape(-1)
    )
    traj_q = [field_to_torch(state_0.joint_q).clone()]
    zero_inputs = torch.zeros((1, trained.external_input_dim), device=device)
    for _ in range(steps):
        learned_step.step_with_inputs(
            state_0,
            state_1,
            zero_inputs,
            dt=scene.dt,
        )
        state_0, state_1 = state_1, state_0
        traj_q.append(field_to_torch(state_0.joint_q).clone())
    return torch.stack(traj_q)


def _replay_frames(scene, viewer, traj_q, camera, label, args):
    """Replay a ``joint_q`` trajectory through the viewer, returning labelled frames."""
    state = scene.model.state()
    viewer.set_model(scene.model)
    viewer.set_camera(*camera)
    frames = []
    for i in range(0, traj_q.shape[0], args.stride):
        field_to_torch(state.joint_q).copy_(traj_q[i])
        newton.eval_fk(scene.model, state.joint_q, state.joint_qd, state)
        viewer.begin_frame(i * scene.dt)
        viewer.log_state(state)
        viewer.end_frame()
        frames.append(draw_text(capture_frame(viewer), label))
    return frames


def _camera(scene, traj_q):
    """Frame the cartpole face-on, looking perpendicular to its X-Z motion plane."""
    model1 = scene.model
    state = model1.state()
    lo = np.array([1e9, 1e9, 1e9])
    hi = -lo
    for i in (0, traj_q.shape[0] // 2, traj_q.shape[0] - 1):
        field_to_torch(state.joint_q).copy_(traj_q[i])
        newton.eval_fk(model1, state.joint_q, state.joint_qd, state)
        pos = field_to_torch(state.body_q)[:, :3].cpu().numpy()
        lo = np.minimum(lo, pos.min(0))
        hi = np.maximum(hi, pos.max(0))
    # Body-origin bounds alone place the camera too close, while framing the
    # entire 8 m rail makes the mechanism unreadably small. Keep a useful
    # face-on crop with room for the passive cart motion and full pole swing.
    lo[0] = min(lo[0], -2.5)
    hi[0] = max(hi[0], 2.5)
    lo[2] = min(lo[2], -1.3)
    hi[2] = max(hi[2], 1.5)
    span = np.maximum(hi - lo, 1.0)
    return frame_bounding_box(
        lo - 0.12 * span, hi + 0.12 * span, azim=90.0, elev=0.0, margin=1.4
    )


def run(args: argparse.Namespace) -> str:
    """Train the inline learned model, render both rollouts, and return the GIF path."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(resolve_device(args.device))

    trained = _train_nerd(args, device)

    render_scene = cartpole.CartpoleScene(1, device)
    # A representative passive swing: the pole released well off the hanging
    # equilibrium, the cart given a small push so it travels along the rail.
    q0 = np.array([[0.0, np.pi - args.release]], np.float32)
    qd0 = np.array([[args.cart_push, 0.0]], np.float32)
    teacher_q, _ = _roll_teacher(render_scene, q0, qd0, args.frames)
    nerd_q = _roll_nerd(render_scene, trained, q0, qd0, args.frames, device)

    camera = _camera(render_scene, teacher_q)
    viewer = headless_viewer(args.width, args.height)
    teacher_frames = _replay_frames(
        render_scene, viewer, teacher_q, camera, "analytical solver", args
    )
    nerd_frames = _replay_frames(
        render_scene, viewer, nerd_q, camera, "scaled inline NeRD", args
    )

    frames = [
        stack_horizontal([a, b], gap=4) for a, b in zip(teacher_frames, nerd_frames)
    ]
    path = out_dir / "cartpole_nerd.gif"
    save_gif(frames, path, fps=args.fps, palette=True)
    return str(path)


def parse_args() -> argparse.Namespace:
    """Parse renderer command-line options."""
    parser = argparse.ArgumentParser(description="Render NeRD vs analytical cartpole")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "cartpole_nerd",
    )
    parser.add_argument("--num-worlds", type=int, default=1024)
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="per-trajectory rollout length for the teacher data used to train NeRD",
    )
    parser.add_argument("--cart-band", type=float, default=1.0)
    parser.add_argument("--init-velocity", type=float, default=1.0)
    parser.add_argument(
        "--force-scale",
        "--torque-scale",
        dest="force_scale",
        type=float,
        default=1500.0,
        help="cart prismatic-joint force scale in newtons (--torque-scale is a "
        "deprecated alias)",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--steps-per-epoch",
        "--iters-per-epoch",
        dest="steps_per_epoch",
        type=int,
        default=500,
    )
    parser.add_argument("--context-frames", type=int, default=10)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument(
        "--model",
        choices=tuple(cartpole.MODEL_LABELS),
        default="nerd-transformer",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=160,
        help="number of frames in the rendered comparison rollout",
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--release", type=float, default=0.6)
    parser.add_argument("--cart-push", type=float, default=0.6)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        help="Newton/model device; defaults to the active rank or PyTorch default",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
