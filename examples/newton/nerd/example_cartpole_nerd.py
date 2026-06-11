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

"""Train and evaluate NeRD on a Newton cartpole.

The Cartpole-specific scene and state/control policy live in
``cartpole_problem.py``. This script owns the user-facing training workflow,
scorecard, and command-line presets.

Run from the PhysicsNeMo repository root:

.. code-block:: bash

    uv run python examples/newton/nerd/example_cartpole_nerd.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartpole_problem as cartpole
import numpy as np
import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.integrations.newton import (
    NeRDTrainingConfig,
    fit_nerd,
    is_main_process,
    resolve_device,
)


def format_report(
    name: str,
    evaluation: cartpole.CartpoleEvaluation,
    config: NeRDTrainingConfig,
    history: dict,
) -> str:
    """Render the Cartpole accuracy and workflow metrics."""
    rows = [
        ("model", history.get("model_class", "unknown")),
        ("training trajectories", f"{history.get('trajectory_count', 0):,}"),
        ("optimizer updates", f"{history.get('optimizer_updates', 0):,}"),
        ("history window (frames)", str(config.context_frames)),
        (
            "teacher-forced one-step base / pole error (m / rad)",
            f"{evaluation.teacher_forced_base_err:.4f} / "
            f"{evaluation.teacher_forced_joint_err:.4f}",
        ),
        ("finite passive rollouts", f"{evaluation.finite_fraction * 100:.0f}%"),
        ("analytical teacher step (ms)", f"{evaluation.teacher_step_ms:.3f}"),
        ("NeRD step (ms)", f"{evaluation.nerd_step_ms:.3f}"),
        ("per-step speedup", f"{evaluation.speedup:.1f}x"),
    ]
    rows.extend(
        (
            f"time-averaged free-running error over {horizon} steps (m / rad)",
            f"{evaluation.free_running_base_err[horizon]:.4f} / "
            f"{evaluation.free_running_joint_err[horizon]:.4f}",
        )
        for horizon in evaluation.horizons
    )
    table = "\n".join(f"| {label} | {value} |" for label, value in rows)
    return (
        f"## {name}: NeRD learned dynamics\n\n"
        "The model is trained and deployed through the representation-generic "
        "`fit_nerd` workflow, then evaluated free-running from frame zero.\n\n"
        f"| metric | value |\n| --- | ---: |\n{table}\n"
    )


def run(args: argparse.Namespace) -> str:
    """Train NeRD and evaluate held-out passive Cartpole rollouts."""
    DistributedManager.initialize()
    device = str(resolve_device(args.device))
    torch.manual_seed(args.seed)

    scene = cartpole.CartpoleScene(args.num_worlds, device)
    dynamics_model, model_kwargs = cartpole.model_selection(args)
    problem = cartpole.make_problem(
        scene,
        cart_band=args.cart_band,
        init_velocity=args.init_velocity,
        force_scale=args.force_scale,
    )
    log = print if args.verbose else (lambda _message: None)
    trained = fit_nerd(
        problem,
        num_trajectories=args.num_trajectories,
        steps=args.steps,
        config=cartpole.training_config(args),
        dynamics_model=dynamics_model,
        model_kwargs=model_kwargs,
        max_abs_state=1.0e5,
        device=device,
        seed=args.seed,
        log=log,
    )
    if not is_main_process():
        return ""

    init_q, init_qd = cartpole.evaluation_initial_state(
        trained.codec.layout,
        args.seed + 1000,
        args.cart_band,
        args.init_velocity,
    )
    evaluation = cartpole.evaluate(
        trained,
        scene,
        init_q=init_q,
        init_qd=init_qd,
        horizons=tuple(args.horizons),
        device=device,
    )

    label = cartpole.MODEL_LABELS[args.model]
    report = format_report(
        f"cartpole {label}", evaluation, trained.config, trained.metadata
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = cartpole.artifact_stem(args)
    (output / f"{stem}_report.md").write_text(report, encoding="utf-8")
    plot_report(
        trained.metadata,
        evaluation,
        scene.dt,
        output / f"{stem}_scorecard.png",
        label,
    )
    return report


def plot_report(
    history: dict,
    evaluation: cartpole.CartpoleEvaluation,
    dt: float,
    path: Path,
    model_label: str,
) -> None:
    """Write the Cartpole training and free-running scorecard."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    green, blue, slate = "#76b900", "#2b6cb0", "#4a5568"
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    figure.suptitle(
        f"Newton cartpole NeRD workflow: {model_label}",
        fontsize=14,
        fontweight="bold",
    )

    curve = history.get("train_curve", [])
    axes[0, 0].plot(range(1, len(curve) + 1), curve, color=green)
    axes[0, 0].set(
        title="(a) NeRD training loss",
        xlabel="epoch",
        ylabel="train MSE (normalized delta)",
        yscale="log",
    )

    horizons = sorted(evaluation.free_running_base_err)
    axes[0, 1].plot(
        horizons,
        [evaluation.free_running_base_err[h] for h in horizons],
        "o-",
        color=green,
        label=f"{model_label}: base (m)",
    )
    axes[0, 1].plot(
        horizons,
        [evaluation.free_running_joint_err[h] for h in horizons],
        "o-",
        color=blue,
        label=f"{model_label}: pole (rad)",
    )
    checkpoint_horizons = [
        horizon for horizon in horizons if horizon in cartpole.CHECKPOINT_JOINT_ERR
    ]
    if checkpoint_horizons:
        axes[0, 1].plot(
            checkpoint_horizons,
            [cartpole.CHECKPOINT_BASE_ERR[h] for h in checkpoint_horizons],
            "+:",
            color=green,
            alpha=0.6,
            label="paper checkpoint: base",
        )
        axes[0, 1].plot(
            checkpoint_horizons,
            [cartpole.CHECKPOINT_JOINT_ERR[h] for h in checkpoint_horizons],
            "x--",
            color=slate,
            alpha=0.6,
            label="paper checkpoint: pole",
        )
    axes[0, 1].set(
        title=f"(b) run vs checkpoint (jerk {evaluation.jerk_ratio:.2f}x)",
        xlabel="free-running horizon (steps)",
        ylabel="time-averaged tracking error",
        yscale="log",
    )
    axes[0, 1].legend(fontsize=8)

    window = min(150, evaluation.overlay["teacher_q"].shape[0])
    teacher = evaluation.overlay["teacher_q"][:window]
    prediction = evaluation.overlay["nerd_q"][:window]
    time = np.arange(window) * dt
    axes[1, 0].plot(time, teacher[:, 1], color=slate, label="analytical solver")
    axes[1, 0].plot(time, prediction[:, 1], "--", color=green, label=model_label)
    axes[1, 0].set(
        title="(c) free-running pole angle",
        xlabel="time (s)",
        ylabel="pole angle (rad)",
    )
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(time, teacher[:, 0], color=slate, label="analytical solver")
    axes[1, 1].plot(time, prediction[:, 0], "--", color=green, label=model_label)
    axes[1, 1].axhline(4.0, color="0.7", ls=":", lw=0.8)
    axes[1, 1].axhline(-4.0, color="0.7", ls=":", lw=0.8)
    axes[1, 1].set(
        title="(d) free-running cart position",
        xlabel="time (s)",
        ylabel="cart position (m)",
    )
    axes[1, 1].legend(fontsize=8)

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _apply_smoke(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the tiny end-to-end configuration."""
    args.num_worlds = 64
    args.num_trajectories = 64
    args.steps = 40
    args.epochs = 3
    args.steps_per_epoch = 20
    args.n_layer = 2
    args.n_head = 4
    args.n_embd = 32
    args.horizons = [20, 40]
    return args


def _apply_paper(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the paper-scale configuration."""
    args.num_worlds = 2048
    args.num_trajectories = 10_000
    args.steps = 100
    args.cart_band = 1.0
    args.init_velocity = 1.0
    args.force_scale = 1500.0
    args.epochs = 1000
    args.steps_per_epoch = 5000
    args.batch_size = 512
    args.context_frames = 10
    args.n_layer = 6
    args.n_head = 12
    args.n_embd = 192
    args.horizons = [100, 500, 1000]
    return args


def parse_args() -> argparse.Namespace:
    """Parse Cartpole training and evaluation options."""
    parser = argparse.ArgumentParser(
        description="NeRD learned dynamics on a Newton cartpole"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "cartpole_nerd",
    )
    parser.add_argument("--num-worlds", type=int, default=2048)
    parser.add_argument("--num-trajectories", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cart-band", type=float, default=1.0)
    parser.add_argument("--init-velocity", type=float, default=1.0)
    parser.add_argument(
        "--force-scale",
        "--torque-scale",
        dest="force_scale",
        type=float,
        default=1500.0,
        help="cart prismatic-joint force scale in newtons",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--steps-per-epoch",
        "--iters-per-epoch",
        dest="steps_per_epoch",
        type=int,
        default=500,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--context-frames", type=int, default=10)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument(
        "--model",
        choices=tuple(cartpole.MODEL_LABELS),
        default="nerd-transformer",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--verbose", action="store_true")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--smoke", action="store_true")
    preset.add_argument("--paper", action="store_true")
    args = parser.parse_args()
    if args.paper:
        return _apply_paper(args)
    return _apply_smoke(args) if args.smoke else args


if __name__ == "__main__":
    result = run(parse_args())
    if result:
        print(result)
