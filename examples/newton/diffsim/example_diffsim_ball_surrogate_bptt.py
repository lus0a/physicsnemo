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

"""Target-conditioned PhysicsNeMo surrogate for Newton's differentiable ball.

Newton gives us gradients through the physics: rolling a scene out under a
differentiable solver yields not just a trajectory but ``d(loss)/d(parameters)``,
which a static dataset and plain PyTorch cannot. This example harvests those
gradients from Newton, trains a surrogate to reproduce them, and then optimizes a
launch velocity through the cheap surrogate.

The headline is broad BPTT proposal search. A fixed-target surrogate does not
generalize, so this model is conditioned on the requested target and trained on
a distribution of reachable targets. Once trained, it optimizes many launch
starts together through the full learned flight without a simulator in the loop.
Every reported proposal is then scored by a full Newton simulation. The example
measures held-out generalization directly: the real Newton miss after optimizing
each held-out target's launch, and the surrogate-vs-Newton gradient cosine.

The example only declares the physics-specific pieces (how to sample a launch and
a reachable target, the loss, the observation). PhysicsNeMo owns the rest, the
differentiable-rollout bridge and the surrogate training/optimization loop.

Run from the PhysicsNeMo repository root:
    uv run python examples/newton/diffsim/example_diffsim_ball_surrogate_bptt.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ball_problem as ball
import numpy as np
import torch

from physicsnemo.experimental.integrations.newton import (
    is_main_process,
)


def run(args: argparse.Namespace) -> str:
    """Harvest target-conditioned teacher gradients, fit the surrogate, then show
    held-out-target generalization and write the scorecard."""
    torch.manual_seed(
        args.seed
    )  # reproducible surrogate init/optimizer -> stable figures
    env = ball.make_env(args)

    surrogate, _train, held_out, fit, evaluation = ball.build_surrogate(env, args)

    # HELD-OUT GENERALIZATION (the headline): optimize launches for K brand-new
    # reachable targets and score the real Newton miss + gradient cosine on each.
    generalization, per_target, plans = ball.heldout_generalization(
        env, surrogate, held_out, args
    )

    # Pick one held-out target to walk through the scorecard end-to-end: the
    # surrogate descent, the Newton revalidation, and (with the rest) the cosine.
    # The first fixed held-out target is the showcase; do not select it after
    # observing optimizer quality.
    showcase_index = 0
    showcase = per_target[showcase_index]
    showcase_target = np.asarray(showcase["target"], np.float32)
    showcase_plan = plans[showcase_index]
    validation = surrogate.validate_in_newton(
        showcase_plan, ball.newton_loss_for_target(env, showcase_target, args)
    )

    report = _format_report(fit, evaluation, showcase_plan, validation)
    report = _append_generalization(report, generalization)
    if is_main_process():  # write outputs once under DistributedManager
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ball_surrogate_bptt_report.md").write_text(report, encoding="utf-8")
        png = out_dir / "diffsim_ball_scorecard.png"
        plot_report(
            surrogate.history,
            showcase_plan,
            validation,
            generalization,
            png,
        )
    return report


def _format_report(fit: dict, evaluate: dict, optimize: dict, validate: dict) -> str:
    """Render the ball example's training and validation metrics."""
    metrics = {
        **fit,
        **evaluate,
        "initial_task_loss": optimize["initial_task_loss"],
        "best_task_loss": optimize["best_task_loss"],
        "opt_steps": optimize["steps"],
        "opt_ms": optimize.get("opt_ms"),
        **validate,
    }

    def value(key: str) -> str:
        metric = metrics.get(key)
        if isinstance(metric, (int, float, np.floating)) and np.isfinite(metric):
            return f"{float(metric):.4g}"
        return "n/a"

    return (
        "# ball Newton diffsim surrogate (BPTT)\n\n"
        "| metric | value |\n| --- | ---: |\n"
        f"| train samples | {value('samples')} |\n"
        f"| horizon | {value('horizon')} |\n"
        f"| train rollout RMSE | {value('train_rollout_rmse')} |\n"
        f"| train adjoint cosine | {value('train_adjoint_cosine_mean')} |\n"
        f"| held-out rollout RMSE | {value('rollout_rmse')} |\n"
        f"| held-out adjoint cosine (mean/min) | {value('adjoint_cosine_mean')} / {value('adjoint_cosine_min')} |\n"
        f"| winning candidate's surrogate objective (initial -> best) | {value('initial_task_loss')} -> {value('best_task_loss')} |\n"
        f"| Newton squared-distance objective (initial -> surrogate-chosen) | {value('newton_initial_loss')} -> {value('newton_optimized_loss')} |\n"
        f"| Newton objective improved? | {metrics.get('newton_improved', 'n/a')} |\n"
        f"| amortized gradient-throughput ratio "
        f"(Newton teacher -> surrogate batch {value('surrogate_grad_batch_size')}) | "
        f"{value('gradient_eval_throughput_ratio')}x |\n"
        f"| optimization (no simulator in loop) | {value('opt_steps')} steps / {value('opt_ms')} ms |\n"
        f"| teacher ms / sample | {value('teacher_ms_per_sample')} |\n"
    )


def _append_generalization(report: str, generalization: dict) -> str:
    """Append the held-out-target generalization headline to the Markdown report."""
    return (
        report
        + "\n## Held-out target generalization\n\n"
        + "| metric | value |\n| --- | ---: |\n"
        + f"| held-out targets | {generalization['heldout_targets']} |\n"
        + f"| mean real Newton miss distance [m] | {generalization['mean_real_miss']:.4g} |\n"
        + f"| mean gradient cosine | {generalization['mean_cosine']:.4g} |\n"
        + f"| min gradient cosine | {generalization['min_cosine']:.4g} |\n"
    )


def plot_report(train_history, optimize, validate, generalization, path):
    """Scorecard for the target-conditioned ball surrogate.

    Panels: (a) surrogate training loss, (b) optimize-through-surrogate task-loss
    descent for one held-out target, (c) Newton revalidation for that target
    (initial vs surrogate-chosen real objective), (d) held-out gradient cosine
    mean/min across the K brand-new targets."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    green, blue, orange, slate = "#76b900", "#2b6cb0", "#dd6b20", "#4a5568"
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle(
        "Newton ball target-conditioned surrogate (held-out targets)",
        fontsize=14,
        fontweight="bold",
    )

    ax[0, 0].plot(
        [h["epoch"] for h in train_history],
        [h["loss"] for h in train_history],
        color=green,
        lw=2,
    )
    ax[0, 0].set(
        title="Surrogate training", xlabel="epoch", ylabel="loss", yscale="log"
    )

    opt = optimize.get("history", [])
    ax[0, 1].plot(
        [h["step"] for h in opt],
        [h["task_loss"] for h in opt],
        color=blue,
        lw=2,
        marker="o",
        ms=3,
    )
    ax[0, 1].set(
        title=(
            "Pointwise best BPTT candidate "
            f"({optimize.get('starts', 1)} starts, no simulator)"
        ),
        xlabel="step",
        ylabel="minimum terminal objective across starts",
    )

    bars = ax[1, 0].bar(
        ["Newton\ninitial", "Newton\nsurrogate-chosen"],
        [validate["newton_initial_loss"], validate["newton_optimized_loss"]],
        color=[slate, green],
    )
    ax[1, 0].bar_label(bars, fmt="%.3g")
    ax[1, 0].set(
        title="Revalidated in Newton (held-out target)",
        ylabel="squared-distance objective [m²]",
    )

    cos_mean = float(generalization["mean_cosine"])
    cos_min = float(generalization["min_cosine"])
    bars = ax[1, 1].bar(
        ["cosine\nmean", "cosine\nmin"],
        [cos_mean, cos_min],
        color=[green, orange],
    )
    ax[1, 1].bar_label(bars, fmt="%.3g")
    ax[1, 1].axhline(0.0, color=slate, lw=0.8)
    ax[1, 1].set(
        title=f"Held-out gradient alignment (across {generalization['heldout_targets']} targets)",
        ylabel="cosine similarity",
        ylim=(min(0.0, cos_min) - 0.08, 1.05),
    )

    for axis in ax.flat:
        axis.grid(alpha=0.4)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. Defaults are the full-run config so a plain
    invocation reproduces the headline; pass smaller values for a smoke run."""
    parser = argparse.ArgumentParser(
        description="Newton differentiable-ball target-conditioned surrogate (BPTT)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "diffsim",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--opt-steps", type=int, default=120)
    parser.add_argument("--opt-samples", type=int, default=64)
    parser.add_argument("--heldout-targets", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--newton-device", default="cpu")
    parser.add_argument("--torch-device", default="cpu")
    args = parser.parse_args()
    for flag, value in (
        ("--samples", args.samples),
        ("--val-samples", args.val_samples),
        ("--epochs", args.epochs),
        ("--steps", args.steps),
        ("--opt-steps", args.opt_steps),
        ("--opt-samples", args.opt_samples),
        ("--heldout-targets", args.heldout_targets),
    ):
        if value <= 0:
            parser.error(f"{flag} must be positive")
    return args


if __name__ == "__main__":
    print(run(parse_args()))
