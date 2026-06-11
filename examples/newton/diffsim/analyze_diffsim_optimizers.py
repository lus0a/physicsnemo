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

"""Safeguarded BPTT design benchmark for the differentiable Newton ball.

The trained target-conditioned surrogate is unrolled for the full trajectory,
and the terminal miss is backpropagated through every learned step to the launch
velocity. Because this BPTT loop is cheap and batched, it can optimize many launch
basins before spending another real Newton rollout.

The benchmark uses a fixed, independently seeded set of reachable long-range
held-out targets and compares three proposal/refinement methods:

* ``Newton cold start``: refine the nominal launch.
* ``Newton multi-start``: forward-screen the nominal launch plus a few unoptimized
  Sobol starts, then refine the best full-Newton candidate.
* ``BPTT branch``: optimize many Sobol starts together through the frozen
  surrogate, forward-screen the best few, then refine the best full-Newton candidate.

All branches receive the same number of differentiable Newton refinement
rollouts. Screening uses cheaper forward-only Newton rollouts and is reported
separately. The public ``BPTT + Newton`` result is safeguarded: it returns the
better full-Newton result from the cold and BPTT branches, so it cannot regress
relative to cold Newton. The target set is fixed before any optimizer runs and is
selected only by geometric distance from the nominal launch endpoint.

Run from the PhysicsNeMo repository root:
    uv run python examples/newton/diffsim/analyze_diffsim_optimizers.py
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import ball_problem as ball
import matplotlib.pyplot as plt
import numpy as np
import torch

from physicsnemo.experimental.integrations.newton import (
    NewtonEnv,
    is_main_process,
    optimize_field_in_newton,
    optimize_field_in_newton_multistart,
)

GREEN, BLUE, ORANGE, SLATE = "#76b900", "#2b6cb0", "#dd6b20", "#4a5568"


def _screen_candidates_in_newton(score_fn, candidates: np.ndarray) -> dict:
    """Forward-screen candidates in Newton and return the best real candidate."""

    screen_losses = np.asarray([score_fn(candidate) for candidate in candidates])
    return {
        "best_params": candidates[int(np.argmin(screen_losses))],
        "forward_screen_evals": len(candidates),
        "screen_losses": screen_losses.tolist(),
    }


def _best_so_far(history: list[dict]) -> np.ndarray:
    return np.minimum.accumulate([float(record["real_loss"]) for record in history])


def _miss(result: dict) -> float:
    return float(np.sqrt(max(float(result["best_loss"]), 0.0)))


def harvest_solver_rollouts(args: argparse.Namespace) -> int:
    """One-time Newton rollouts used to build train and validation teacher data."""

    return 2 * (args.samples + args.val_samples)


def _runtime_provenance(args: argparse.Namespace) -> dict[str, str | bool]:
    """Return enough runtime metadata to interpret recorded benchmark timings."""

    def package_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unknown"

    if torch.cuda.is_available():
        gpu_names = sorted(
            {
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            }
        )
        gpu = ", ".join(gpu_names)
    else:
        gpu = "none"

    root = Path(__file__).resolve().parents[3]
    git = shutil.which("git")
    try:
        if git is None:
            raise OSError("git executable not found")
        commit = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(  # noqa: S603
                [git, "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            != ""
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda or "none"),
        "newton": package_version("newton"),
        "warp": package_version("warp-lang"),
        "gpu": gpu,
        "newton_device": str(args.newton_device),
        "torch_device": str(args.torch_device),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _select_benchmark_targets(
    env: NewtonEnv,
    args: argparse.Namespace,
    nominal_endpoint: np.ndarray,
) -> tuple[list[np.ndarray], int]:
    """Select the full independently seeded target set before optimization."""

    rng = np.random.default_rng(args.seed + 950_000)
    targets: list[np.ndarray] = []
    attempts = 0
    while len(targets) < args.benchmark_targets:
        target = ball.reachable_target(env, rng, args.steps)
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("could not sample enough long-range benchmark targets")
        if np.linalg.norm(target - nominal_endpoint) >= args.min_nominal_miss:
            targets.append(target)
    return targets, attempts


def _benchmark_target(
    env: NewtonEnv,
    surrogate,
    held_out,
    target: np.ndarray,
    args: argparse.Namespace,
    index: int,
) -> tuple[dict, dict]:
    """Run all proposal/refinement methods for one fixed held-out target."""

    loss_fn = ball.ball_loss_fn(env, target)
    score_fn = ball.newton_loss_for_target(env, target, args)
    batch = ball.single_target_batch(held_out, target)
    initial_candidates = ball.feasible_sobol_launches(
        args.surrogate_starts, seed=args.seed + 960_000 + index
    )
    plan = surrogate.optimize_multistart(
        batch,
        starts=args.surrogate_starts,
        initial_params=initial_candidates,
        steps=args.surrogate_steps,
        lr=args.surrogate_lr,
        seed=args.seed + 960_000 + index,
    )

    order = np.argsort(plan["candidate_best_task_losses"])
    bptt_candidates = np.asarray(plan["candidate_best_params"], np.float32)[
        order[: args.screened_candidates]
    ]
    # The solver-only baseline receives the same feasible initial candidates and
    # screening/refinement structure, but no BPTT optimization or ranking.
    newton_multistart_candidates = initial_candidates[: args.screened_candidates]

    bptt_screen = _screen_candidates_in_newton(score_fn, bptt_candidates)
    newton_multistart_screen = _screen_candidates_in_newton(
        score_fn, newton_multistart_candidates
    )

    # The public multistart helper is the safeguard: fully refine both the
    # nominal cold start and the BPTT proposal, then return the better real run.
    safeguarded = optimize_field_in_newton_multistart(
        env,
        loss_fn=loss_fn,
        initials=np.stack((ball.LAUNCH_MEAN, bptt_screen["best_params"])),
        field="particle_qd",
        optimization_steps=args.refine_steps,
        lr=args.newton_lr,
        steps=args.steps,
    )
    cold_branch, bptt_branch = safeguarded["runs"]
    newton_multistart = optimize_field_in_newton(
        env,
        loss_fn=loss_fn,
        field="particle_qd",
        initial=newton_multistart_screen["best_params"],
        optimization_steps=args.refine_steps,
        lr=args.newton_lr,
        steps=args.steps,
    )
    tape_budget = int(cold_branch["solver_evals"])
    if (
        int(newton_multistart["solver_evals"]) != tape_budget
        or int(bptt_branch["solver_evals"]) != tape_budget
    ):
        raise RuntimeError("refinement branches spent different tape-rollout budgets")

    bptt_miss = _miss(bptt_branch)
    cold_miss = _miss(cold_branch)
    record = {
        "index": index,
        "target": target.tolist(),
        "bptt_branch_miss": bptt_miss,
        "safeguarded_bptt_miss": _miss(safeguarded),
        "safeguard_selected": "bptt" if safeguarded["best_index"] == 1 else "cold",
        "newton_multistart_miss": _miss(newton_multistart),
        "newton_cold_miss": cold_miss,
        "tape_budget": tape_budget,
        "safeguarded_serial_tape_evals": int(safeguarded["solver_evals"]),
        "bptt_screen_evals": int(bptt_screen["forward_screen_evals"]),
        "newton_multistart_screen_evals": int(
            newton_multistart_screen["forward_screen_evals"]
        ),
        "bptt_opt_ms": float(plan["opt_ms"]),
        "newton_tape_ms_per_eval": float(
            cold_branch["total_ms"] / cold_branch["solver_evals"]
        ),
    }
    details = {
        "plan": plan,
        "bptt_branch": bptt_branch,
        "safeguarded": safeguarded,
        "newton_multistart": newton_multistart,
        "newton_cold": cold_branch,
    }
    return record, details


def _summary(records: list[dict], key: str) -> dict:
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    return {
        "mean_miss": float(values.mean()),
        "median_miss": float(np.median(values)),
        "max_miss": float(values.max()),
        "misses": values.tolist(),
    }


def run(args: argparse.Namespace) -> dict:
    """Train once, then run the fixed-target safeguarded BPTT benchmark."""

    torch.manual_seed(args.seed)
    env = ball.make_env(args)
    surrogate, _train, held_out, _fit, evaluation = ball.build_surrogate(env, args)

    nominal_endpoint = ball.newton_final_position(env, ball.LAUNCH_MEAN, args.steps)
    targets, target_generation_rollouts = _select_benchmark_targets(
        env, args, nominal_endpoint
    )
    records: list[dict] = []
    representative = None
    for index, target in enumerate(targets):
        record, details = _benchmark_target(
            env, surrogate, held_out, target, args, index
        )
        record["nominal_miss"] = float(np.linalg.norm(target - nominal_endpoint))
        records.append(record)
        if representative is None:
            representative = details

    bptt = _summary(records, "bptt_branch_miss")
    safeguarded = _summary(records, "safeguarded_bptt_miss")
    multistart = _summary(records, "newton_multistart_miss")
    cold = _summary(records, "newton_cold_miss")
    bptt["wins_vs_cold"] = int(
        np.sum(np.asarray(bptt["misses"]) < np.asarray(cold["misses"]))
    )
    bptt["wins_vs_newton_multistart"] = int(
        np.sum(np.asarray(bptt["misses"]) < np.asarray(multistart["misses"]))
    )
    safeguarded["wins_vs_cold"] = int(
        np.sum(np.asarray(safeguarded["misses"]) < np.asarray(cold["misses"]))
    )
    safeguarded["ties_vs_cold"] = int(
        np.sum(np.asarray(safeguarded["misses"]) == np.asarray(cold["misses"]))
    )
    newton_tape_ms = float(
        np.mean([record["newton_tape_ms_per_eval"] for record in records])
    )
    surrogate_grad_ms = float(evaluation["surrogate_grad_ms_per_sample"])
    surrogate_grad_batch_size = int(evaluation["surrogate_grad_batch_size"])
    bptt_proposal_ms = float(np.mean([record["bptt_opt_ms"] for record in records]))
    serial_newton_proposal_ms = (
        newton_tape_ms * args.surrogate_starts * args.surrogate_steps
    )
    results = {
        "case": "ball",
        "benchmark_targets": int(args.benchmark_targets),
        "tape_budget": int(records[0]["tape_budget"]),
        "safeguarded_serial_tape_evals": int(
            records[0]["safeguarded_serial_tape_evals"]
        ),
        "bptt_screen_evals": int(records[0]["bptt_screen_evals"]),
        "newton_multistart_screen_evals": int(
            records[0]["newton_multistart_screen_evals"]
        ),
        "surrogate_starts": int(args.surrogate_starts),
        "screened_candidates": int(args.screened_candidates),
        "min_nominal_miss": float(args.min_nominal_miss),
        "steps": int(args.steps),
        "newton_tape_ms_per_eval": newton_tape_ms,
        "surrogate_grad_ms_per_candidate": surrogate_grad_ms,
        "surrogate_grad_batch_size": surrogate_grad_batch_size,
        "amortized_gradient_throughput_ratio": newton_tape_ms / surrogate_grad_ms,
        "bptt_proposal_ms": bptt_proposal_ms,
        "serial_newton_proposal_ms_estimate": serial_newton_proposal_ms,
        "proposal_search_speedup_estimate": serial_newton_proposal_ms
        / bptt_proposal_ms,
        "surrogate_steps": int(args.surrogate_steps),
        "harvest_solver_rollouts": harvest_solver_rollouts(args),
        "nominal_endpoint_rollouts": 1,
        "target_generation_rollouts": target_generation_rollouts,
        "provenance": _runtime_provenance(args),
        "records": records,
        "bptt_branch": bptt,
        "safeguarded_bptt": safeguarded,
        "newton_multistart": multistart,
        "newton_cold": cold,
        "_representative": representative,
    }

    if is_main_process():
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        benchmark_png = out_dir / "diffsim_ball_bptt_benchmark.png"
        budget_png = out_dir / "diffsim_ball_safeguarded_bptt.png"
        speedup_png = out_dir / "diffsim_ball_speedup.png"
        plot_bptt_benchmark(results, benchmark_png)
        plot_safeguarded_bptt(results, budget_png)
        plot_speedup(results, speedup_png)
        (out_dir / "ball_bptt_benchmark.md").write_text(
            _format_report(results), encoding="utf-8"
        )
    return results


def plot_bptt_benchmark(results: dict, path: Path) -> None:
    """Show multi-start BPTT descent and held-out safeguarded wins."""

    plt.switch_backend("Agg")
    fig, (ax_bptt, ax_targets) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    fig.suptitle(
        "Newton ball: BPTT searches launch basins before spending solver rollouts",
        fontsize=15,
        fontweight="bold",
    )

    history = results["_representative"]["plan"]["history"]
    steps = [record["step"] for record in history]
    ax_bptt.plot(
        steps,
        [record["mean_task_loss"] for record in history],
        color=BLUE,
        lw=2,
        label=f"mean across {results['surrogate_starts']} starts",
    )
    ax_bptt.plot(
        steps,
        [record["task_loss"] for record in history],
        color=GREEN,
        lw=2.5,
        label="pointwise best candidate",
    )
    ax_bptt.set(
        title=f"BPTT through {results['steps']} learned dynamics steps",
        xlabel="surrogate-gradient step (no Newton rollout)",
        ylabel="surrogate terminal objective [m²]",
        yscale="log",
    )
    ax_bptt.legend(frameon=False)

    target_index = np.arange(results["benchmark_targets"])
    ax_targets.plot(
        target_index,
        results["newton_cold"]["misses"],
        color=SLATE,
        marker="o",
        lw=1.8,
        label="Newton cold start",
    )
    ax_targets.plot(
        target_index,
        results["newton_multistart"]["misses"],
        color=ORANGE,
        marker="^",
        lw=1.5,
        label="Newton multi-start",
    )
    ax_targets.plot(
        target_index,
        results["bptt_branch"]["misses"],
        color=BLUE,
        marker="s",
        lw=1.8,
        label="BPTT branch",
    )
    ax_targets.plot(
        target_index,
        results["safeguarded_bptt"]["misses"],
        color=GREEN,
        marker="D",
        lw=2.6,
        label="safeguarded BPTT + Newton",
    )
    wins = results["bptt_branch"]["wins_vs_cold"]
    ax_targets.text(
        0.03,
        0.96,
        f"BPTT branch wins {wins}/{results['benchmark_targets']}; safeguard never regresses",
        transform=ax_targets.transAxes,
        va="top",
        color=GREEN,
        fontweight="bold",
    )
    ax_targets.set(
        title=(
            f"Long-range held-out targets; each branch gets "
            f"{results['tape_budget']} Newton tape rollouts"
        ),
        xlabel="fixed held-out target index",
        ylabel="final Newton miss distance [m]",
        yscale="log",
    )
    ax_targets.set_xticks(target_index)
    ax_targets.legend(frameon=False)

    for axis in (ax_bptt, ax_targets):
        axis.grid(alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_safeguarded_bptt(results: dict, path: Path) -> None:
    """Show detailed first-target solver curves and aggregate safeguarded miss."""

    plt.switch_backend("Agg")
    fig, (ax_curve, ax_summary) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    fig.suptitle(
        "Newton ball: BPTT branch plus a cold-Newton safeguard",
        fontsize=15,
        fontweight="bold",
    )

    representative = results["_representative"]
    for key, color, marker, label in (
        ("newton_cold", SLATE, "o", "Newton cold start"),
        ("newton_multistart", ORANGE, "^", "Newton multi-start"),
        ("bptt_branch", BLUE, "s", "BPTT branch"),
    ):
        history = representative[key]["history"]
        ax_curve.plot(
            [record["solver_evals"] for record in history],
            _best_so_far(history),
            color=color,
            marker=marker,
            ms=3.5,
            lw=2,
            label=label,
        )
    cold_history = representative["newton_cold"]["history"]
    bptt_history = representative["bptt_branch"]["history"]
    safe_loss = np.minimum(_best_so_far(cold_history), _best_so_far(bptt_history))
    ax_curve.plot(
        [record["solver_evals"] for record in cold_history],
        safe_loss,
        color=GREEN,
        marker="D",
        ms=3.5,
        lw=2.6,
        label="safeguarded BPTT + Newton",
    )
    ax_curve.set(
        title="First fixed held-out target (not selected by result)",
        xlabel="real Newton solver evaluations",
        ylabel="best Newton squared-distance objective [m²]",
        yscale="log",
    )
    ax_curve.legend(frameon=False)

    labels = [
        "Newton\ncold",
        "Newton\nmulti-start",
        "BPTT\nbranch",
        "safeguarded\nBPTT + Newton",
    ]
    means = [
        results["newton_cold"]["mean_miss"],
        results["newton_multistart"]["mean_miss"],
        results["bptt_branch"]["mean_miss"],
        results["safeguarded_bptt"]["mean_miss"],
    ]
    bars = ax_summary.bar(
        np.arange(4), means, color=[SLATE, ORANGE, BLUE, GREEN], width=0.65
    )
    ax_summary.bar_label(bars, fmt="%.3g", padding=4, fontweight="bold")
    ax_summary.set(
        title=f"Mean miss distance across {results['benchmark_targets']} fixed targets",
        ylabel="mean Newton miss distance [m]",
        yscale="log",
    )
    ax_summary.set_xticks(np.arange(4))
    ax_summary.set_xticklabels(labels)

    for axis in (ax_curve, ax_summary):
        axis.grid(alpha=0.35, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.005,
        f"{results['surrogate_starts']} feasible starts are optimized together through "
        f"the surrogate; each refinement branch gets {results['tape_budget']} Newton "
        f"tape rollouts, proposal branches use {results['bptt_screen_evals']} "
        f"forward-only screens, and the serial safeguard spends "
        f"{results['safeguarded_serial_tape_evals']} tape rollouts",
        ha="center",
        fontsize=10.5,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speedup(results: dict, path: Path) -> None:
    """Report gradient throughput, proposal-search speedup, and safeguard cost."""

    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2))
    fig.suptitle(
        "Newton ball: measured batched BPTT throughput and solver cost",
        fontsize=15,
        fontweight="bold",
    )

    gradient_ms = [
        results["newton_tape_ms_per_eval"],
        results["surrogate_grad_ms_per_candidate"],
    ]
    bars = axes[0].bar(
        [
            "Newton tape\nrollout",
            "amortized surrogate gradient\nper candidate",
        ],
        gradient_ms,
        color=[SLATE, GREEN],
        width=0.65,
    )
    axes[0].bar_label(bars, fmt="%.3g ms", padding=4, fontweight="bold")
    axes[0].text(
        0.96,
        0.82,
        f"{results['amortized_gradient_throughput_ratio']:.1f}x throughput ratio",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=GREEN,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    axes[0].set(
        title=(
            "Gradient cost per candidate "
            f"(surrogate batch={results['surrogate_grad_batch_size']})"
        ),
        ylabel="measured/amortized time [ms]",
        yscale="log",
    )

    search_seconds = [
        results["serial_newton_proposal_ms_estimate"] / 1000.0,
        results["bptt_proposal_ms"] / 1000.0,
    ]
    bars = axes[1].bar(
        ["serial Newton tape\nestimate", "batched BPTT\nmeasured"],
        search_seconds,
        color=[SLATE, GREEN],
        width=0.65,
    )
    axes[1].bar_label(bars, fmt="%.3g s", padding=4, fontweight="bold")
    axes[1].text(
        0.96,
        0.82,
        f"{results['proposal_search_speedup_estimate']:.1f}x estimated work ratio",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color=GREEN,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    axes[1].set(
        title=(
            f"{results['surrogate_starts']} starts x "
            f"{results['surrogate_steps']} gradient steps"
        ),
        ylabel="proposal-search wall time [s]",
        yscale="log",
    )

    labels = ["cold\nbranch", "BPTT\nbranch", "safeguarded\nportfolio"]
    tape = np.asarray(
        [
            results["tape_budget"],
            results["tape_budget"],
            results["safeguarded_serial_tape_evals"],
        ]
    )
    screens = np.asarray(
        [0, results["bptt_screen_evals"], results["bptt_screen_evals"]]
    )
    axes[2].bar(labels, tape, color=[SLATE, BLUE, GREEN], label="Newton tape")
    axes[2].bar(
        labels,
        screens,
        bottom=tape,
        color=ORANGE,
        label="forward-only screen",
    )
    axes[2].bar_label(
        axes[2].containers[0],
        labels=[f"{value} tape" for value in tape],
        label_type="center",
        color="white",
        fontweight="bold",
    )
    axes[2].set(
        title="Per-target real-solver work",
        ylabel="Newton rollouts",
    )
    axes[2].legend(frameon=False, fontsize=9)

    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.005,
        "Serial Newton proposal-search time is estimated from mean tape-rollout "
        "latency; BPTT is measured end to end. One-time teacher harvest: "
        f"{results['harvest_solver_rollouts']} Newton rollouts.",
        ha="center",
        fontsize=10.5,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_report(results: dict) -> str:
    bptt = results["bptt_branch"]
    safeguarded = results["safeguarded_bptt"]
    multistart = results["newton_multistart"]
    cold = results["newton_cold"]
    provenance = results["provenance"]
    return (
        "# Ball safeguarded BPTT benchmark\n\n"
        f"The target-conditioned surrogate backpropagates through "
        f"{results['steps']} learned dynamics steps and optimizes "
        f"{results['surrogate_starts']} feasible launch starts in one batch. The best "
        f"{results['screened_candidates']} proposals are screened in Newton and "
        f"the best full-simulation candidate is refined. Every refinement branch receives "
        f"{results['tape_budget']} differentiable Newton refinement rollouts; "
        f"proposal branches additionally spend {results['bptt_screen_evals']} cheaper "
        f"forward-only screens. Targets begin at least "
        f"{results['min_nominal_miss']:.3g} m from the nominal endpoint.\n\n"
        "| method | mean miss [m] | median miss [m] | max miss [m] |\n"
        "| --- | ---: | ---: | ---: |\n"
        f"| Newton cold start | {cold['mean_miss']:.4g} | "
        f"{cold['median_miss']:.4g} | {cold['max_miss']:.4g} |\n"
        f"| Newton multi-start | {multistart['mean_miss']:.4g} | "
        f"{multistart['median_miss']:.4g} | {multistart['max_miss']:.4g} |\n"
        f"| BPTT proposals + Newton | {bptt['mean_miss']:.4g} | "
        f"{bptt['median_miss']:.4g} | {bptt['max_miss']:.4g} |\n"
        f"| safeguarded BPTT + Newton | {safeguarded['mean_miss']:.4g} | "
        f"{safeguarded['median_miss']:.4g} | {safeguarded['max_miss']:.4g} |\n\n"
        f"The BPTT branch wins against cold-start Newton on "
        f"{bptt['wins_vs_cold']}/{results['benchmark_targets']} fixed held-out "
        f"targets and against Newton multi-start on "
        f"{bptt['wins_vs_newton_multistart']}/{results['benchmark_targets']}. "
        f"The safeguarded result is the better full-Newton result from the cold "
        f"and BPTT branches, so it wins on {safeguarded['wins_vs_cold']} targets "
        f"and ties on {safeguarded['ties_vs_cold']}; it cannot be worse. The "
        f"safeguard spends {results['safeguarded_serial_tape_evals']} tape rollouts "
        f"when its two independent branches run serially.\n\n"
        "## Speed and cost\n\n"
        "| measurement | value |\n"
        "| --- | ---: |\n"
        f"| mean Newton tape rollout | {results['newton_tape_ms_per_eval']:.4g} ms |\n"
        f"| amortized surrogate gradient per candidate "
        f"(batch {results['surrogate_grad_batch_size']}) | "
        f"{results['surrogate_grad_ms_per_candidate']:.4g} ms |\n"
        f"| measured amortized gradient-throughput ratio | "
        f"{results['amortized_gradient_throughput_ratio']:.4g}x |\n"
        f"| measured batched BPTT proposal search | "
        f"{results['bptt_proposal_ms'] / 1000.0:.4g} s |\n"
        f"| estimated serial Newton tape proposal search | "
        f"{results['serial_newton_proposal_ms_estimate'] / 1000.0:.4g} s |\n"
        f"| estimated serial-work / measured-BPTT ratio | "
        f"{results['proposal_search_speedup_estimate']:.4g}x |\n"
        f"| cold/BPTT branch tape rollouts | {results['tape_budget']} each |\n"
        f"| safeguarded serial tape rollouts | "
        f"{results['safeguarded_serial_tape_evals']} |\n\n"
        f"One-time teacher harvest: {results['harvest_solver_rollouts']} Newton "
        "rollouts. Benchmark target setup: "
        f"{results['nominal_endpoint_rollouts']} nominal-endpoint rollout plus "
        f"{results['target_generation_rollouts']} reachable-target rollouts.\n\n"
        "## Runtime provenance\n\n"
        "| field | value |\n"
        "| --- | --- |\n"
        f"| GPU | {provenance['gpu']} |\n"
        f"| Newton / Warp | {provenance['newton']} / {provenance['warp']} |\n"
        f"| PyTorch / CUDA runtime | {provenance['torch']} / "
        f"{provenance['cuda_runtime']} |\n"
        f"| Python | {provenance['python']} |\n"
        f"| Newton / Torch device | {provenance['newton_device']} / "
        f"{provenance['torch_device']} |\n"
        f"| Git commit | `{provenance['git_commit']}` "
        f"({'dirty' if provenance['git_dirty'] else 'clean'}) |\n"
        f"| Platform | {provenance['platform']} |\n"
    )


def parse_args() -> argparse.Namespace:
    """Defaults reproduce the headline study; pass smaller values for a smoke run."""

    parser = argparse.ArgumentParser(
        description="Newton ball safeguarded multi-start BPTT benchmark"
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
    parser.add_argument("--benchmark-targets", type=int, default=8)
    parser.add_argument("--min-nominal-miss", type=float, default=1.0)
    parser.add_argument("--surrogate-steps", type=int, default=120)
    parser.add_argument("--surrogate-starts", type=int, default=64)
    parser.add_argument("--surrogate-lr", type=float, default=ball.BPTT_LR)
    parser.add_argument("--screened-candidates", type=int, default=4)
    parser.add_argument("--refine-steps", type=int, default=ball.NEWTON_REFINE_STEPS)
    parser.add_argument("--newton-lr", type=float, default=ball.NEWTON_REFINE_LR)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--newton-device", default="cpu")
    parser.add_argument("--torch-device", default="cpu")
    args = parser.parse_args()
    if args.benchmark_targets <= 0:
        parser.error("--benchmark-targets must be positive")
    if args.min_nominal_miss < 0:
        parser.error("--min-nominal-miss must be non-negative")
    if args.surrogate_starts <= 0:
        parser.error("--surrogate-starts must be positive")
    if not 0 < args.screened_candidates <= args.surrogate_starts:
        parser.error("--screened-candidates must be in [1, --surrogate-starts]")
    if args.refine_steps < 0:
        parser.error("--refine-steps must be non-negative")
    return args


if __name__ == "__main__":
    print(_format_report(run(parse_args())))
