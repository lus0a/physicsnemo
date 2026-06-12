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

"""Inverse-design a Newton MPM nozzle with surrogate-guided simulator queries.

This is the CAE / design-optimization story: a batch of candidate nozzle
geometries is simulated in Newton's (non-differentiable) implicit-MPM solver,
which acts as the oracle in a lightweight PhysicsNeMo design loop. A fast
:class:`~physicsnemo.models.mlp.FullyConnected` surrogate learns the map from a
6-parameter design to a flow-quality objective, and a surrogate-guided query
spends the expensive simulator only on the designs most likely to improve it.

This example treats the implicit-MPM solve as a black-box oracle (contrast
``differentiable_rollout``/``optimize_field_in_newton``, which backprop through
the solver). The physics lives in ``nozzle_scene.py``;
PhysicsNeMo owns the reusable design loop in
``physicsnemo.experimental.integrations.newton.optimize_design``.

Run from the PhysicsNeMo repository root (GPU recommended):
    uv run python example_mpm_nozzle_design.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nozzle_scene as scene
import numpy as np

from physicsnemo.experimental.integrations.newton import (
    DesignResult,
    DesignSurrogate,
    is_main_process,
    optimize_design,
    resolve_device,
)

# Objective substituted for a failed / non-finite design so the records and the
# value handed to the optimizer always agree (see evaluate()).
FAILED_DESIGN_SCORE = 10.0


def make_options(args: argparse.Namespace) -> scene.SimulationOptions:
    """Build :class:`SimulationOptions` from parsed command-line arguments."""
    return scene.SimulationOptions(
        sim_time=args.sim_time,
        fps=args.fps,
        substeps=args.substeps,
        max_iterations=args.max_iterations,
        voxel_size=args.voxel_size,
        target_flow_rate_norm=args.target_flow,
    )


def run(args: argparse.Namespace) -> str:
    """Run the active-learning inverse-design loop and write the nozzle report."""
    bounds = scene.DesignBounds()
    opts = make_options(args)
    # Select the Warp device for the MPM solve. When --newton-device is unset,
    # leave it as None so the solve runs on Warp's default device (GPU when one is
    # visible), preserving the prior behavior; otherwise honor the requested arg
    # (resolve_device also maps it to this rank's device under DistributedManager).
    newton_device = (
        str(resolve_device(args.newton_device))
        if args.newton_device is not None
        else None
    )
    measured: dict[bytes, dict] = {}  # in-loop flow details, keyed by design
    records: list[dict] = []  # append-ordered per-candidate metrics for the Pareto plot

    def evaluate(units: np.ndarray) -> np.ndarray:
        """The oracle: simulate each proposed design in Newton MPM -> objective."""
        designs = np.stack([scene.unit_to_design(u, bounds) for u in units], axis=0)
        metrics, details = scene.evaluate_designs(designs, opts, device=newton_device)
        for i, unit in enumerate(units):
            measured[np.asarray(unit, np.float32).tobytes()] = {
                k: float(v[i]) for k, v in details.items()
            }
            records.append(
                {
                    "unit": np.asarray(unit, np.float32),
                    "flow_rate_norm": float(details["flow_rate_norm"][i]),
                    "flow_stability": float(details["flow_stability"][i]),
                    "flow_coverage": float(details["flow_coverage"][i]),
                    "score": float(
                        np.nan_to_num(
                            metrics[i],
                            nan=FAILED_DESIGN_SCORE,
                            posinf=FAILED_DESIGN_SCORE,
                            neginf=FAILED_DESIGN_SCORE,
                        )
                    ),
                }
            )
        return np.nan_to_num(
            metrics,
            nan=FAILED_DESIGN_SCORE,
            posinf=FAILED_DESIGN_SCORE,
            neginf=FAILED_DESIGN_SCORE,
        )

    surrogate = DesignSurrogate.mlp(
        scene.DESIGN_DIM,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        device=resolve_device(args.torch_device),
    )
    result = optimize_design(
        evaluate,
        design_dim=scene.DESIGN_DIM,
        rounds=args.rounds,
        batch_size=args.num_worlds,
        initial_samples=args.bootstrap,
        surrogate=surrogate,
        surrogate_epochs=args.surrogate_epochs,
        seed=args.seed,
    )

    best_design = scene.unit_to_design(result.best_design, bounds)
    details = measured[np.asarray(result.best_design, np.float32).tobytes()]
    report = format_report(args, result, best_design, details)
    if is_main_process():  # write outputs once under DistributedManager
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "nozzle_design_report.md").write_text(report, encoding="utf-8")
        profile = scene.design_to_profile(best_design)
        profile_z = np.linspace(
            opts.z_offset, opts.z_offset + opts.height, len(profile)
        )
        plot_report(
            result,
            profile,
            profile_z,
            details["flow_rate_norm"],
            args.target_flow,
            Path(args.media_dir or out_dir) / "newton_nozzle_design.png",
        )
        if args.save_pareto_data:
            best_key = np.asarray(result.best_design, np.float32)
            best_index = next(
                (
                    i
                    for i, r in enumerate(records)
                    if np.array_equal(r["unit"], best_key)
                ),
                int(np.argmin([r["score"] for r in records])),
            )
            np.savez(
                out_dir / "pareto_data.npz",
                units=np.stack([r["unit"] for r in records]),
                scores=np.array([r["score"] for r in records], np.float32),
                flow_rate_norm=np.array(
                    [r["flow_rate_norm"] for r in records], np.float32
                ),
                flow_stability=np.array(
                    [r["flow_stability"] for r in records], np.float32
                ),
                flow_coverage=np.array(
                    [r["flow_coverage"] for r in records], np.float32
                ),
                target=np.float32(args.target_flow),
                bootstrap=np.int64(args.bootstrap),
                num_worlds=np.int64(args.num_worlds),
                best_index=np.int64(best_index),
            )
    return report


def plot_report(
    result: DesignResult,
    profile_radii: np.ndarray,
    profile_z: np.ndarray,
    best_flow: float,
    target_flow: float,
    path: Path,
) -> None:
    """Scorecard: AL convergence, surrogate parity, optimized nozzle cross-section, flow vs target."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    green, blue, slate = "#76b900", "#2b6cb0", "#4a5568"
    measured = result.scores
    predicted = result.surrogate.predict(result.normalized_designs)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Newton MPM nozzle inverse-design (active learning)",
        fontsize=14,
        fontweight="bold",
    )

    ax[0, 0].plot(
        range(len(result.history)), result.history, color=green, lw=2, marker="o", ms=4
    )
    ax[0, 0].set(
        title="Active-learning convergence",
        xlabel="round (0 = bootstrap)",
        ylabel="best objective so far",
    )

    lo, hi = (
        float(min(measured.min(), predicted.min())),
        float(max(measured.max(), predicted.max())),
    )
    ax[0, 1].plot([lo, hi], [lo, hi], color=slate, ls="--", lw=1)
    ax[0, 1].scatter(
        measured, predicted, color=blue, s=22, alpha=0.8, edgecolor="white", lw=0.5
    )
    ax[0, 1].set(
        title="Surrogate parity (on evaluated designs)",
        xlabel="Newton-measured objective",
        ylabel="surrogate prediction",
    )

    ax[1, 0].plot(profile_radii, profile_z, color=green, lw=2)
    ax[1, 0].plot(-profile_radii, profile_z, color=green, lw=2)
    ax[1, 0].fill_betweenx(
        profile_z, -profile_radii, profile_radii, color=green, alpha=0.12
    )
    ax[1, 0].set(
        title="Optimized nozzle profile", xlabel="radius [m]", ylabel="height z [m]"
    )
    ax[1, 0].set_aspect("equal", adjustable="datalim")

    bars = ax[1, 1].bar(
        ["best design", "target"], [best_flow, target_flow], color=[green, slate]
    )
    ax[1, 1].bar_label(bars, fmt="%.3g")
    ax[1, 1].set(title="Flow rate vs target", ylabel="normalized mass-flow rate")

    for axis in ax.flat:
        axis.grid(alpha=0.4)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_report(
    args: argparse.Namespace,
    result: DesignResult,
    best_design: np.ndarray,
    details: dict[str, float],
) -> str:
    """Render the Markdown report for the optimized nozzle design."""
    evals = args.bootstrap + args.rounds * args.num_worlds
    design_rows = "\n".join(
        f"| {name} | {value:.4f} |"
        for name, value in zip(scene.DESIGN_NAMES, best_design.tolist())
    )
    rows = [
        ("Newton MPM evaluations (oracle calls)", str(evals)),
        ("active-learning rounds", str(args.rounds)),
        ("target flow rate (normalized)", f"{args.target_flow:.3f}"),
        (
            "best objective (bootstrap -> final)",
            f"{result.history[0]:.4f} -> {result.best_score:.4f}",
        ),
        ("best design flow rate (normalized)", f"{details['flow_rate_norm']:.3f}"),
        (
            "best design flow stability (cv, lower better)",
            f"{details['flow_stability']:.3f}",
        ),
        ("best design coverage", f"{details['flow_coverage']:.3f}"),
    ]
    metric_rows = "\n".join(f"| {label} | {value} |" for label, value in rows)
    history = " -> ".join(f"{h:.3f}" for h in result.history)
    return (
        "## Newton MPM nozzle inverse-design (active learning)\n\n"
        f"| metric | value |\n| --- | ---: |\n{metric_rows}\n\n"
        f"best-so-far objective per round: {history}\n\n"
        f"### Best design\n\n| parameter | value |\n| --- | ---: |\n{design_rows}\n"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the nozzle inverse-design example."""
    parser = argparse.ArgumentParser(
        description="Newton MPM nozzle inverse-design via PhysicsNeMo active learning"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "nozzle"
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        help="directory for newton_nozzle_design.png (default: --output-dir)",
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=6,
        help="designs simulated per Newton batch / AL round",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=18, help="initial Sobol coverage batch size"
    )
    parser.add_argument(
        "--rounds", type=int, default=6, help="surrogate-guided active-learning rounds"
    )
    parser.add_argument("--surrogate-epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--target-flow",
        type=float,
        default=0.75,
        help="desired normalized mass-flow rate",
    )
    parser.add_argument("--sim-time", type=float, default=1.25)
    parser.add_argument("--fps", type=float, default=90.0)
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=220)
    parser.add_argument("--voxel-size", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch-device",
        default="cpu",
        help="device for the tiny surrogate",
    )
    parser.add_argument(
        "--newton-device",
        default=None,
        help="Warp device for the MPM solve (default: Warp's default device)",
    )
    parser.add_argument(
        "--save-pareto-data",
        action="store_true",
        help="also write outputs/nozzle/pareto_data.npz for the Pareto/render scripts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
