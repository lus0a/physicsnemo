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

"""Pareto-front exploration plot for the nozzle inverse-design (N4).

Reads the per-candidate data the example persists to
``outputs/nozzle/pareto_data.npz`` and plots the explored designs as flow-error
versus stability (cv), colored by active-learning round, with the non-dominated
front highlighted and the chosen best design starred.

The two plotted axes (flow-error and cv) are two of the four terms in the design
objective (coverage and a manufacturability penalty also enter), so the 2D
non-dominated front is a PROJECTION of the full trade-off, not the full Pareto
set.

Run:
    python plot_pareto.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rounds_from_schedule(n: int, bootstrap: int, num_worlds: int) -> np.ndarray:
    """Assign an active-learning round per candidate from the append schedule.

    Round 0 is the Sobol bootstrap of size ``bootstrap``; each later round adds
    ``num_worlds`` guided designs. Robust to oracle batch chunking because it
    buckets by append order, not by evaluate-call count.
    """
    rounds = np.zeros(n, dtype=np.int64)
    edges = [bootstrap] + [num_worlds] * (
        (max(0, n - bootstrap) + num_worlds - 1) // max(1, num_worlds)
    )
    start, r = 0, 0
    for size in edges:
        rounds[start : min(n, start + size)] = r
        start += size
        r += 1
        if start >= n:
            break
    return rounds


def pareto_min_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated set minimizing both ``x`` and ``y``."""
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            mask[i] = False
            continue
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        if np.any(dominated):
            mask[i] = False
    return mask


def plot_pareto(
    flow: np.ndarray,
    cv: np.ndarray,
    coverage: np.ndarray,
    target: float,
    bootstrap: int,
    num_worlds: int,
    best_index: int,
    path: Path,
) -> None:
    """Scatter cv vs flow-error colored by AL round, with the 2D Pareto front."""
    plt.switch_backend("Agg")
    n = len(flow)
    flow_error = np.abs(flow - target) / max(float(target), 1.0e-6)
    rounds = rounds_from_schedule(n, bootstrap, num_worlds)
    sizes = 28.0 + 90.0 * np.clip(coverage, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    fig.suptitle(
        "Nozzle design exploration: flow-error vs stability (Pareto front)",
        fontsize=13,
        fontweight="bold",
    )
    sc = ax.scatter(
        cv,
        flow_error,
        c=rounds,
        s=sizes,
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("active-learning round (0 = Sobol bootstrap)")

    front = pareto_min_mask(cv, flow_error)
    fidx = np.where(front)[0]
    order = fidx[np.argsort(cv[fidx])]
    ax.plot(
        cv[order],
        flow_error[order],
        color="#e53e3e",
        lw=1.6,
        zorder=3,
        label="non-dominated front",
    )
    ax.scatter(
        cv[order],
        flow_error[order],
        s=140,
        facecolors="none",
        edgecolors="#e53e3e",
        linewidth=1.6,
        zorder=4,
    )
    ax.scatter(
        cv[best_index],
        flow_error[best_index],
        marker="*",
        s=320,
        color="#76b900",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="chosen best (full 4-term objective)",
    )
    ax.set(
        xlabel="flow stability cv (lower better)",
        ylabel="flow-error  |flow - target| / target (lower better)",
    )
    ax.grid(alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.text(
        0.5,
        0.015,
        "Marker size = coverage. The 2D front is a projection of the 4-term objective\n"
        "(flow-error, cv, coverage, manufacturability), so the starred best may sit just off it.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4a5568",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(
        f"[plot_pareto] wrote {path} ({n} designs, {int(front.sum())} on the 2D front)"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the nozzle Pareto plot."""
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description="Nozzle Pareto-front exploration plot (N4)")
    p.add_argument(
        "--pareto-data",
        type=Path,
        default=here / "outputs" / "nozzle" / "pareto_data.npz",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=here / "outputs" / "nozzle" / "newton_nozzle_pareto.png",
    )
    return p.parse_args()


def main() -> int:
    """Load saved nozzle evaluations and write the Pareto-front plot."""
    args = parse_args()
    if not Path(args.pareto_data).exists():
        print(
            f"[plot_pareto] {args.pareto_data} not found. Run the example with "
            "--save-pareto-data first, e.g.\n"
            "    uv run python examples/newton/nozzle/example_mpm_nozzle_design.py "
            "--save-pareto-data"
        )
        return 1
    data = np.load(args.pareto_data)
    plot_pareto(
        flow=data["flow_rate_norm"],
        cv=data["flow_stability"],
        coverage=data["flow_coverage"],
        target=float(data["target"]),
        bootstrap=int(data["bootstrap"]),
        num_worlds=int(data["num_worlds"]),
        best_index=int(data["best_index"]),
        path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
