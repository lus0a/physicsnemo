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

"""Show how rollout BPTT improves a learned Newton cart-pole model.

Both surrogates see the same Newton trajectories and begin from identical model
weights. The baseline is trained only on teacher-forced one-step transitions.
The BPTT model adds a free-running rollout loss: each predicted state becomes
the next model input, so later errors provide feedback through all earlier
learned steps.

No task objective, solver adjoint, or control optimizer is involved. The example
isolates the training question: does feedback through the model's own predicted
states make its free-running dynamics more accurate?

Run from the PhysicsNeMo repository root:
    uv run python examples/newton/diffsim/example_diffsim_cartpole_bptt.py
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import newton
import torch

from physicsnemo.experimental.integrations.newton import (
    BPTTSurrogate,
    NewtonEnv,
    ResidualDynamics,
    TeacherBatch,
    field_to_torch,
    is_main_process,
    resolve_device,
)

STATE_DIM = 4
JOINT_DOF = 2
FORCE_LIMIT = 1000.0


def make_cartpole_env(world_count: int, device: torch.device) -> NewtonEnv:
    """Build ``world_count`` independent Featherstone cart-poles."""

    if world_count <= 0:
        raise ValueError("world_count must be positive")

    cartpole = newton.ModelBuilder()
    cartpole.default_joint_cfg.armature = 0.01
    cartpole.default_joint_cfg.limit_ke = 1.0e4
    cartpole.default_joint_cfg.limit_kd = 1.0e1
    cartpole.add_urdf(
        str(
            Path(__file__).resolve().parents[1]
            / "nerd"
            / "assets"
            / "cartpole_nerd.urdf"
        ),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )

    builder = newton.ModelBuilder()
    builder.replicate(cartpole, world_count, spacing=(0.0, 2.0, 0.0))
    model = builder.finalize(device=str(device))
    solver = newton.solvers.SolverFeatherstone(model, update_mass_matrix_interval=5)

    def observe(state: Any) -> torch.Tensor:
        q = field_to_torch(state.joint_q).reshape(world_count, JOINT_DOF)
        qd = field_to_torch(state.joint_qd).reshape(world_count, JOINT_DOF)
        return torch.cat((q, qd), dim=-1)

    return NewtonEnv.from_model(
        model,
        solver=solver,
        dt=1.0 / 300.0,
        substeps=5,
        collisions=False,
        observe=observe,
    )


def sample_force_sequences(count: int, horizon: int, *, seed: int) -> torch.Tensor:
    """Draw one bounded cart force per trajectory and frame."""

    if count <= 0 or horizon <= 0:
        raise ValueError("count and horizon must be positive")
    generator = torch.Generator().manual_seed(seed)
    return (2.0 * torch.rand(count, horizon, generator=generator) - 1.0) * FORCE_LIMIT


def collect_trajectories(
    sample_count: int,
    validation_count: int,
    horizon: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[TeacherBatch, TeacherBatch]:
    """Generate train and held-out trajectories in replicated Newton worlds."""

    if sample_count <= 0 or validation_count <= 0:
        raise ValueError("sample_count and validation_count must be positive")
    total = sample_count + validation_count
    generator = torch.Generator().manual_seed(seed)

    q = torch.zeros(total, JOINT_DOF)
    q[:, 0] = 2.0 * torch.rand(total, generator=generator) - 1.0
    q[:, 1] = (2.0 * torch.rand(total, generator=generator) - 1.0) * math.pi
    qd = 2.0 * torch.rand(total, JOINT_DOF, generator=generator) - 1.0
    forces = sample_force_sequences(total, horizon, seed=seed + 1)

    env = make_cartpole_env(total, device)
    env.reset(
        joint_q=q.to(device).reshape(-1),
        joint_qd=qd.to(device).reshape(-1),
    )
    observations = [env.observe().clone()]
    start = time.perf_counter()
    for frame in range(horizon):
        cart_force = forces[:, frame].to(device)

        def apply_force(
            _state: Any,
            control: Any,
            _contacts: Any,
            _dt: float,
            _substep: int,
        ) -> None:
            joint_force = field_to_torch(control.joint_f).reshape(total, JOINT_DOF)
            joint_force.zero_()
            joint_force[:, 0].copy_(cart_force)

        observations.append(env.step(before_substep=apply_force).clone())
    _synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / total

    states = torch.stack(observations, dim=1).detach().cpu()
    metadata = {"teacher_ms_per_sample": elapsed_ms}
    train = TeacherBatch(
        states=states[:sample_count],
        parameters=forces[:sample_count],
        metadata=dict(metadata),
    )
    held_out = TeacherBatch(
        states=states[sample_count:],
        parameters=forces[sample_count:],
        metadata=dict(metadata),
    )
    return train, held_out


def surrogate_inputs(
    forces: torch.Tensor, batch: TeacherBatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the recorded initial state and one cart force per model step."""

    return batch.states[:, 0].to(forces), forces.unsqueeze(-1)


def build_surrogate(
    train: TeacherBatch,
    args: argparse.Namespace,
    device: torch.device,
    *,
    rollout_weight: float,
) -> tuple[BPTTSurrogate, float]:
    """Fit one surrogate from a reproducible initialization."""

    torch.manual_seed(args.model_seed)
    surrogate = BPTTSurrogate(
        state_dim=STATE_DIM,
        param_dim=args.horizon,
        input_dim=1,
        to_inputs=surrogate_inputs,
        model=ResidualDynamics.mlp(
            state_dim=STATE_DIM,
            input_dim=1,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        ),
        device=device,
    )
    _synchronize(device)
    start = time.perf_counter()
    surrogate.fit(
        train,
        epochs=args.epochs,
        rollout_weight=rollout_weight,
        rollout_warmup_epochs=args.rollout_warmup_epochs,
    )
    _synchronize(device)
    return surrogate, time.perf_counter() - start


def run(args: argparse.Namespace) -> str:
    """Train the two models and compare their held-out dynamics errors."""

    device = resolve_device(args.device)
    train, held_out = collect_trajectories(
        args.samples,
        args.validation_samples,
        args.horizon,
        seed=args.data_seed,
        device=device,
    )

    one_step, one_step_seconds = build_surrogate(
        train, args, device, rollout_weight=0.0
    )
    bptt, bptt_seconds = build_surrogate(
        train, args, device, rollout_weight=args.rollout_weight
    )
    results = {
        "horizon": args.horizon,
        "samples": args.samples,
        "one_step_eval": one_step.evaluate(held_out),
        "bptt_eval": bptt.evaluate(held_out),
        "one_step_train_seconds": one_step_seconds,
        "bptt_train_seconds": bptt_seconds,
    }
    report = format_report(results)

    if is_main_process():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "cartpole_bptt_report.md").write_text(
            report, encoding="utf-8"
        )
        plot_results(
            results,
            args.output_dir / "diffsim_cartpole_bptt_error.png",
        )
    return report


def format_report(results: dict[str, Any]) -> str:
    """Format the comparison as a compact Markdown table."""

    one_eval = results["one_step_eval"]
    bptt_eval = results["bptt_eval"]
    return (
        "# Cart-pole rollout-BPTT comparison\n\n"
        "Both models use identical data, architecture, initialization, and "
        "one-step loss. Only the BPTT model adds free-running rollout loss.\n\n"
        "| metric | one-step only | one-step + BPTT |\n"
        "| --- | ---: | ---: |\n"
        f"| held-out one-step RMSE | {one_eval['one_step_rmse']:.4f} | "
        f"{bptt_eval['one_step_rmse']:.4f} |\n"
        f"| held-out rollout RMSE | {one_eval['rollout_rmse']:.4f} | "
        f"{bptt_eval['rollout_rmse']:.4f} |\n"
        f"| held-out terminal RMSE | {one_eval['terminal_rmse']:.4f} | "
        f"{bptt_eval['terminal_rmse']:.4f} |\n"
        f"| training time [s] | {results['one_step_train_seconds']:.3f} | "
        f"{results['bptt_train_seconds']:.3f} |\n"
    )


def plot_results(results: dict[str, Any], path: Path) -> None:
    """Plot local and free-running held-out errors."""

    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    green, gray = "#76b900", "#718096"
    figure, axis = plt.subplots(figsize=(7.6, 4.8))

    horizon = int(results["horizon"])
    labels = ["one-step RMSE", f"{horizon}-step rollout RMSE"]
    baseline = [
        results["one_step_eval"]["one_step_rmse"],
        results["one_step_eval"]["rollout_rmse"],
    ]
    bptt = [
        results["bptt_eval"]["one_step_rmse"],
        results["bptt_eval"]["rollout_rmse"],
    ]
    x = [0.0, 1.0]
    width = 0.34
    baseline_bars = axis.bar(
        [position - width / 2 for position in x],
        baseline,
        width,
        label="one-step trained",
        color=gray,
    )
    bptt_bars = axis.bar(
        [position + width / 2 for position in x],
        bptt,
        width,
        label="rollout BPTT trained",
        color=green,
    )
    axis.bar_label(baseline_bars, fmt="%.3f", padding=3)
    axis.bar_label(bptt_bars, fmt="%.3f", padding=3)
    reduction = baseline[1] / bptt[1]
    axis.set(
        title=f"Similar one-step accuracy, {reduction:.1f}x lower rollout error",
        ylabel="held-out state RMSE (lower is better)",
        xticks=x,
        xticklabels=labels,
    )
    axis.legend(frameon=False, loc="upper left")
    axis.grid(alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _synchronize(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one-step and rollout-BPTT Newton cart-pole surrogates"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "diffsim",
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--rollout-weight", type=float, default=0.005)
    parser.add_argument("--rollout-warmup-epochs", type=int, default=100)
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for name in (
        "samples",
        "validation_samples",
        "horizon",
        "epochs",
        "hidden_dim",
        "depth",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rollout_weight <= 0.0:
        parser.error("--rollout-weight must be positive")
    if args.rollout_warmup_epochs < 0:
        parser.error("--rollout-warmup-epochs must be non-negative")
    return args


if __name__ == "__main__":
    print(run(parse_args()))
