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

"""Train NeRD to replace the RJ45 scene's VBD contact solve.

The cable, plug, and latch are represented as entity tokens. A
PhysicsNeMo :class:`~physicsnemo.experimental.models.nerd.NeRDEntityTransformer` exchanges
information among bodies within each frame, then advances every body's history
with causal attention. It predicts normalized relative rigid-body dynamics
directly, including quaternion changes as rotation-vector deltas.

The example owns only the scene and inputs sweep. The reusable one-stage
trainer, DDP synchronization, body-state codec, runtime, and evaluator use the
same :mod:`physicsnemo.experimental.integrations.newton.nerd` workflow as other Newton runs.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rj45_scene
import torch
import warp as wp

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.integrations.newton import (
    NeRDTrainingConfig,
    TrainedNeRDModel,
    is_main_process,
    resolve_device,
)
from physicsnemo.experimental.integrations.newton.nerd import (
    NeRDBodyStateCodec,
    NeRDDataset,
    train_nerd,
)


def _rank_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def make_scene(world_count: int, device: str) -> rj45_scene.BatchedRJ45Scene:
    """Build a replicated RJ45 VBD scene on this rank's device."""
    return rj45_scene.BatchedRJ45Scene(world_count, device)


def _local_count(global_count: int, rank: int, world_size: int) -> int:
    base, remainder = divmod(global_count, world_size)
    return base + int(rank < remainder)


def _local_dataset_counts(args: argparse.Namespace) -> tuple[int, int]:
    """Return this rank's train and validation trajectory counts."""
    rank, world_size = _rank_world_size()
    minimum_count = 2 * world_size
    if args.trajectories < minimum_count:
        raise ValueError(
            f"trajectories ({args.trajectories}) must be at least twice the "
            f"distributed world size ({world_size}) so every rank receives both "
            "training and held-out trajectories"
        )
    max_validation_count = args.trajectories - world_size
    validation_count = max(
        world_size,
        int(round(args.trajectories * args.val_fraction)),
    )
    validation_count = min(validation_count, max_validation_count)
    training_count = args.trajectories - validation_count
    return (
        _local_count(training_count, rank, world_size),
        _local_count(validation_count, rank, world_size),
    )


def collect_teacher(
    args: argparse.Namespace,
    device: str,
) -> tuple[Any, NeRDBodyStateCodec, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect the serially defined trajectory set in parallel Newton worlds."""
    rank, world_size = _rank_world_size()
    local_training_count, local_validation_count = _local_dataset_counts(args)
    trajectory_count = local_training_count + local_validation_count
    rng = np.random.default_rng(args.seed + rank * 1_000_003)
    world_batch_size = min(args.world_batch_size, trajectory_count)
    parameters = []
    # Preserve the original serial RNG sequence; batching changes execution only.
    for _ in range(trajectory_count):
        insertion_focused = rng.random() < args.insertion_focus_fraction
        if insertion_focused:
            x_amplitude = rng.uniform(-args.focus_xy_jitter, args.focus_xy_jitter)
            z_amplitude = rng.uniform(-args.focus_z_jitter, args.focus_z_jitter)
            insertion = rng.uniform(args.focus_insert_min, args.focus_insert_max)
        else:
            x_amplitude = rng.uniform(-args.xy_jitter, args.xy_jitter)
            z_amplitude = rng.uniform(-args.z_jitter, args.z_jitter)
            insertion = rng.uniform(args.insert_min, args.insert_max)
        parameters.append(
            (
                insertion_focused,
                x_amplitude,
                z_amplitude,
                insertion,
                rng.uniform(0.0, 2.0 * np.pi),
            )
        )

    all_states, all_commands, all_step_ms = [], [], []
    scene = make_scene(world_batch_size, device)
    codec = NeRDBodyStateCodec(scene.model)
    collected = 0

    while collected < trajectory_count:
        if collected:
            scene.reset()
        take = min(world_batch_size, trajectory_count - collected)
        batch_parameters = parameters[collected : collected + take]
        batch_parameters.extend([batch_parameters[-1]] * (world_batch_size - take))
        insertion_focused = np.asarray(
            [values[0] for values in batch_parameters], dtype=bool
        )
        x_amplitude = np.asarray([values[1] for values in batch_parameters])
        z_amplitude = np.asarray([values[2] for values in batch_parameters])
        insertion = np.asarray([values[3] for values in batch_parameters])
        phase_shift = np.asarray([values[4] for values in batch_parameters])
        states = torch.empty(
            (world_batch_size, args.frames, *codec.state_shape),
            dtype=torch.float32,
            device=device,
        )
        commands = torch.empty(
            (world_batch_size, args.frames, 6),
            dtype=torch.float32,
            device=device,
        )
        step_ms = torch.empty(
            (world_batch_size, args.frames),
            dtype=torch.float64,
            device=device,
        )
        previous_delta = np.zeros((world_batch_size, 3), dtype=np.float32)

        for frame in range(args.frames):
            phase = frame / max(args.frames - 1, 1)
            progress = rj45_scene.smoothstep(min(phase * 1.25, 1.0))
            wobble = np.sin(2.0 * np.pi * phase + phase_shift)
            alignment = np.where(
                insertion_focused,
                1.0 - rj45_scene.smoothstep(min(phase * 4.0, 1.0)),
                1.0,
            )
            delta = np.stack(
                (
                    x_amplitude * wobble * alignment,
                    insertion * progress,
                    z_amplitude * np.sin(np.pi * phase + 0.5 * phase_shift) * alignment,
                ),
                axis=-1,
            ).astype(np.float32)
            delta_tensor = torch.as_tensor(delta, device=device)
            scene.set_targets(scene.rest_positions + delta_tensor)
            commands[:, frame].copy_(
                torch.as_tensor(
                    np.concatenate((delta, delta - previous_delta), axis=-1),
                    device=device,
                )
            )
            start = time.perf_counter()
            scene.step()
            wp.synchronize()
            step_ms[:, frame] = (time.perf_counter() - start) * 1000.0
            states[:, frame].copy_(codec.read(scene.state_0))
            previous_delta = delta

        all_states.append(states[:take])
        all_commands.append(commands[:take])
        all_step_ms.append(step_ms[:take])
        collected += take
        if rank == 0:
            print(
                f"teacher trajectories {collected}/{trajectory_count} on rank 0 "
                f"({world_batch_size} Newton worlds per frame)"
            )

    return (
        scene,
        codec,
        torch.cat(all_states),
        torch.cat(all_commands),
        torch.cat(all_step_ms),
    )


def _config(args: argparse.Namespace) -> NeRDTrainingConfig:
    return NeRDTrainingConfig(
        context_frames=args.context_frames,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
    )


def _model_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        hidden_size=args.hidden_size,
        entity_depth=args.entity_depth,
        temporal_depth=args.temporal_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
    )


@dataclass
class RJ45Evaluation:
    """Fully autoregressive accuracy and synchronized deployment latency."""

    body_rmse_mm: float
    tracked_body_rmse_mm: float
    final_body_rmse_mm: float
    teacher_step_ms: float
    nerd_step_ms: float
    rollout_frames: int
    held_out_trajectories: int
    finite_fraction: float
    rmse_by_frame: np.ndarray

    @property
    def speedup(self) -> float:
        """Return the synchronized VBD-to-NeRD batched-frame latency ratio."""
        return self.teacher_step_ms / max(self.nerd_step_ms, 1.0e-9)


def _benchmark_live_nerd(
    trained: TrainedNeRDModel,
    scene: Any,
    states: torch.Tensor,
    inputs: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    """Measure the live command-conditioned adapter in many-world batches."""
    step_model = trained.as_step_model(
        newton_model=scene.model,
        device=device,
    )
    step_ms = []
    world_batch_size = step_model.codec.batch_size

    def padded_batch(
        tensor: torch.Tensor, start_index: int
    ) -> tuple[torch.Tensor, int]:
        batch = tensor[start_index : start_index + world_batch_size]
        valid_count = batch.shape[0]
        if valid_count < world_batch_size:
            batch = torch.cat(
                (
                    batch,
                    batch[:1].expand(world_batch_size - valid_count, *batch.shape[1:]),
                )
            )
        return batch, valid_count

    warm_states, _ = padded_batch(states[:, 0], 0)
    warm_inputs, _ = padded_batch(inputs, 0)
    step_model.codec.write(scene.state_0, warm_states)
    step_model.reset()
    for frame in range(warm_inputs.shape[1]):
        step_model.step_with_inputs(
            scene.state_0,
            scene.state_0,
            warm_inputs[:, frame],
            dt=scene.frame_dt,
        )
    wp.synchronize()

    for start_index in range(0, states.shape[0], world_batch_size):
        state_batch, _ = padded_batch(states[:, 0], start_index)
        input_batch, _ = padded_batch(inputs, start_index)
        step_model.codec.write(scene.state_0, state_batch)
        wp.synchronize()
        step_model.reset()
        for frame in range(inputs.shape[1]):
            start = time.perf_counter()
            step_model.step_with_inputs(
                scene.state_0,
                scene.state_0,
                input_batch[:, frame],
                dt=scene.frame_dt,
            )
            wp.synchronize()
            step_ms.append((time.perf_counter() - start) * 1000.0)

    return torch.tensor(
        [float(np.sum(step_ms)), len(step_ms)],
        dtype=torch.float64,
        device=device,
    )


def evaluate_rj45(
    trained: TrainedNeRDModel,
    scene: Any,
    states: torch.Tensor,
    inputs: torch.Tensor,
    teacher_step_ms: torch.Tensor,
    *,
    tracked_body: int,
    device: str,
) -> RJ45Evaluation:
    """Evaluate local held-out shards and reduce metrics through PhysicsNeMo DDP."""
    frame_count, body_count = states.shape[1:3]
    prediction = trained.rollout(
        states[:, 0],
        inputs,
        device=device,
    )
    step_time = _benchmark_live_nerd(
        trained,
        scene,
        states,
        inputs,
        device=device,
    )
    finite = torch.isfinite(prediction).flatten(start_dim=1).all(dim=1)
    finite_count = finite.sum(dtype=torch.int64)
    if bool(finite.all()):
        error = prediction[:, 1:, :, :3] - states[:, 1:, :, :3]
        square = error.square()
        frame_square = square.sum(dim=(0, 2, 3))
        total_square = square.sum()
        tracked_square = square[:, :, tracked_body].sum()
    else:
        frame_square = torch.full((frame_count - 1,), float("inf"), device=device)
        total_square = torch.full((), float("inf"), device=device)
        tracked_square = torch.full((), float("inf"), device=device)

    teacher_samples = teacher_step_ms[:, 1:]
    teacher_time = torch.stack(
        (teacher_samples.sum(), teacher_samples.new_tensor(teacher_samples.numel()))
    )
    counts = torch.tensor(
        [states.shape[0], states.shape[0] * (frame_count - 1)],
        dtype=torch.int64,
        device=device,
    )
    if DistributedManager().world_size > 1:
        group = DistributedManager().group()
        for value in (
            frame_square,
            total_square,
            tracked_square,
            finite_count,
            teacher_time,
            step_time,
            counts,
        ):
            torch.distributed.all_reduce(value, group=group)
    trajectory_count, step_count = (int(value) for value in counts)
    rmse_by_frame = torch.sqrt(frame_square / max(trajectory_count * body_count * 3, 1))
    return RJ45Evaluation(
        body_rmse_mm=float(
            torch.sqrt(total_square / max(step_count * body_count * 3, 1))
        )
        * 1000.0,
        tracked_body_rmse_mm=float(torch.sqrt(tracked_square / max(step_count * 3, 1)))
        * 1000.0,
        final_body_rmse_mm=float(rmse_by_frame[-1]) * 1000.0,
        teacher_step_ms=float(teacher_time[0] / teacher_time[1].clamp_min(1.0)),
        nerd_step_ms=float(step_time[0] / step_time[1].clamp_min(1.0)),
        rollout_frames=frame_count - 1,
        held_out_trajectories=trajectory_count,
        finite_fraction=int(finite_count) / max(trajectory_count, 1),
        rmse_by_frame=rmse_by_frame.cpu().numpy() * 1000.0,
    )


def format_report(evaluation: RJ45Evaluation, trained: TrainedNeRDModel) -> str:
    """Render the RJ45 result and shared NeRD workflow metadata."""
    history = trained.metadata
    rows = [
        ("model", history.get("model_class", "unknown")),
        ("inference", history.get("inference_compile_mode", "eager")),
        ("training trajectories", f"{history.get('trajectory_count', 0):,}"),
        ("optimizer updates", f"{history.get('optimizer_updates', 0):,}"),
        (
            "parallel Newton worlds / frame",
            str(history.get("newton_world_batch_size", 1)),
        ),
        ("held-out trajectories", str(evaluation.held_out_trajectories)),
        ("free-running frames / trajectory", str(evaluation.rollout_frames)),
        ("body-position RMSE vs VBD (mm)", f"{evaluation.body_rmse_mm:.4f}"),
        ("plug-position RMSE vs VBD (mm)", f"{evaluation.tracked_body_rmse_mm:.4f}"),
        ("finite held-out trajectories", f"{evaluation.finite_fraction * 100:.0f}%"),
        ("VBD batched frame (ms)", f"{evaluation.teacher_step_ms:.2f}"),
        ("NeRD live batched frame (ms)", f"{evaluation.nerd_step_ms:.2f}"),
        ("synchronized batched-frame speedup", f"{evaluation.speedup:.1f}x"),
    ]
    table = "\n".join(f"| {label} | {value} |" for label, value in rows)
    return (
        "## Newton RJ45: entity-aware NeRD dynamics\n\n"
        "The model uses world-frame rigid-body state, the commanded plug motion, "
        "and the PhysicsNeMo entity transformer. Latency is measured for the "
        "reported number of parallel Newton worlds.\n\n"
        f"| metric | value |\n| --- | ---: |\n{table}\n"
    )


def run(args: argparse.Namespace) -> str:
    """Collect VBD data, fit rigid-body NeRD, evaluate it, and write outputs."""
    DistributedManager.initialize()
    device = str(resolve_device(args.device))
    if torch.device(device).type == "cuda":
        torch.set_float32_matmul_precision("high")
    rank, _ = _rank_world_size()
    log = print if rank == 0 else (lambda _message: None)
    trained = None
    if args.load_checkpoint is not None:
        trained = TrainedNeRDModel.load(args.load_checkpoint, device=device)
        if tuple(trained.external_input_shape) != (6,):
            raise ValueError(
                "the RJ45 example expects a command-only checkpoint with input "
                f"shape (6,), got {tuple(trained.external_input_shape)}"
            )
        if getattr(trained.codec, "reference_frame", None) is not None:
            raise ValueError(
                "the RJ45 example expects a world-frame body-state checkpoint"
            )
        log(f"loaded checkpoint {args.load_checkpoint}")

    scene, codec, states, commands, teacher_step_ms = collect_teacher(args, device)

    local_training_count, _ = _local_dataset_counts(args)
    train_states, held_states = (
        states[:local_training_count],
        states[local_training_count:],
    )
    train_commands, held_commands = (
        commands[:local_training_count],
        commands[local_training_count:],
    )
    held_teacher_step_ms = teacher_step_ms[local_training_count:]
    entity_weights = torch.ones(states.shape[2], dtype=torch.float32, device=device)
    plug = scene.plug_body_index
    latch = scene.latch_body_index
    entity_weights[plug] = args.connector_loss_weight
    entity_weights[latch] = args.connector_loss_weight
    loss_weights = entity_weights[:, None].expand(*codec.prediction_shape)
    if trained is None:
        trained = train_nerd(
            NeRDDataset(
                states=train_states,
                inputs=train_commands[:, 1:],
                codec=codec,
                frame_dt=scene.frame_dt,
            ),
            _config(args),
            dynamics_model="NeRDEntityTransformer",
            model_kwargs=_model_kwargs(args),
            loss_weights=loss_weights,
            device=device,
            seed=args.seed,
            log=log,
        )

    trained.metadata["max_entity_weight"] = float(entity_weights.max())
    trained.metadata["newton_world_batch_size"] = scene.world_count
    output = Path(args.output_dir)
    if is_main_process():
        output.mkdir(parents=True, exist_ok=True)
        if args.save_checkpoint:
            checkpoint = output / "rj45_nerd_model.pt"
            temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
            trained.save(temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint)
    inference = (
        trained.compile_for_inference(device=device) if args.compile_model else trained
    )
    evaluation = evaluate_rj45(
        inference,
        scene,
        held_states,
        held_commands[:, 1:],
        held_teacher_step_ms,
        tracked_body=plug,
        device=device,
    )
    if not is_main_process():
        return ""
    report = format_report(evaluation, inference)
    (output / "rj45_nerd_report.md").write_text(report, encoding="utf-8")
    scorecard = output / "rj45_nerd.png"
    plot_report(evaluation, inference.metadata, scorecard)
    return report


def plot_report(evaluation, history, path: Path) -> None:
    """Plot free-running accuracy, direct-dynamics training, and step cost."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    green, blue, slate = "#76b900", "#2b6cb0", "#4a5568"
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    figure.suptitle(
        "Newton RJ45: entity-aware NeRD rigid-body dynamics",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.91,
        f"Trained checkpoint | {history.get('optimizer_updates', 0):,} optimizer updates | "
        f"inference {history.get('inference_compile_mode', 'eager')} | "
        f"distributed world size {history.get('distributed_world_size', 1)} | "
        f"{history.get('newton_world_batch_size', 1)} Newton worlds / frame | "
        f"{history.get('max_entity_weight', 1.0):g}x max entity loss weight | "
        f"{evaluation.rollout_frames}-step frame-zero free-running evaluation",
        ha="center",
        fontsize=9,
        color="#4a5568",
    )

    rmse = evaluation.rmse_by_frame
    axes[0].plot(range(1, len(rmse) + 1), rmse, color=green, lw=2)
    axes[0].set(
        title="Fully autoregressive accuracy",
        xlabel="free-running rollout step",
        ylabel="body-position RMSE [mm]",
    )

    curve = history.get("train_curve", [])
    axes[1].plot(range(1, len(curve) + 1), curve, color=blue, lw=2)
    axes[1].set(
        title="Direct relative-dynamics training",
        xlabel="epoch",
        ylabel="normalized delta MSE",
        yscale="log",
    )

    inference_mode = history.get("inference_compile_mode", "eager")
    nerd_label = (
        "NeRD rigid-body\ntrained checkpoint"
        if inference_mode == "eager"
        else "NeRD rigid-body\ntorch.compile"
    )
    bars = axes[2].bar(
        ["VBD teacher", nerd_label],
        [evaluation.teacher_step_ms, evaluation.nerd_step_ms],
        color=[slate, green],
    )
    axes[2].bar_label(bars, fmt="%.2f")
    axes[2].set(
        title="Synchronized batched-frame latency",
        ylabel=(f"ms / {history.get('newton_world_batch_size', 1)}-world frame"),
        yscale="log",
    )
    axes[2].text(
        0.5,
        0.92,
        f"{evaluation.speedup:.1f}x speedup\nbody RMSE {evaluation.body_rmse_mm:.2f} mm",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontweight="bold",
        bbox=dict(boxstyle="round", fc="#f7fafc", ec="#d7dce2"),
    )
    for axis in axes:
        axis.grid(alpha=0.4)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _apply_smoke(args: argparse.Namespace) -> argparse.Namespace:
    args.trajectories = 4
    args.frames = 40
    args.context_frames = 4
    args.hidden_size = 32
    args.entity_depth = 1
    args.temporal_depth = 1
    args.num_heads = 4
    args.head_hidden = 64
    args.epochs = 3
    args.steps_per_epoch = 20
    args.batch_size = 32
    return args


def _apply_full(args: argparse.Namespace) -> argparse.Namespace:
    args.trajectories = 64
    args.frames = 320
    args.context_frames = 8
    args.hidden_size = 192
    args.entity_depth = 3
    args.temporal_depth = 4
    args.num_heads = 6
    args.head_hidden = 256
    args.epochs = 200
    args.steps_per_epoch = 1000
    args.batch_size = 128
    return args


def parse_args() -> argparse.Namespace:
    """Parse cable rigid-body NeRD example options."""
    parser = argparse.ArgumentParser(
        description="Newton RJ45 entity-aware NeRD learned rigid-body dynamics"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "rj45_nerd",
    )
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument(
        "--world-batch-size",
        type=int,
        default=8,
        help="independent RJ45 worlds advanced by each 60 Hz frame",
    )
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--insert-min", type=float, default=0.0)
    parser.add_argument("--insert-max", type=float, default=0.042)
    parser.add_argument("--xy-jitter", type=float, default=0.006)
    parser.add_argument("--z-jitter", type=float, default=0.005)
    parser.add_argument("--insertion-focus-fraction", type=float, default=0.75)
    parser.add_argument("--focus-insert-min", type=float, default=0.027)
    parser.add_argument("--focus-insert-max", type=float, default=0.031)
    parser.add_argument("--focus-xy-jitter", type=float, default=0.003)
    parser.add_argument("--focus-z-jitter", type=float, default=0.002)
    parser.add_argument("--connector-loss-weight", type=float, default=8.0)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--entity-depth", type=int, default=2)
    parser.add_argument("--temporal-depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--head-hidden", type=int, default=192)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--steps-per-epoch",
        "--iters-per-epoch",
        dest="steps_per_epoch",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="global batch size divided across distributed ranks",
    )
    parser.add_argument("--lr-start", type=float, default=5.0e-4)
    parser.add_argument("--lr-end", type=float, default=5.0e-5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--device",
        help="Newton/model device; defaults to the active rank or PyTorch default",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="load a trained NeRD checkpoint and regenerate held-out reports",
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the trained or loaded checkpoint for lower inference latency",
    )
    parser.add_argument("--save-checkpoint", action="store_true")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--smoke", action="store_true")
    preset.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.world_batch_size <= 0:
        parser.error("--world-batch-size must be positive")
    if args.smoke:
        return _apply_smoke(args)
    return _apply_full(args) if args.full else args


if __name__ == "__main__":
    result = run(parse_args())
    if result:
        print(result)
