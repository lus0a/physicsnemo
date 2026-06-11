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

"""Render the local Newton RJ45 teacher beside a trained NeRD rollout."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rj45_scene
import torch
import warp as wp
from PIL import Image, ImageDraw, ImageFont

from physicsnemo.experimental.integrations.newton import (
    NeRDStepModel,
    TrainedNeRDModel,
    resolve_device,
)
from physicsnemo.experimental.integrations.newton.visualization import (
    capture_frame,
    headless_viewer,
    save_gif,
    stack_horizontal,
)


def make_scene(viewer, device: str) -> rj45_scene.BatchedRJ45Scene:
    """Build the same single-world scene used by RJ45 training."""
    scene = rj45_scene.BatchedRJ45Scene(1, device)
    viewer.set_model(scene.model)
    viewer.set_camera(
        pos=wp.vec3(0.125, float(scene.rest_positions[0, 1]) - 0.025, 0.03),
        pitch=-10.0,
        yaw=180.0,
    )
    return scene


def _set_target(scene: rj45_scene.BatchedRJ45Scene, delta: np.ndarray) -> None:
    target = torch.as_tensor(
        delta,
        dtype=scene.rest_positions.dtype,
        device=scene.rest_positions.device,
    )
    scene.set_targets(scene.rest_positions + target)


def _capture(scene: rj45_scene.BatchedRJ45Scene, viewer, frame: int) -> np.ndarray:
    viewer.begin_frame((frame + 1) * scene.frame_dt)
    viewer.log_state(scene.state_0)
    viewer.end_frame()
    return np.asarray(capture_frame(viewer))


def render_vbd(
    scene: rj45_scene.BatchedRJ45Scene,
    viewer,
    frames: int,
    insertion_distance: float,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Render and time the Newton VBD teacher."""
    images, times, seat_errors = [], [], []
    previous = np.zeros(3, dtype=np.float32)
    for frame in range(frames):
        delta, _ = rj45_scene.insertion_command(
            insertion_distance, frame, frames, previous
        )
        previous = delta
        _set_target(scene, delta)
        start = time.perf_counter()
        scene.step()
        wp.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
        images.append(_capture(scene, viewer, frame))
        seat_errors.append(scene.seat_error_mm())
    return images, times, seat_errors


def render_nerd(
    scene: rj45_scene.BatchedRJ45Scene,
    viewer,
    nerd: NeRDStepModel,
    frames: int,
    insertion_distance: float,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Render NeRD after the shared first VBD frame."""
    images, times, seat_errors = [], [], []
    previous = np.zeros(3, dtype=np.float32)

    delta, _ = rj45_scene.insertion_command(insertion_distance, 0, frames, previous)
    previous = delta
    _set_target(scene, delta)
    scene.step()
    wp.synchronize()
    times.append(float("nan"))
    images.append(_capture(scene, viewer, 0))
    seat_errors.append(scene.seat_error_mm())
    nerd.reset()

    for frame in range(1, frames):
        delta, command = rj45_scene.insertion_command(
            insertion_distance, frame, frames, previous
        )
        previous = delta
        command_tensor = torch.as_tensor(
            command,
            dtype=torch.float32,
            device=nerd.device,
        ).unsqueeze(0)
        start = time.perf_counter()
        nerd.step_with_inputs(
            scene.state_0,
            scene.state_0,
            command_tensor,
            dt=scene.frame_dt,
        )
        wp.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
        images.append(_capture(scene, viewer, frame))
        seat_errors.append(scene.seat_error_mm())
    return images, times, seat_errors


def warm_compiled_nerd(
    trained: TrainedNeRDModel,
    scene: rj45_scene.BatchedRJ45Scene,
    nerd: NeRDStepModel,
    device: str,
) -> None:
    """Compile every causal history length before timing."""
    initial_state = nerd.codec.read(scene.state_0)
    inputs = np.zeros(
        (1, trained.config.context_frames, *trained.external_input_shape),
        dtype=np.float32,
    )
    trained.rollout(initial_state, inputs, device=device)
    wp.synchronize()


def _overlay(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 66), fill=(8, 10, 14, 205))
    draw.text((8, 7), title, fill=(118, 185, 0, 255), font=font)
    for index, line in enumerate(lines):
        draw.text((8, 22 + index * 13), line, fill=(220, 236, 255, 255), font=font)
    return np.asarray(image)


def save_comparison(
    vbd_frames: list[np.ndarray],
    nerd_frames: list[np.ndarray],
    vbd_ms: list[float],
    nerd_ms: list[float],
    vbd_seat_errors: list[float],
    nerd_seat_errors: list[float],
    path: Path,
    fps: int,
) -> None:
    """Compose and save the side-by-side comparison."""
    images: list[Image.Image] = []
    for index, (vbd_frame, nerd_frame) in enumerate(zip(vbd_frames, nerd_frames)):
        teacher_median = float(np.median(vbd_ms[1 : index + 1] or vbd_ms[:1]))
        nerd_samples = nerd_ms[1 : index + 1]
        nerd_median = float(np.median(nerd_samples)) if nerd_samples else None
        left = _overlay(
            vbd_frame,
            "Newton VBD teacher",
            [
                f"median {teacher_median:.2f} ms",
                f"seat error {vbd_seat_errors[index]:.2f} mm",
            ],
        )
        right_lines = (
            [
                "shared VBD initialization",
                f"seat error {nerd_seat_errors[index]:.2f} mm",
            ]
            if nerd_median is None
            else [
                f"median {nerd_median:.2f} ms",
                f"speedup {teacher_median / nerd_median:.1f}x",
                f"seat error {nerd_seat_errors[index]:.2f} mm",
            ]
        )
        right = _overlay(
            nerd_frame,
            "NeRD rigid-body torch.compile",
            right_lines,
        )
        images.append(stack_horizontal([left, right]))
    save_gif(images, path, fps=fps)


def parse_args() -> argparse.Namespace:
    """Parse renderer options."""
    parser = argparse.ArgumentParser(description="Render VBD vs NeRD rigid-body")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "outputs" / "rj45_nerd" / "rj45_nerd_model.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent
        / "outputs"
        / "rj45_nerd"
        / "rj45_vbd_vs_nerd.gif",
    )
    parser.add_argument("--frames", type=int, default=320)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--insertion-distance", type=float, default=0.031)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and render the comparison."""
    args = parse_args()
    device = str(resolve_device(args.device))
    if torch.device(device).type == "cuda":
        torch.set_float32_matmul_precision("high")
    trained = TrainedNeRDModel.load(
        args.checkpoint, device=device
    ).compile_for_inference(device=device)
    if tuple(trained.external_input_shape) != (6,):
        raise ValueError(
            "the RJ45 renderer expects a command-only checkpoint with input "
            f"shape (6,), got {tuple(trained.external_input_shape)}"
        )
    if getattr(trained.codec, "reference_frame", None) is not None:
        raise ValueError("the RJ45 renderer expects a world-frame checkpoint")

    viewer = headless_viewer(args.width, args.height)
    try:
        scene = make_scene(viewer, device)
        vbd_frames, vbd_ms, vbd_seat_errors = render_vbd(
            scene, viewer, args.frames, args.insertion_distance
        )
        scene.reset()
        nerd = trained.as_step_model(newton_model=scene.model, device=device)
        warm_compiled_nerd(trained, scene, nerd, device)
        nerd_frames, nerd_ms, nerd_seat_errors = render_nerd(
            scene, viewer, nerd, args.frames, args.insertion_distance
        )
    finally:
        viewer.close()

    save_comparison(
        vbd_frames,
        nerd_frames,
        vbd_ms,
        nerd_ms,
        vbd_seat_errors,
        nerd_seat_errors,
        args.out,
        args.fps,
    )
    print(
        f"{args.out}\n"
        f"final seat error: VBD={vbd_seat_errors[-1]:.3f} mm, "
        f"NeRD rigid-body={nerd_seat_errors[-1]:.3f} mm"
    )


if __name__ == "__main__":
    main()
