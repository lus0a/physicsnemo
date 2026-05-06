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

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from physicsnemo.nn.functional import mesh_raycast


def cube_mesh(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a colored cube mesh for the raycast example."""
    vertices = torch.tensor(
        [
            [-0.6, -0.6, -0.6],
            [0.6, -0.6, -0.6],
            [0.6, 0.6, -0.6],
            [-0.6, 0.6, -0.6],
            [-0.6, -0.6, 0.6],
            [0.6, -0.6, 0.6],
            [0.6, 0.6, 0.6],
            [-0.6, 0.6, 0.6],
        ],
        device=device,
    )
    indices = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [3, 0, 4],
            [3, 4, 7],
        ],
        device=device,
        dtype=torch.int32,
    )
    colors = torch.tensor(
        [
            [35, 105, 255],
            [40, 190, 255],
            [60, 230, 160],
            [245, 220, 80],
            [255, 120, 80],
            [230, 80, 170],
            [150, 85, 255],
            [80, 235, 235],
        ],
        device=device,
        dtype=torch.uint8,
    )
    return vertices, indices, colors


def rotate_y(vertices: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate vertices about the vertical axis."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    rotation = torch.stack(
        [
            torch.stack([c, torch.zeros_like(c), s]),
            torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)]),
            torch.stack([-s, torch.zeros_like(c), c]),
        ]
    )
    return vertices @ rotation.T


def main() -> None:
    """Generate an animation with the ``mesh_raycast`` functional."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--implementation", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_mesh_raycast"))
    parser.add_argument("--gif-path", type=Path, default=None)
    parser.add_argument("--gif-duration-ms", type=int, default=80)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.gif_path or args.output_dir / "mesh_raycast.gif"
    gif_frames: list[Image.Image] = []

    vertices, indices, colors = cube_mesh(device)
    eye = torch.tensor([0.0, 0.15, -3.0], device=device)
    look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    light = torch.tensor([-0.4, 0.7, -1.0], device=device)
    background = np.array([0.015, 0.018, 0.024], dtype=np.float32)

    for frame in range(args.frames):
        angle = torch.tensor(
            2.0 * torch.pi * frame / max(args.frames, 1), device=device
        )
        frame_vertices = rotate_y(vertices, angle)
        rgba, _, _ = mesh_raycast(
            frame_vertices,
            indices,
            args.image_size,
            args.image_size,
            eye,
            look_at,
            up,
            40.0,
            vertex_colors=colors,
            light_direction=light,
            implementation=args.implementation,
        )
        image = rgba.detach().clamp(0.0, 1.0).cpu().numpy()
        alpha = image[..., 3:4]
        composite = image[..., :3] * alpha + background * (1.0 - alpha)
        gif_image = (composite * 255.0).clip(0, 255).astype(np.uint8)
        frame_image = Image.fromarray(gif_image, mode="RGB")
        frame_image.save(args.output_dir / f"mesh_{frame:04d}.png")
        gif_frames.append(frame_image)

    if gif_frames:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=args.gif_duration_ms,
            loop=0,
        )


if __name__ == "__main__":
    main()
