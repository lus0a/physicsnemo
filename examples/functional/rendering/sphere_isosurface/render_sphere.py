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

from physicsnemo.nn.functional import isosurface_render


def sphere_field(
    grid_n: int,
    center: torch.Tensor,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a signed-distance scalar field for a sphere."""
    coords = torch.linspace(-1.0, 1.0, grid_n, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    dx = x - center[0]
    dy = y - center[1]
    dz = z - center[2]
    return torch.sqrt(dx * dx + dy * dy + dz * dz) - radius


def color_field(grid_n: int, device: torch.device) -> torch.Tensor:
    """Create a uint8 RGB volume used to color the sphere."""
    coords = torch.linspace(0.0, 1.0, grid_n, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    rgb = torch.stack([0.15 + 0.85 * x, 0.25 + 0.55 * y, 0.95 - 0.45 * z], dim=-1)
    return (rgb * 255.0).to(torch.uint8)


def main() -> None:
    """Generate an animation with the ``isosurface_render`` functional."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--implementation", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs_sphere_isosurface")
    )
    parser.add_argument("--gif-path", type=Path, default=None)
    parser.add_argument("--gif-duration-ms", type=int, default=80)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.gif_path or args.output_dir / "sphere_isosurface.gif"
    gif_frames: list[Image.Image] = []

    bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
    eye = torch.tensor([0.0, 0.2, -3.0], device=device)
    look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    light = torch.tensor([-0.45, 0.75, -1.0], device=device)
    background = np.array([0.015, 0.018, 0.024], dtype=np.float32)
    colors = color_field(args.grid_size, device)

    for frame in range(args.frames):
        phase = 2.0 * torch.pi * frame / max(args.frames, 1)
        center = torch.tensor(
            [
                0.35 * torch.cos(torch.tensor(phase)),
                0.18 * torch.sin(torch.tensor(phase)),
                0.0,
            ],
            device=device,
        )
        field = sphere_field(args.grid_size, center, radius=0.42, device=device)
        rgba, _, _ = isosurface_render(
            field,
            args.image_size,
            args.image_size,
            eye,
            look_at,
            up,
            38.0,
            bounds_min,
            bounds_max,
            threshold=0.0,
            step_size=2.0 / args.grid_size,
            max_steps=2 * args.grid_size,
            color_field=colors,
            light_direction=light,
            implementation=args.implementation,
        )

        image = rgba.detach().clamp(0.0, 1.0).cpu().numpy()
        alpha = image[..., 3:4]
        composite = image[..., :3] * alpha + background * (1.0 - alpha)
        gif_image = (composite * 255.0).clip(0, 255).astype(np.uint8)
        frame_image = Image.fromarray(gif_image, mode="RGB")
        frame_image.save(args.output_dir / f"sphere_{frame:04d}.png")
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
