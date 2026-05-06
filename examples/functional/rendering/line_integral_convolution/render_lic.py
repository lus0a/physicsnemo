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

from physicsnemo.nn.functional import line_integral_convolution


def dipole_field(
    grid_size: int, depth_size: int, phase: float, device: torch.device
) -> torch.Tensor:
    """Build a thin 3D dipole vector field for LIC slice rendering."""
    coords = torch.linspace(-2.0, 2.0, grid_size, device=device)
    z_coords = torch.linspace(-0.05, 0.05, depth_size, device=device)
    x, y, z = torch.meshgrid(coords, coords, z_coords, indexing="ij")
    angle = torch.tensor(phase, device=device)
    axis = torch.stack([torch.cos(angle), torch.sin(angle)])
    separation = 0.65
    positive = separation * axis
    negative = -separation * axis

    rx_pos = x - positive[0]
    ry_pos = y - positive[1]
    rz_pos = 0.25 * z
    rx_neg = x - negative[0]
    ry_neg = y - negative[1]
    rz_neg = 0.25 * z

    eps = 0.045
    r_pos = (rx_pos * rx_pos + ry_pos * ry_pos + rz_pos * rz_pos + eps) ** 1.5
    r_neg = (rx_neg * rx_neg + ry_neg * ry_neg + rz_neg * rz_neg + eps) ** 1.5
    return torch.stack(
        [
            rx_pos / r_pos - rx_neg / r_neg,
            ry_pos / r_pos - ry_neg / r_neg,
            rz_pos / r_pos - rz_neg / r_neg,
        ],
        dim=-1,
    )


def seed_pattern(grid_size: int, device: torch.device) -> torch.Tensor:
    """Create the fixed random seed texture advected by LIC."""
    return torch.rand(grid_size, grid_size, 1, device=device)


def jet_colormap(value: np.ndarray) -> np.ndarray:
    """Map normalized scalar values to RGB jet colors."""
    red = np.clip(np.minimum(4.0 * value - 1.5, -4.0 * value + 4.5), 0.0, 1.0)
    green = np.clip(np.minimum(4.0 * value - 0.5, -4.0 * value + 3.5), 0.0, 1.0)
    blue = np.clip(np.minimum(4.0 * value + 0.5, -4.0 * value + 2.5), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def _draw_marker(
    image: np.ndarray, point: np.ndarray, color: np.ndarray, radius: int
) -> None:
    height, width = image.shape[:2]
    cx = int((point[0] + 2.0) * 0.25 * float(width - 1))
    cy = int((point[1] + 2.0) * 0.25 * float(height - 1))
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    image[mask] = color


def lic_to_rgb(
    lic: torch.Tensor, vector_field: torch.Tensor, phase: float
) -> np.ndarray:
    """Convert a LIC slice and vector magnitude to an RGB image."""
    center = lic.shape[2] // 2
    lic_image = lic[:, :, center].detach().cpu().numpy()
    low = float(np.percentile(lic_image, 1.0))
    high = float(np.percentile(lic_image, 99.0))
    lic_image = np.clip((lic_image - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    magnitude = vector_field.norm(dim=-1)[:, :, center].detach().cpu().numpy()
    magnitude = np.log1p(magnitude)
    high = max(float(np.percentile(magnitude, 99.0)), 1.0e-6)
    magnitude = np.clip(magnitude / high, 0.0, 1.0)
    color = jet_colormap(magnitude)
    shade = 0.10 + 0.90 * lic_image
    image = color * shade[..., None]

    border = max(1, image.shape[0] // 64)
    image[:border, :, :] = 1.0
    image[-border:, :, :] = 1.0
    image[:, :border, :] = 1.0
    image[:, -border:, :] = 1.0
    axis = np.array([np.cos(phase), np.sin(phase)], dtype=np.float32)
    separation = 0.65
    marker_radius = max(2, image.shape[0] // 32)
    _draw_marker(image, separation * axis, np.array([1.0, 0.12, 0.08]), marker_radius)
    _draw_marker(image, -separation * axis, np.array([0.08, 0.2, 1.0]), marker_radius)
    return np.rot90(image, k=1)


def main() -> None:
    """Generate a 2D LIC animation from a rotating dipole field."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--depth-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--implementation", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs_line_integral_convolution")
    )
    parser.add_argument("--gif-path", type=Path, default=None)
    parser.add_argument("--gif-duration-ms", type=int, default=80)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.gif_path or args.output_dir / "line_integral_convolution.gif"
    gif_frames: list[Image.Image] = []
    seed = seed_pattern(args.grid_size, device).expand(
        args.grid_size, args.grid_size, args.depth_size
    )

    for frame in range(args.frames):
        phase = 2.0 * np.pi * frame / max(args.frames, 1)
        field = dipole_field(args.grid_size, args.depth_size, phase, device)
        lic = line_integral_convolution(
            field,
            seed,
            step_size=0.65,
            num_steps=52,
            contrast=2.2,
            implementation=args.implementation,
        )
        image = (lic_to_rgb(lic, field, phase) * 255.0).clip(0, 255).astype(np.uint8)
        frame_image = Image.fromarray(image, mode="RGB")
        frame_image.save(args.output_dir / f"lic_{frame:04d}.png")
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
