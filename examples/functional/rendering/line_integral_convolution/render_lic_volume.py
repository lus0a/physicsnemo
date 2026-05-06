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

from physicsnemo.nn.functional import (
    line_integral_convolution,
    volume_render,
    wireframe_render,
)


def dipole_field(grid_size: int, device: torch.device) -> torch.Tensor:
    """Build a steady 3D dipole vector field for volume LIC."""
    coords = torch.linspace(-1.0, 1.0, grid_size, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    positive = torch.tensor([0.42, 0.0, 0.0], device=device)
    negative = torch.tensor([-0.42, 0.0, 0.0], device=device)

    r_pos = torch.stack([x - positive[0], y - positive[1], z - positive[2]], dim=-1)
    r_neg = torch.stack([x - negative[0], y - negative[1], z - negative[2]], dim=-1)
    eps = 0.035
    d_pos = (r_pos.square().sum(dim=-1, keepdim=True) + eps).pow(1.5)
    d_neg = (r_neg.square().sum(dim=-1, keepdim=True) + eps).pow(1.5)
    return r_pos / d_pos - r_neg / d_neg


def cube_edges(device: torch.device) -> torch.Tensor:
    """Return the line segments for a unit context cube."""
    vertices = torch.tensor(
        [
            [-1.05, -1.05, -1.05],
            [1.05, -1.05, -1.05],
            [1.05, 1.05, -1.05],
            [-1.05, 1.05, -1.05],
            [-1.05, -1.05, 1.05],
            [1.05, -1.05, 1.05],
            [1.05, 1.05, 1.05],
            [-1.05, 1.05, 1.05],
        ],
        device=device,
    )
    edge_indices = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        device=device,
    )
    return vertices[edge_indices]


def rotate_edges(edges: torch.Tensor, phase: float) -> torch.Tensor:
    """Rotate context-cube edges around two axes."""
    angle = torch.tensor(phase, device=edges.device, dtype=edges.dtype)
    c = torch.cos(angle)
    s = torch.sin(angle)
    rotation_y = torch.stack(
        [
            torch.stack([c, torch.zeros_like(c), s]),
            torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)]),
            torch.stack([-s, torch.zeros_like(c), c]),
        ]
    )
    half = 0.45 * angle
    ch = torch.cos(half)
    sh = torch.sin(half)
    rotation_x = torch.stack(
        [
            torch.stack(
                [torch.ones_like(ch), torch.zeros_like(ch), torch.zeros_like(ch)]
            ),
            torch.stack([torch.zeros_like(ch), ch, -sh]),
            torch.stack([torch.zeros_like(ch), sh, ch]),
        ]
    )
    return edges @ (rotation_y @ rotation_x).T


def jet_colormap(value: torch.Tensor) -> torch.Tensor:
    """Map normalized tensor values to RGB jet colors."""
    red = torch.minimum(4.0 * value - 1.5, -4.0 * value + 4.5).clamp(0.0, 1.0)
    green = torch.minimum(4.0 * value - 0.5, -4.0 * value + 3.5).clamp(0.0, 1.0)
    blue = torch.minimum(4.0 * value + 0.5, -4.0 * value + 2.5).clamp(0.0, 1.0)
    return torch.stack([red, green, blue], dim=-1)


def make_lic_rgba_volume(
    vector_field: torch.Tensor,
    lic: torch.Tensor,
    max_opacity: float,
) -> torch.Tensor:
    """Convert vector magnitude and LIC values into a uint8 RGBA volume."""
    magnitude = torch.log1p(vector_field.norm(dim=-1))
    magnitude = (magnitude / torch.quantile(magnitude.reshape(-1), 0.985)).clamp(
        0.0, 1.0
    )

    lic_low = torch.quantile(lic.reshape(-1), 0.01)
    lic_high = torch.quantile(lic.reshape(-1), 0.99)
    lic_norm = ((lic - lic_low) / (lic_high - lic_low).clamp_min(1.0e-6)).clamp(
        0.0, 1.0
    )

    color = jet_colormap(magnitude)
    color = color * (0.18 + 0.82 * lic_norm[..., None])
    alpha = (0.02 + 0.98 * lic_norm) * magnitude.sqrt() * max_opacity
    volume = torch.cat([color, alpha[..., None].clamp(0.0, 1.0)], dim=-1)
    return (volume * 255.0).clamp(0, 255).to(torch.uint8)


def composite_rgba(rgba: torch.Tensor, background: np.ndarray) -> np.ndarray:
    """Composite a rendered RGBA image over a background color."""
    image = rgba.detach().clamp(0.0, 1.0).cpu().numpy()
    alpha = image[..., 3:4]
    return image[..., :3] * alpha + background * (1.0 - alpha)


def overlay_wire(volume_rgb: np.ndarray, wire_rgba: torch.Tensor) -> np.ndarray:
    """Alpha composite a wireframe render over a volume RGB image."""
    wire = wire_rgba.detach().clamp(0.0, 1.0).cpu().numpy()
    alpha = wire[..., 3:4]
    return wire[..., :3] * alpha + volume_rgb * (1.0 - alpha)


def main() -> None:
    """Generate a 3D LIC volume-render animation with a rotating cube."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=56)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--sweep-degrees", type=float, default=90.0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--implementation", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs_line_integral_volume")
    )
    parser.add_argument("--gif-path", type=Path, default=None)
    parser.add_argument("--gif-duration-ms", type=int, default=80)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.gif_path or args.output_dir / "line_integral_convolution_3d.gif"
    gif_frames: list[Image.Image] = []

    vector_field = dipole_field(args.grid_size, device)
    seed = torch.rand(args.grid_size, args.grid_size, args.grid_size, device=device)
    lic = line_integral_convolution(
        vector_field,
        seed,
        step_size=0.55,
        num_steps=26,
        contrast=2.0,
        implementation=args.implementation,
    )
    lic_volume = make_lic_rgba_volume(vector_field, lic, max_opacity=0.16)

    center = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
    edges = cube_edges(device)
    background = np.array([0.015, 0.018, 0.024], dtype=np.float32)

    sweep = np.deg2rad(args.sweep_degrees)
    for frame in range(args.frames):
        t = 0.5 if args.frames == 1 else frame / (args.frames - 1)
        phase = (t - 0.5) * sweep
        eye = torch.tensor(
            [3.1 * np.sin(phase), 0.16, -3.1 * np.cos(phase)],
            device=device,
            dtype=torch.float32,
        )
        volume_rgba, _ = volume_render(
            lic_volume,
            args.image_size,
            args.image_size,
            eye,
            center,
            up,
            42.0,
            bounds_min,
            bounds_max,
            step_size=2.0 / args.grid_size,
            max_steps=2 * args.grid_size,
            opacity_threshold=0.97,
            implementation=args.implementation,
        )
        volume_rgb = composite_rgba(volume_rgba, background)
        wire_rgba, _ = wireframe_render(
            rotate_edges(edges, phase),
            args.image_size,
            args.image_size,
            eye,
            center,
            up,
            42.0,
            line_color=torch.tensor([1.0, 0.92, 0.18], device=device),
            line_thickness=2,
            implementation=args.implementation,
        )
        image = (
            (overlay_wire(volume_rgb, wire_rgba) * 255.0).clip(0, 255).astype(np.uint8)
        )
        frame_image = Image.fromarray(image, mode="RGB")
        frame_image.save(args.output_dir / f"lic_volume_{frame:04d}.png")
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
