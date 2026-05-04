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

import pytest
import torch

from physicsnemo.nn.functional import (
    line_integral_convolution,
    point_cloud_render,
    wireframe_render,
)
from physicsnemo.nn.functional.rendering import (
    LineIntegralConvolution,
    PointCloudRender,
    WireframeRender,
)
from test.conftest import requires_module


def _camera(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.0, 0.0, -2.0], device=device),
        torch.tensor([0.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )


@requires_module("warp")
def test_line_integral_convolution_returns_bounded_field(device: str):
    coords = torch.linspace(-1.0, 1.0, 8, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    vector_field = torch.stack([-y, x, 0.2 * torch.ones_like(z)], dim=-1)
    seed = torch.linspace(0.0, 1.0, 8, device=device).reshape(8, 1, 1)
    seed = seed.expand(8, 8, 8).contiguous()

    lic = line_integral_convolution(
        vector_field,
        seed,
        step_size=0.4,
        num_steps=4,
        implementation="warp",
    )

    assert lic.shape == (8, 8, 8)
    assert float(lic.min()) >= 0.0
    assert float(lic.max()) <= 1.0
    assert "warp" in LineIntegralConvolution.available_implementations()


@requires_module("warp")
def test_point_cloud_render_returns_color_and_depth(device: str):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.35, 0.2, 0.0]], device=device, dtype=torch.float32
    )
    colors = torch.tensor(
        [[255, 0, 0], [0, 128, 255]], device=device, dtype=torch.uint8
    )
    eye, center, up = _camera(device)

    rgba, depth = point_cloud_render(
        points,
        21,
        21,
        eye,
        center,
        up,
        45.0,
        point_colors=colors,
        point_size=1,
        implementation="warp",
    )

    assert rgba.shape == (21, 21, 4)
    assert depth.shape == (21, 21)
    assert float(rgba[..., 3].sum()) == pytest.approx(2.0)
    assert torch.isfinite(depth).any()
    assert "warp" in PointCloudRender.available_implementations()


@requires_module("warp")
def test_wireframe_render_returns_color_and_depth(device: str):
    edges = torch.tensor(
        [[[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]],
        device=device,
        dtype=torch.float32,
    )
    eye, center, up = _camera(device)

    rgba, depth = wireframe_render(
        edges,
        21,
        21,
        eye,
        center,
        up,
        45.0,
        line_color=torch.tensor([0.8, 0.7, 0.2], device=device),
        implementation="warp",
    )

    assert rgba.shape == (21, 21, 4)
    assert depth.shape == (21, 21)
    assert float(rgba[..., 3].sum()) > 0.0
    assert torch.isfinite(depth).any()
    assert "warp" in WireframeRender.available_implementations()


@requires_module("warp")
def test_lic_and_raster_make_inputs_forward(device: str):
    for spec in (LineIntegralConvolution, PointCloudRender, WireframeRender):
        label, args, kwargs = next(iter(spec.make_inputs_forward(device)))
        assert isinstance(label, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        output = spec.dispatch(*args, implementation="warp", **kwargs)
        assert output is not None


@requires_module("warp")
def test_lic_and_raster_error_handling(device: str):
    vector_field = torch.zeros(4, 4, 4, 3, device=device)
    seed = torch.zeros(4, 4, 4, device=device)
    with pytest.raises(ValueError, match="num_steps"):
        line_integral_convolution(
            vector_field, seed, num_steps=0, implementation="warp"
        )

    eye, center, up = _camera(device)
    with pytest.raises(ValueError, match="point_size"):
        point_cloud_render(
            torch.zeros(1, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            point_size=0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="either point_colors or point_color"):
        point_cloud_render(
            torch.zeros(1, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            point_colors=torch.zeros(1, 3, device=device),
            point_color=torch.ones(3, device=device),
            implementation="warp",
        )

    with pytest.raises(ValueError, match="at least one point"):
        point_cloud_render(
            torch.zeros(0, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="line_thickness"):
        wireframe_render(
            torch.zeros(1, 2, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            line_thickness=0,
            implementation="warp",
        )

    with pytest.raises(ValueError, match="at least one edge"):
        wireframe_render(
            torch.zeros(0, 2, 3, device=device),
            16,
            16,
            eye,
            center,
            up,
            45.0,
            implementation="warp",
        )
