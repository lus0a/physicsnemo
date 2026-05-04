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
    scalar_field_to_rgba,
    vector_field_to_rgba,
    volume_render,
)
from physicsnemo.nn.functional.rendering import (
    ScalarFieldToRGBA,
    VectorFieldToRGBA,
    VolumeRender,
)
from test.conftest import requires_module


def _camera(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.0, 0.0, -2.5], device=device),
        torch.tensor([0.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )


@requires_module("warp")
def test_scalar_field_to_rgba_maps_transfer_volume(device: str):
    field = torch.linspace(0.0, 1.0, 16, device=device).reshape(4, 4, 1)
    field = field.expand(4, 4, 4).contiguous()

    rgba_volume = scalar_field_to_rgba(
        field,
        0.0,
        1.0,
        max_opacity=0.5,
        opacity_threshold=0.25,
        implementation="warp",
    )

    assert rgba_volume.shape == (4, 4, 4, 4)
    assert rgba_volume.dtype == torch.uint8
    assert int(rgba_volume[..., 3].min()) == 0
    assert int(rgba_volume[..., 3].max()) <= 128
    assert int(rgba_volume[..., :3].max()) > 0
    assert "warp" in ScalarFieldToRGBA.available_implementations()
    assert "torch" in ScalarFieldToRGBA.available_implementations()


@requires_module("warp")
def test_scalar_field_to_rgba_torch_matches_warp(device: str):
    field = torch.linspace(-0.2, 1.2, 5 * 6 * 7, device=device).reshape(5, 6, 7)

    rgba_warp = scalar_field_to_rgba(
        field,
        0.0,
        1.0,
        max_opacity=0.7,
        opacity_threshold=0.2,
        implementation="warp",
    )
    rgba_torch = scalar_field_to_rgba(
        field,
        0.0,
        1.0,
        max_opacity=0.7,
        opacity_threshold=0.2,
        implementation="torch",
    )

    torch.testing.assert_close(rgba_warp, rgba_torch, atol=1, rtol=0)


@requires_module("warp")
def test_vector_field_to_rgba_uses_magnitude_and_lic(device: str):
    coords = torch.linspace(-1.0, 1.0, 6, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    vector_field = torch.stack([-y, x, 0.25 * torch.ones_like(z)], dim=-1)
    lic_field = torch.ones(6, 6, 6, device=device)

    rgba_volume = vector_field_to_rgba(
        vector_field,
        lic_field,
        0.0,
        1.5,
        max_opacity=0.75,
        lic_threshold=0.25,
        implementation="warp",
    )

    assert rgba_volume.shape == (6, 6, 6, 4)
    assert rgba_volume.dtype == torch.uint8
    assert int(rgba_volume[..., 3].max()) > 0
    assert "warp" in VectorFieldToRGBA.available_implementations()
    assert "torch" in VectorFieldToRGBA.available_implementations()


@requires_module("warp")
def test_vector_field_to_rgba_torch_matches_warp(device: str):
    coords = torch.linspace(-1.0, 1.0, 5, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
    vector_field = torch.stack([-y, x, 0.25 + z.square()], dim=-1)
    lic_field = torch.linspace(0.0, 1.0, 5 * 5 * 5, device=device).reshape(5, 5, 5)

    rgba_warp = vector_field_to_rgba(
        vector_field,
        lic_field,
        0.0,
        1.75,
        max_opacity=0.65,
        lic_threshold=0.3,
        implementation="warp",
    )
    rgba_torch = vector_field_to_rgba(
        vector_field,
        lic_field,
        0.0,
        1.75,
        max_opacity=0.65,
        lic_threshold=0.3,
        implementation="torch",
    )

    torch.testing.assert_close(rgba_warp, rgba_torch, atol=1, rtol=0)


@requires_module("warp")
def test_volume_render_returns_color_and_depth(device: str):
    rgba_volume = torch.zeros(16, 16, 16, 4, device=device, dtype=torch.uint8)
    rgba_volume[5:11, 5:11, 5:11, 0] = 255
    rgba_volume[5:11, 5:11, 5:11, 3] = 128
    bounds_min = torch.tensor([-1.0, -1.0, -1.0], device=device)
    bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device)
    eye, center, up = _camera(device)

    rgba, depth = volume_render(
        rgba_volume,
        25,
        25,
        eye,
        center,
        up,
        35.0,
        bounds_min,
        bounds_max,
        step_size=0.08,
        max_steps=80,
        implementation="warp",
    )

    assert rgba.shape == (25, 25, 4)
    assert depth.shape == (25, 25)
    assert float(rgba[..., 3].sum()) > 0.0
    assert float(rgba[..., 0].max()) > 0.8
    assert torch.isfinite(depth).any()
    assert torch.isinf(depth[0, 0])
    assert "warp" in VolumeRender.available_implementations()


@requires_module("warp")
def test_volume_render_accepts_sequence_camera_inputs(device: str):
    rgba_volume = torch.zeros(8, 8, 8, 4, device=device, dtype=torch.uint8)
    rgba_volume[2:6, 2:6, 2:6, 1] = 255
    rgba_volume[2:6, 2:6, 2:6, 3] = 128

    rgba, depth = volume_render(
        rgba_volume,
        11,
        11,
        [0.0, 0.0, -2.5],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        35.0,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        step_size=0.12,
        max_steps=48,
        implementation="warp",
    )

    assert rgba.shape == (11, 11, 4)
    assert depth.shape == (11, 11)
    assert float(rgba[..., 3].sum()) > 0.0


@requires_module("warp")
def test_transfer_and_volume_make_inputs_forward(device: str):
    for spec in (ScalarFieldToRGBA, VectorFieldToRGBA, VolumeRender):
        label, args, kwargs = next(iter(spec.make_inputs_forward(device)))
        assert isinstance(label, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        output = spec.dispatch(*args, implementation="warp", **kwargs)
        assert output is not None


@requires_module("warp")
def test_transfer_error_handling(device: str):
    field = torch.zeros(4, 4, 4, device=device)
    with pytest.raises(ValueError, match="vmax"):
        scalar_field_to_rgba(field, 1.0, 1.0, implementation="warp")

    vector_field = torch.zeros(4, 4, 4, 2, device=device)
    with pytest.raises(ValueError, match="vector_field"):
        vector_field_to_rgba(
            vector_field,
            torch.zeros(4, 4, 4, device=device),
            0.0,
            1.0,
            implementation="warp",
        )
