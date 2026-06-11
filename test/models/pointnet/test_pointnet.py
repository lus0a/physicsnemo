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

from physicsnemo.models.pointnet import PointNetEncoder, PointNetMLP
from test import common


def _pointnet_case(device, model_type):
    points = torch.randn(2, 11, 3, device=device)
    if model_type == "encoder":
        return (
            PointNetEncoder(
                in_channels=3,
                out_features=8,
                hidden_channels=(8,),
            ).to(device),
            (points,),
        )
    return (
        PointNetMLP(
            global_features=2,
            point_features=8,
            point_hidden_channels=(8,),
            hidden_features=8,
            hidden_layers=2,
        ).to(device),
        (points, torch.randn(2, 2, device=device)),
    )


def test_pointnet_forward_and_permutation_invariance(device):
    torch.manual_seed(3)
    model = PointNetEncoder(
        in_channels=4,
        out_features=24,
        hidden_channels=(16, 32),
    ).to(device)
    points = torch.randn(5, 37, 4, device=device)
    permutation = torch.randperm(points.shape[1], device=device)

    encoded = model(points)
    permuted = model(points[:, permutation])

    assert encoded.shape == (5, 24)
    torch.testing.assert_close(encoded, permuted)


@pytest.mark.parametrize("pooling", ["max", "mean"])
def test_pointnet_pooling_backpropagates(device, pooling):
    model = PointNetEncoder(
        in_channels=3,
        out_features=8,
        hidden_channels=(12,),
        pooling=pooling,
    ).to(device)
    points = torch.randn(2, 11, 3, device=device, requires_grad=True)
    model(points).square().mean().backward()

    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


def test_pointnet_validates_inputs():
    with pytest.raises(ValueError, match="hidden_channels"):
        PointNetEncoder(hidden_channels=())
    with pytest.raises(ValueError, match="pooling"):
        PointNetEncoder(pooling="sum")

    model = PointNetEncoder(in_channels=3)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="expected 3"):
        model(torch.zeros(2, 8, 4))


def test_pointnet_checkpoint(device):
    model_1 = PointNetEncoder(
        in_channels=4,
        out_features=12,
        hidden_channels=(8, 16),
        pooling="mean",
    ).to(device)
    model_2 = PointNetEncoder(
        in_channels=4,
        out_features=12,
        hidden_channels=(8, 16),
        pooling="mean",
    ).to(device)
    points = torch.randn(3, 19, 4, device=device)

    assert common.validate_checkpoint(model_1, model_2, (points,))


def test_pointnet_metadata_matches_supported_transforms():
    assert PointNetEncoder().meta.jit
    assert not PointNetEncoder().meta.torch_fx
    assert PointNetMLP().meta.jit
    assert not PointNetMLP().meta.torch_fx


@pytest.mark.parametrize("model_type", ["encoder", "conditioned"])
def test_pointnet_optimizations(device, model_type):
    for validator in (
        common.validate_cuda_graphs,
        common.validate_jit,
        common.validate_amp,
        common.validate_combo_optims,
    ):
        model, inputs = _pointnet_case(device, model_type)
        assert validator(model, inputs)


@pytest.mark.parametrize("model_type", ["encoder", "conditioned"])
def test_pointnet_onnx_export(device, model_type, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model, inputs = _pointnet_case(device, model_type)
    assert common.validate_onnx_export(model, inputs)


@common.check_ort_version()
@pytest.mark.parametrize("model_type", ["encoder", "conditioned"])
def test_pointnet_onnx_runtime(device, model_type):
    model, inputs = _pointnet_case(device, model_type)
    assert common.validate_onnx_runtime(model, inputs)


def test_pointnet_mlp_conditions_on_global_features(device):
    model = PointNetMLP(
        point_channels=3,
        global_features=5,
        out_features=4,
        point_features=12,
        point_hidden_channels=(8, 12),
        hidden_features=16,
        hidden_layers=2,
    ).to(device)
    points = torch.randn(3, 19, 3, device=device, requires_grad=True)
    context = torch.randn(3, 5, device=device, requires_grad=True)

    output = model(points, context)
    output.square().mean().backward()

    assert output.shape == (3, 4)
    assert points.grad is not None
    assert context.grad is not None


def test_pointnet_mlp_validates_context(device):
    model = PointNetMLP(global_features=2).to(device)
    points = torch.zeros(3, 8, 3, device=device)

    with pytest.raises(ValueError, match="got None"):
        model(points)
    with pytest.raises(ValueError, match="expected global_features"):
        model(points, torch.zeros(3, 3, device=device))


def test_pointnet_mlp_checkpoint(device):
    kwargs = {
        "point_channels": 3,
        "global_features": 4,
        "out_features": 2,
        "point_features": 10,
        "point_hidden_channels": (8, 10),
        "hidden_features": 12,
        "hidden_layers": 2,
    }
    model_1 = PointNetMLP(**kwargs).to(device)
    model_2 = PointNetMLP(**kwargs).to(device)
    points = torch.randn(3, 13, 3, device=device)
    context = torch.randn(3, 4, device=device)

    assert common.validate_checkpoint(model_1, model_2, (points, context))
