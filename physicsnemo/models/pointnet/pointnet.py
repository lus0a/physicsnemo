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

"""Lightweight permutation-invariant point-cloud encoder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, nn

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.nn import get_activation


@dataclass
class MetaData(ModelMetaData):
    """Optimization and deployment capabilities for :class:`PointNetEncoder`."""

    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = False
    onnx: bool = True
    onnx_runtime: bool = True
    func_torch: bool = True
    auto_grad: bool = True


class PointNetEncoder(Module):
    r"""Encode an unordered point set into one global feature vector.

    A stack of shared point-wise linear maps is followed by symmetric max or
    mean pooling. The configured activation is applied after each hidden layer,
    while the final projection to ``out_features`` is linear; pooling reduces
    over that linear projection. Inputs use the common ``(batch, points,
    channels)`` layout.

    Parameters
    ----------
    in_channels : int, optional, default=3
        Number of features :math:`C` on each point.
    out_features : int, optional, default=128
        Width :math:`D_{out}` of the global point-set embedding.
    hidden_channels : Sequence[int], optional, default=(64, 128)
        Width of each shared hidden layer.
    activation_fn : str, optional, default="silu"
        PhysicsNeMo activation name used after every hidden layer.
    pooling : str, optional, default="max"
        Symmetric reduction, either ``"max"`` or ``"mean"``.

    Forward
    -------
    points : torch.Tensor
        Point cloud of shape :math:`(B, N, C)` where :math:`B` is the batch
        size, :math:`N` is the number of points, and :math:`C` is ``in_channels``.

    Outputs
    -------
    torch.Tensor
        Global point-set embedding of shape :math:`(B, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.pointnet import PointNetEncoder
    >>> model = PointNetEncoder(in_channels=3, out_features=64)
    >>> points = torch.randn(8, 256, 3)
    >>> output = model(points)
    >>> output.shape
    torch.Size([8, 64])
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_features: int = 128,
        hidden_channels: Sequence[int] = (64, 128),
        activation_fn: str = "silu",
        pooling: str = "max",
    ) -> None:
        super().__init__(meta=MetaData())
        if in_channels <= 0 or out_features <= 0:
            raise ValueError("in_channels and out_features must be positive")
        if not hidden_channels or any(width <= 0 for width in hidden_channels):
            raise ValueError("hidden_channels must contain positive widths")
        if pooling not in {"max", "mean"}:
            raise ValueError("pooling must be 'max' or 'mean'")

        self.in_channels = int(in_channels)
        self.out_features = int(out_features)
        self.hidden_channels = tuple(int(width) for width in hidden_channels)
        self.activation_fn = activation_fn
        self.pooling = pooling

        widths = (self.in_channels, *self.hidden_channels, self.out_features)
        self.layers = nn.ModuleList(
            nn.Conv1d(widths[index], widths[index + 1], kernel_size=1)
            for index in range(len(widths) - 1)
        )
        self.activation = get_activation(activation_fn)

    def forward(
        self, points: Float[Tensor, "batch points channels"]
    ) -> Float[Tensor, "batch out_features"]:
        r"""Return global features for a :math:`(B, N, C)` point-cloud tensor."""
        if not torch.compiler.is_compiling():
            if points.ndim != 3:
                raise ValueError(
                    "points must have shape (batch, points, channels); "
                    f"got {tuple(points.shape)}"
                )
            if points.shape[-1] != self.in_channels:
                raise ValueError(
                    f"expected {self.in_channels} point channels, "
                    f"got {points.shape[-1]}"
                )
            if points.shape[1] == 0:
                raise ValueError("point clouds must contain at least one point")

        features = points.transpose(1, 2)
        for layer in self.layers[:-1]:
            features = self.activation(layer(features))
        features = self.layers[-1](features)
        if self.pooling == "max":
            return torch.amax(features, dim=-1)
        return torch.mean(features, dim=-1)
