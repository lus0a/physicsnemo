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

"""Point-cloud surrogate conditioned on arbitrary global features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.pointnet.pointnet import PointNetEncoder


@dataclass
class MetaData(ModelMetaData):
    """Optimization and deployment capabilities for :class:`PointNetMLP`."""

    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = False
    onnx: bool = True
    onnx_runtime: bool = True
    func_torch: bool = True
    auto_grad: bool = True


class PointNetMLP(Module):
    r"""Predict global outputs from a point cloud and conditioning vector.

    This composition is useful for geometry-conditioned surrogates: PointNet
    embeds an unordered surface sample, then a PhysicsNeMo MLP combines that
    embedding with any global state, control, material, or design variables.

    Parameters
    ----------
    point_channels : int, optional, default=3
        Number of features :math:`C` on each point.
    global_features : int, optional, default=0
        Width :math:`G` of the per-sample conditioning vector. Set to zero for
        a pure point-cloud model.
    out_features : int, optional, default=1
        Number of predicted global outputs :math:`D_{out}`.
    point_features : int, optional, default=128
        Width of the pooled PointNet embedding.
    point_hidden_channels : Sequence[int], optional, default=(64, 128)
        Widths of the shared point-wise layers.
    hidden_features : int, optional, default=256
        Width of the conditioned MLP.
    hidden_layers : int, optional, default=5
        Number of hidden MLP layers.
    activation_fn : str, optional, default="silu"
        PhysicsNeMo activation name.
    pooling : str, optional, default="max"
        Symmetric point reduction, ``"max"`` or ``"mean"``.

    Forward
    -------
    points : torch.Tensor
        Point cloud of shape :math:`(B, N, C)` where :math:`B` is the batch
        size, :math:`N` is the number of points, and :math:`C` is
        ``point_channels``.
    global_features : torch.Tensor, optional
        Per-sample conditioning vector of shape :math:`(B, G)`. Required when
        the model is configured with ``global_features > 0`` and must be
        omitted (or empty) otherwise.

    Outputs
    -------
    torch.Tensor
        Predicted global outputs of shape :math:`(B, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.pointnet import PointNetMLP
    >>> model = PointNetMLP(point_channels=3, global_features=4, out_features=2)
    >>> points = torch.randn(8, 256, 3)
    >>> global_features = torch.randn(8, 4)
    >>> output = model(points, global_features)
    >>> output.shape
    torch.Size([8, 2])
    """

    def __init__(
        self,
        point_channels: int = 3,
        global_features: int = 0,
        out_features: int = 1,
        *,
        point_features: int = 128,
        point_hidden_channels: Sequence[int] = (64, 128),
        hidden_features: int = 256,
        hidden_layers: int = 5,
        activation_fn: str = "silu",
        pooling: str = "max",
    ) -> None:
        super().__init__(meta=MetaData())
        if point_channels <= 0 or point_features <= 0:
            raise ValueError("point_channels and point_features must be positive")
        if global_features < 0:
            raise ValueError("global_features must be non-negative")
        if out_features <= 0 or hidden_features <= 0 or hidden_layers <= 0:
            raise ValueError(
                "out_features, hidden_features, and hidden_layers must be positive"
            )
        self.point_channels = int(point_channels)
        self.global_features = int(global_features)
        self.out_features = int(out_features)
        self.point_features = int(point_features)

        self.point_encoder = PointNetEncoder(
            in_channels=point_channels,
            out_features=point_features,
            hidden_channels=point_hidden_channels,
            activation_fn=activation_fn,
            pooling=pooling,
        )
        self.head = FullyConnected(
            in_features=point_features + global_features,
            out_features=out_features,
            layer_size=hidden_features,
            num_layers=hidden_layers,
            activation_fn=activation_fn,
            skip_connections=False,
        )

    def forward(
        self,
        points: Float[Tensor, "batch points channels"],
        global_features: Float[Tensor, "batch global_features"] | None = None,
    ) -> Float[Tensor, "batch out_features"]:
        """Return predictions for aligned point clouds and global features."""
        encoded = self.point_encoder(points)
        if self.global_features == 0:
            if global_features is not None and global_features.numel() != 0:
                raise ValueError("this model was configured without global features")
            return self.head(encoded)
        if global_features is None:
            raise ValueError(
                f"expected {self.global_features} global features, got None"
            )
        if not torch.compiler.is_compiling():
            if global_features.ndim != 2:
                raise ValueError(
                    "global_features must have shape (batch, features); "
                    f"got {tuple(global_features.shape)}"
                )
            if global_features.shape != (
                points.shape[0],
                self.global_features,
            ):
                raise ValueError(
                    "expected global_features with shape "
                    f"({points.shape[0]}, {self.global_features}), got "
                    f"{tuple(global_features.shape)}"
                )
        return self.head(torch.cat((encoded, global_features), dim=-1))
