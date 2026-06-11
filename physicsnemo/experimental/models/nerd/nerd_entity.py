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

"""Entity-aware causal dynamics model following NeRD.

``NeRDEntityTransformer`` extends NeRD's direct, supervised causal
relative-dynamics objective to structured state. It retains one token per
interacting body, particle, or other entity instead of collapsing the system
into one global vector.

Each frame first mixes all entity tokens with non-causal self-attention, so
articulated, cable, and learned contact effects can propagate across the system.
Each entity then attends causally through its own history. The output is a
per-entity dynamics delta at every token, suitable for teacher-forced training
and autoregressive deployment.

The model is application neutral. Rigid-body delta encoding, normalization,
window sampling, and simulator write-back belong to the integration layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.nn import TimmSelfAttention


@dataclass
class MetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`NeRDEntityTransformer`."""

    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    torch_fx: bool = False
    onnx: bool = False
    func_torch: bool = False
    auto_grad: bool = False


class _AttentionBlock(nn.Module):
    """Pre-norm PhysicsNeMo attention block with a residual MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_size, bias=bias)
        self.attn = TimmSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attn_drop_rate=dropout,
            proj_drop_rate=dropout,
            qkv_bias=bias,
            proj_bias=bias,
        )
        self.norm_mlp = nn.LayerNorm(hidden_size, bias=bias)
        self.mlp = FullyConnected(
            in_features=hidden_size,
            layer_size=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            num_layers=1,
            activation_fn="gelu",
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, tokens: Float[Tensor, "b s d"], *, is_causal: bool
    ) -> Float[Tensor, "b s d"]:
        """Mix one token sequence, optionally with causal attention."""
        tokens = tokens + self.attn(self.norm_attn(tokens), is_causal=is_causal)
        return tokens + self.drop(self.mlp(self.norm_mlp(tokens)))


class NeRDEntityTransformer(Module):
    r"""Entity-aware causal transformer that predicts per-entity dynamics deltas.

    The input is a history of entity features. Application-defined inputs can be
    concatenated to those features before calling the model. Within each frame,
    entity-attention blocks exchange information between bodies.
    Temporal-attention blocks then process each body's history causally. The
    model emits one dynamics delta per entity and frame.

    Parameters
    ----------
    feature_dim : int
        Per-entity input feature width.
    prediction_dim : int
        Per-entity output delta width.
    num_entities : int
        Number of entities in each frame.
    context_frames : int
        Maximum deployed history length. Defaults to 8.
    hidden_size : int
        Token embedding width. Defaults to 192.
    entity_depth : int
        Number of within-frame entity-attention blocks. Defaults to 3.
    temporal_depth : int
        Number of causal temporal-attention blocks. Defaults to 4.
    num_heads : int
        Attention-head count. Defaults to 6.
    mlp_ratio : float
        Residual MLP expansion ratio. Defaults to 4.
    head_hidden : int
        Hidden width of the output head. Defaults to 256.
    head_layers : int
        Hidden-layer count of the output head. Defaults to 2.
    dropout : float
        Dropout rate applied to attention, projections, the residual head, and
        the embedding sum. Defaults to 0.
    bias : bool
        Whether attention projections and LayerNorms carry bias. Defaults to
        true.

    Forward
    -------
    features : torch.Tensor
        Entity-feature history, shape
        :math:`(B, T, N, \text{feature\_dim})` with :math:`T \leq`
        ``context_frames`` and :math:`N =` ``num_entities``.

    Outputs
    -------
    torch.Tensor
        Per-entity, per-frame prediction, shape
        :math:`(B, T, N, \text{prediction\_dim})`.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.experimental.models.nerd import NeRDEntityTransformer
    >>> model = NeRDEntityTransformer(
    ...     feature_dim=5, prediction_dim=4, num_entities=3, context_frames=8
    ... )
    >>> features = torch.randn(2, 8, 3, 5)
    >>> model(features).shape
    torch.Size([2, 8, 3, 4])
    >>> model(features)[:, -1].shape
    torch.Size([2, 3, 4])
    """

    def __init__(
        self,
        feature_dim: int,
        prediction_dim: int,
        num_entities: int,
        context_frames: int = 8,
        hidden_size: int = 192,
        entity_depth: int = 3,
        temporal_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        head_hidden: int = 256,
        head_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize the entity-aware NeRD transformer."""
        super().__init__(meta=MetaData())
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if context_frames <= 0:
            raise ValueError("context_frames must be positive")

        self.feature_dim = feature_dim
        self.prediction_dim = prediction_dim
        self.num_entities = num_entities
        self.context_frames = context_frames
        self.hidden_size = hidden_size

        # Unlike the nanoGPT-port NeRDTransformer, which applies an explicit
        # normal(0, 0.02) embedding/linear init plus scaled-residual projection
        # init, this is a distinct architecture with no nanoGPT reference to stay
        # faithful to. It intentionally relies on PyTorch default init for the
        # embeddings and feature_proj and on FCLayer's xavier_uniform + zero-bias
        # init for the FullyConnected MLP and head layers.
        self.feature_proj = nn.Linear(feature_dim, hidden_size)
        self.entity_embed = nn.Embedding(num_entities, hidden_size)
        self.time_embed = nn.Embedding(context_frames, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.entity_blocks = nn.ModuleList(
            [
                _AttentionBlock(hidden_size, num_heads, mlp_ratio, dropout, bias)
                for _ in range(entity_depth)
            ]
        )
        self.temporal_blocks = nn.ModuleList(
            [
                _AttentionBlock(hidden_size, num_heads, mlp_ratio, dropout, bias)
                for _ in range(temporal_depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, bias=bias)
        # The regression head deliberately uses SiLU rather than the residual
        # blocks' GELU: its smooth, non-saturating shape is well suited to the
        # continuous per-entity delta regression this head emits.
        self.head = FullyConnected(
            in_features=hidden_size,
            layer_size=head_hidden,
            out_features=prediction_dim,
            num_layers=head_layers,
            activation_fn="silu",
        )

    def forward(
        self,
        features: Float[Tensor, "b t n f"],
    ) -> Float[Tensor, "b t n p"]:
        """Predict a dynamics delta for every entity token in the history."""
        if features.ndim != 4:
            raise ValueError(
                "features must have shape [batch, time, entity, feature_dim]"
            )
        batch, time_count, entity_count = features.shape[:3]
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim {self.feature_dim}, got {features.shape[-1]}"
            )
        if time_count == 0:
            raise ValueError("sequence length must be at least 1")
        if time_count > self.context_frames:
            raise ValueError(
                f"sequence length {time_count} exceeds context_frames {self.context_frames}"
            )
        if entity_count != self.num_entities:
            raise ValueError(
                f"expected {self.num_entities} entities, got {entity_count}"
            )

        entity_ids = torch.arange(entity_count, device=features.device)
        time_ids = torch.arange(time_count, device=features.device)
        tokens = self.feature_proj(features)
        tokens = tokens + self.entity_embed(entity_ids).view(1, 1, entity_count, -1)
        tokens = tokens + self.time_embed(time_ids).view(1, time_count, 1, -1)
        tokens = self.drop(tokens)

        # Exchange information among all bodies independently at each frame.
        tokens = tokens.reshape(batch * time_count, entity_count, -1)
        for block in self.entity_blocks:
            tokens = block(tokens, is_causal=False)
        tokens = tokens.reshape(batch, time_count, entity_count, -1)

        # Advance every body's representation causally through the frame history.
        tokens = tokens.permute(0, 2, 1, 3).reshape(
            batch * entity_count, time_count, -1
        )
        for block in self.temporal_blocks:
            tokens = block(tokens, is_causal=True)
        tokens = self.norm(tokens)
        prediction = self.head(tokens)
        return prediction.reshape(
            batch, entity_count, time_count, self.prediction_dim
        ).permute(0, 2, 1, 3)
