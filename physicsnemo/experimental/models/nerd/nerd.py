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

"""A causal transformer dynamics model (the network behind Neural Robot Dynamics).

Given a length-``context_frames`` window of per-step input tokens (a robot state
plus its actuation and any conditioning), the model runs a causal GPT and emits a
per-token prediction of the one-step dynamics, expressed as a per-DoF delta. It is
application neutral: choose ``input_dim`` (the token width) and ``prediction_dim``
(the delta width) for any setting where a fixed-topology system advances one step
at a time. The model has no notion of joints, quaternions, contacts, normalization,
or any simulator. The delta-to-next-state conversion, the robot-centric framing, and
the input/output normalization belong to the caller (the Newton integration glue).

``forward`` returns every token's prediction, which is what teacher-forced training
consumes (each token predicts its own next step). Autoregressive deployment uses
the final prediction.

The causal blocks are a compact port of nanoGPT: a learned token projection and a
learned positional embedding, pre-norm blocks using
:class:`physicsnemo.nn.TimmSelfAttention` with native causal attention, a final
LayerNorm, and a small two-layer ReLU output head. The nanoGPT-style causal
transformer mirrors the reference NeRD network, so this stays faithful while
reusing PhysicsNeMo's attention layer and :class:`~physicsnemo.core.Module` base
for ``.mdlus`` checkpointing. One deliberate addition sits between ``ln_f`` and
the head: a bias-free ``feature_proj`` linear that over-parameterizes (factors)
the head's input map. Because no nonlinearity separates it from the head's first
linear, it adds no inference-time representational capacity over a single linear,
but it is kept on purpose for its effect on training dynamics and to match the
reference parameter count, so it is not part of stock nanoGPT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.nn import TimmSelfAttention


@dataclass
class MetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`NeRDTransformer`."""

    # A causal flash-attention stack does not reliably TorchScript or CUDA-graph
    # capture, so jit / cuda_graphs / torch_fx are disabled, as in DiT. The
    # single amp flag is enough; ModelMetaData.__post_init__ derives
    # amp_cpu and amp_gpu from it.
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    torch_fx: bool = False
    # Transformer ONNX export is fragile, so ONNX is disabled.
    onnx: bool = False
    # Data-driven dynamics surrogate, not a PDE network.
    func_torch: bool = False
    auto_grad: bool = False


class _Block(nn.Module):
    """Pre-norm transformer block: attention then a GELU MLP, both residual."""

    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = TimmSelfAttention(
            hidden_size=n_embd,
            num_heads=n_head,
            attn_drop_rate=dropout,
            proj_drop_rate=dropout,
            qkv_bias=bias,
            proj_bias=bias,
        )
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Float[Tensor, "b t d"]) -> Float[Tensor, "b t d"]:
        x = x + self.attn(self.ln_1(x), is_causal=True)
        x = x + self.mlp(self.ln_2(x))
        return x


class NeRDTransformer(Module):
    r"""Causal transformer that predicts one dynamics step as a per-DoF delta.

    Consumes a window of per-step input tokens (a robot state embedding
    concatenated with its actuation and any conditioning), runs a causal GPT, and
    predicts the one-step change of the system. ``forward`` returns all tokens for
    teacher-forced training; autoregressive deployment uses the final prediction.
    The model is pure sequence-to-sequence: the delta-to-next-state conversion,
    robot-centric framing, and normalization live in the caller.

    Parameters
    ----------
    input_dim : int
        Per-token input feature width (state embedding + actuation + conditioning).
    prediction_dim : int
        Per-token output width (the per-DoF delta the caller integrates).
    context_frames : int
        History window length :math:`h` the caller feeds at deployment.
        Defaults to 10.
    block_size : int
        Maximum sequence length the positional embedding supports
        (:math:`\geq` ``context_frames``). Defaults to 32.
    n_layer : int
        Number of causal transformer blocks. Defaults to 6.
    n_head : int
        Number of attention heads (must divide ``n_embd``). Defaults to 12.
    n_embd : int
        Token embedding width. Defaults to 192.
    head_hidden : int
        Hidden width of the two-layer output head. Defaults to 64.
    dropout : float
        Dropout probability used in attention, the MLP, and the embedding sum.
        Defaults to 0.
    bias : bool
        Whether Transformer block linear layers and LayerNorms carry a bias. The
        input projection and output MLP retain biases, matching the NeRD reference.
        Defaults to false.

    Forward
    -------
    tokens : torch.Tensor
        Input window, shape :math:`(B, T, \text{input\_dim})` with
        :math:`T \leq` ``block_size``.

    Outputs
    -------
    torch.Tensor
        Per-token prediction, shape :math:`(B, T, \text{prediction\_dim})`.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.experimental.models.nerd import NeRDTransformer
    >>> model = NeRDTransformer(input_dim=5, prediction_dim=4, context_frames=10)
    >>> tokens = torch.randn(8, 10, 5)
    >>> model(tokens).shape
    torch.Size([8, 10, 4])
    >>> model(tokens)[:, -1].shape
    torch.Size([8, 4])
    """

    def __init__(
        self,
        input_dim: int,
        prediction_dim: int,
        context_frames: int = 10,
        block_size: int = 32,
        n_layer: int = 6,
        n_head: int = 12,
        n_embd: int = 192,
        head_hidden: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        """Initialize the causal NeRD transformer."""
        super().__init__(meta=MetaData())
        if block_size < context_frames:
            raise ValueError(
                f"block_size ({block_size}) must be >= context_frames ({context_frames})"
            )
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
            )

        self.input_dim = input_dim
        self.prediction_dim = prediction_dim
        self.context_frames = context_frames
        self.block_size = block_size

        # The reference nanoGPT input projection retains its bias even when the
        # Transformer block bias option is disabled.
        self.token_embed = nn.Linear(input_dim, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_Block(n_embd, n_head, dropout, bias) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd, bias=bias)
        self.feature_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.head = nn.Sequential(
            nn.Linear(n_embd, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, prediction_dim),
        )

        self.apply(self._init_weights)
        # nanoGPT scaled init on both residual projections in every block.
        for block in self.blocks:
            nn.init.normal_(
                block.attn.attn_op.proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layer),
            )
            nn.init.normal_(
                block.mlp[2].weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer)
            )
        # The reference output MLP is separate from GPT and keeps PyTorch's
        # default Linear initialization.
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Float[Tensor, "b t f"]) -> Float[Tensor, "b t p"]:
        """Predict a dynamics delta for every token in the causal history."""
        time_count = tokens.shape[1]
        if time_count == 0:
            raise ValueError("sequence length must be at least 1")
        if time_count > self.block_size:
            raise ValueError(
                f"sequence length {time_count} exceeds block_size {self.block_size}"
            )
        positions = torch.arange(time_count, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(positions).unsqueeze(0)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.feature_proj(self.ln_f(x)))
