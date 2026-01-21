# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import LayerNorm

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.transformer_decoder import (
    DecoderOnlyLayer,
    TransformerDecoder,
)


@dataclass
class MetaData(ModelMetaData):
    # Keep consistent defaults with other models
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    onnx: bool = False
    func_torch: bool = True
    auto_grad: bool = True


class Sequence_Model(Module):
    r"""Decoder-only multi-head attention temporal model.

    Parameters
    ----------
    input_dim : int
        Number of latent features per token (e.g., :math:`N_{pivotal} \times D_{enc}` flattened or projected).
    input_context_dim : int
        Number of physical context features per token.
    dist : Any
        Distribution or device handle; must provide ``dist.device`` attribute.
    dropout_rate : float, optional, default=0.0
        Dropout probability for the decoder.
    num_layers_decoder : int, optional, default=3
        Number of sub-layers in the transformer decoder.
    num_heads : int, optional, default=8
        Number of attention heads.
    dim_feedforward_scale : int, optional, default=4
        Scale factor for the FFN dimension relative to ``input_dim``.
    num_layers_context_encoder : int, optional, default=2
        MLP layers for context feature encoder.
    num_layers_input_encoder : int, optional, default=2
        MLP layers for input feature encoder.
    num_layers_output_encoder : int, optional, default=2
        MLP layers for output feature encoder.
    activation : str, optional, default="gelu"
        Activation function for transformer blocks. One of ``"relu"``, ``"gelu"``.

    Forward
    -------
    x : torch.Tensor
        Input token sequence of shape :math:`(B, T, D_{in})`.
    context : torch.Tensor, optional
        Optional conditioning tokens of shape :math:`(B, T_c, D_{ctx})` that are
        concatenated before ``x`` along the sequence dimension.

    Outputs
    -------
    torch.Tensor
        Output token sequence of shape :math:`(B, T, D_{in})`.

    Notes
    -----
    Reference: Han, Xu, et al. "Predicting physics in mesh-reduced space with temporal attention."
    arXiv preprint arXiv:2201.09113 (2022).

    See also :class:`~physicsnemo.nn.transformer_decoder.TransformerDecoder`
    and :class:`~physicsnemo.nn.gnn_layers.mesh_graph_mlp.MeshGraphMLP`.
    """

    def __init__(
        self,
        input_dim: int,
        input_context_dim: int,
        dist,
        dropout_rate: float = 0.0000,
        num_layers_decoder: int = 3,
        num_heads: int = 8,
        dim_feedforward_scale: int = 4,
        num_layers_context_encoder: int = 2,
        num_layers_input_encoder: int = 2,
        num_layers_output_encoder: int = 2,
        activation: str = "gelu",
    ):
        super().__init__(meta=MetaData())
        self.dist = dist
        decoder_layer = DecoderOnlyLayer(
            input_dim,
            num_heads,
            dim_feedforward_scale * input_dim,
            dropout_rate,
            activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        decoder_norm = LayerNorm(input_dim, eps=1e-5, bias=True)
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers_decoder, decoder_norm
        )

        self.input_dim = input_dim
        self.input_context_dim = input_context_dim

        self.input_encoder = MeshGraphMLP(
            input_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_input_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.output_encoder = MeshGraphMLP(
            input_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_output_encoder,
            activation_fn=nn.ReLU(),
            norm_type=None,
            recompute_activation=False,
        )
        self.context_encoder = MeshGraphMLP(
            input_context_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_context_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )

    def forward(
        self,
        x: Float[Tensor, "batch time input_dim"],
        context: Optional[Float[Tensor, "batch time_ctx input_context_dim"]] = None,
    ) -> Float[Tensor, "batch time input_dim"]:
        """Run decoder-only transformer with optional context tokens."""
        if not torch.compiler.is_compiling():
            if x.ndim != 3 or x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected tensor of shape (B, T, {self.input_dim}) but got tensor of shape {tuple(x.shape)}"
                )
            if context is not None:
                if context.ndim != 3 or context.shape[-1] != self.input_context_dim:
                    raise ValueError(
                        f"Expected context shape (B, T_c, {self.input_context_dim}) but got tensor of shape {tuple(context.shape)}"
                    )

        if context is not None:
            context = self.context_encoder(context)
            x = torch.cat([context, x], dim=1)

        x = self.input_encoder(x)
        tgt_mask = self.generate_square_subsequent_mask(
            x.size()[1], device=self.dist.device
        )
        output = self.decoder(x, tgt_mask=tgt_mask)
        output = self.output_encoder(output)
        return output[:, 1:]

    @torch.no_grad()
    def sample(
        self,
        z0: Float[Tensor, "batch 1 input_dim"],
        step_size: int,
        context: Optional[Float[Tensor, "batch time_ctx input_context_dim"]] = None,
    ) -> Float[Tensor, "batch time_total input_dim"]:
        r"""Autoregressively sample a sequence starting from ``z0``.

        Parameters
        ----------
        z0 : torch.Tensor
            Initial token(s) of shape :math:`(B, 1, D_{in})`.
        step_size : int
            Number of autoregressive steps to generate.
        context : torch.Tensor, optional
            Optional context tokens of shape :math:`(B, T_c, D_{ctx})`.

        Returns
        -------
        torch.Tensor
            Concatenated sequence including ``z0`` and generated tokens,
            of shape :math:`(B, 1 + T_{gen}, D_{in})`.
        """
        z = z0
        for _ in range(step_size):
            prediction = self.forward(z, context)[:, -1].unsqueeze(1)
            z = torch.concat([z, prediction], dim=1)
        return z

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
    ) -> Tensor:
        r"""Generate a causal mask for an autoregressive decoder.

        Parameters
        ----------
        sz : int
            Sequence length.
        device : torch.device, optional
            Device of the returned mask.
        dtype : torch.dtype, optional
            Dtype of the returned mask.

        Returns
        -------
        torch.Tensor
            Upper-triangular mask of shape :math:`(T, T)` with :math:`-\infty` above the diagonal.
        """
        return torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
