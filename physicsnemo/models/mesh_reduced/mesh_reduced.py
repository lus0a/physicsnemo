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

import importlib

import torch

from physicsnemo.core.version_check import require_version_spec
from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNet
from physicsnemo.nn.module.gnn_layers.graph_types import GraphType  # noqa


class Mesh_Reduced(torch.nn.Module):
        r"""PbGMR-GMUS architecture.

        A mesh-reduced architecture that combines encoding and decoding processors
        for physics prediction in reduced mesh space.

        Parameters
        ----------
        input_dim_nodes : int
            Number of node features.
        input_dim_edges : int
            Number of edge features.
        output_decode_dim : int
            Number of decoding outputs (per node).
        output_encode_dim : int, optional, default=3
            Number of encoding outputs (per pivotal position).
        processor_size : int, optional, default=15
            Number of message passing blocks.
        num_layers_node_processor : int, optional, default=2
            Number of MLP layers for processing nodes in each message passing block.
        num_layers_edge_processor : int, optional, default=2
            Number of MLP layers for processing edge features in each message passing block.
        hidden_dim_processor : int, optional, default=128
            Hidden layer size for the message passing blocks.
        hidden_dim_node_encoder : int, optional, default=128
            Hidden layer size for the node feature encoder.
        num_layers_node_encoder : int, optional, default=2
            Number of MLP layers for the node feature encoder.
        hidden_dim_edge_encoder : int, optional, default=128
            Hidden layer size for the edge feature encoder.
        num_layers_edge_encoder : int, optional, default=2
            Number of MLP layers for the edge feature encoder.
        hidden_dim_node_decoder : int, optional, default=128
            Hidden layer size for the node feature decoder.
        num_layers_node_decoder : int, optional, default=2
            Number of MLP layers for the node feature decoder.
        k : int, optional, default=3
            Number of nearest neighbors used for pivotal/mesh interpolation.
        aggregation : str, optional, default="mean"
            Message aggregation type.

        Notes
        -----
        Reference: `Predicting physics in mesh-reduced space with temporal attention <https://arxiv.org/pdf/2201.09113>`
        """

        def __init__(
            self,
            input_dim_nodes: int,
            input_dim_edges: int,
            output_decode_dim: int,
            output_encode_dim: int = 3,
            processor_size: int = 15,
            num_layers_node_processor: int = 2,
            num_layers_edge_processor: int = 2,
            hidden_dim_processor: int = 128,
            hidden_dim_node_encoder: int = 128,
            num_layers_node_encoder: int = 2,
            hidden_dim_edge_encoder: int = 128,
            num_layers_edge_encoder: int = 2,
            hidden_dim_node_decoder: int = 128,
            num_layers_node_decoder: int = 2,
            k: int = 3,
            aggregation: str = "mean",
        ):
            super(Mesh_Reduced, self).__init__()
            self.knn_encoder_already = False
            self.knn_decoder_already = False
            self.encoder_processor = MeshGraphNet(
                input_dim_nodes,
                input_dim_edges,
                output_encode_dim,
                processor_size,
                "relu",
                num_layers_node_processor,
                num_layers_edge_processor,
                hidden_dim_processor,
                hidden_dim_node_encoder,
                num_layers_node_encoder,
                hidden_dim_edge_encoder,
                num_layers_edge_encoder,
                hidden_dim_node_decoder,
                num_layers_node_decoder,
                aggregation,
            )
            self.decoder_processor = MeshGraphNet(
                output_encode_dim,
                input_dim_edges,
                output_decode_dim,
                processor_size,
                "relu",
                num_layers_node_processor,
                num_layers_edge_processor,
                hidden_dim_processor,
                hidden_dim_node_encoder,
                num_layers_node_encoder,
                hidden_dim_edge_encoder,
                num_layers_edge_encoder,
                hidden_dim_node_decoder,
                num_layers_node_decoder,
                aggregation,
            )
            self.k = k
            self.PivotalNorm = torch.nn.LayerNorm(output_encode_dim)

        @require_version_spec("torch_cluster")
        @require_version_spec("torch_scatter")
        def knn_interpolate(
            self,
            x: torch.Tensor,
            pos_x: torch.Tensor,
            pos_y: torch.Tensor,
            batch_x: torch.Tensor = None,
            batch_y: torch.Tensor = None,
            k: int = 3,
            num_workers: int = 1,
        ):
            r"""Perform k-nearest neighbor interpolation.

            Parameters
            ----------
            x : torch.Tensor
                Source features of shape :math:`(N_x, D_x)`.
            pos_x : torch.Tensor
                Source positions of shape :math:`(N_x, D_{pos})`.
            pos_y : torch.Tensor
                Target positions of shape :math:`(N_y, D_{pos})`.
            batch_x : torch.Tensor, optional
                Batch indices for source positions of shape :math:`(N_x,)`, by default ``None``.
            batch_y : torch.Tensor, optional
                Batch indices for target positions of shape :math:`(N_y,)`, by default ``None``.
            k : int, optional, default=3
                Number of nearest neighbors to consider.
            num_workers : int, optional, default=1
                Number of workers for parallel processing.

            Returns
            -------
            torch.Tensor
                Interpolated features of shape :math:`(N_y, D_x)`.
            torch.Tensor
                Source indices.
            torch.Tensor
                Target indices.
            torch.Tensor
                Interpolation weights.
            """
            with torch.no_grad():
                torch_cluster = importlib.import_module("torch_cluster")
                assign_index = torch_cluster.knn(
                    pos_x,
                    pos_y,
                    k,
                    batch_x=batch_x,
                    batch_y=batch_y,
                    num_workers=num_workers,
                )
                y_idx, x_idx = assign_index[0], assign_index[1]
                diff = pos_x[x_idx] - pos_y[y_idx]
                squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
                weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

            torch_scatter = importlib.import_module("torch_scatter")
            y = torch_scatter.scatter(
                x[x_idx] * weights, y_idx, 0, dim_size=pos_y.size(0), reduce="sum"
            )
            y = y / torch_scatter.scatter(
                weights, y_idx, 0, dim_size=pos_y.size(0), reduce="sum"
            )

            return y.float(), x_idx, y_idx, weights

        @require_version_spec("torch_geometric")
        def encode(
            self,
            x: torch.Tensor,
            edge_features: torch.Tensor,
            graph: GraphType,
            position_mesh: torch.Tensor,
            position_pivotal: torch.Tensor,
        ):
            r"""Encode mesh features to pivotal space.

            Parameters
            ----------
            x : torch.Tensor
                Input node features of shape :math:`(N_{nodes}^{batch}, D_{in}^{node})`.
            edge_features : torch.Tensor
                Edge features of shape :math:`(N_{edges}^{batch}, D_{in}^{edge})`.
            graph : :class:`~physicsnemo.nn.module.gnn_layers.graph_types.GraphType`
                PyG graph container with batch information.
            position_mesh : torch.Tensor
                Per-graph reference mesh positions of shape :math:`(N_{mesh}, D_{pos})`.
                These positions are repeated internally across the batch.
            position_pivotal : torch.Tensor
                Per-graph pivotal positions of shape :math:`(N_{pivotal}, D_{pos})`.
                These positions are repeated internally across the batch.

            Returns
            -------
            torch.Tensor
                Encoded features in pivotal space of shape
                :math:`(N_{pivotal}^{batch}, D_{enc})`.
            """
            if not torch.compiler.is_compiling():
                if x.ndim != 2:
                    raise ValueError(
                        f"Expected 2D node features (N_nodes, D_in) but got shape {tuple(x.shape)}"
                    )
                if edge_features.ndim != 2:
                    raise ValueError(
                        f"Expected 2D edge features (N_edges, D_in) but got shape {tuple(edge_features.shape)}"
                    )
                if position_mesh.ndim != 2 or position_pivotal.ndim != 2:
                    raise ValueError(
                        f"Expected position tensors to be 2D, got {tuple(position_mesh.shape)} and {tuple(position_pivotal.shape)}"
                    )
            x = self.encoder_processor(x, edge_features, graph)
            x = self.PivotalNorm(x)
            nodes_index = torch.arange(graph.batch_size).to(x.device)
            pyg = importlib.import_module("torch_geometric")
            if isinstance(graph, pyg.data.Data):
                batch_mesh = graph.batch
            else:
                raise ValueError(f"Unsupported graph type: {type(graph)}")
            position_mesh_batch = position_mesh.repeat(graph.batch_size, 1)
            position_pivotal_batch = position_pivotal.repeat(graph.batch_size, 1)
            batch_pivotal = nodes_index.repeat_interleave(
                torch.tensor([len(position_pivotal)] * graph.batch_size).to(x.device)
            )

            x, _, _, _ = self.knn_interpolate(
                x=x,
                pos_x=position_mesh_batch,
                pos_y=position_pivotal_batch,
                batch_x=batch_mesh,
                batch_y=batch_pivotal,
            )

            return x

        @require_version_spec("torch_geometric")
        def decode(
            self,
            x: torch.Tensor,
            edge_features: torch.Tensor,
            graph: GraphType,
            position_mesh: torch.Tensor,
            position_pivotal: torch.Tensor,
        ):
            r"""Decode pivotal features back to mesh space.

            Parameters
            ----------
            x : torch.Tensor
                Input features in pivotal space of shape
                :math:`(N_{pivotal}^{batch}, D_{enc})`.
            edge_features : torch.Tensor
                Edge features of shape :math:`(N_{edges}^{batch}, D_{in}^{edge})`.
            graph : :class:`~physicsnemo.nn.module.gnn_layers.graph_types.GraphType`
                PyG graph container with batch information.
            position_mesh : torch.Tensor
                Per-graph mesh positions of shape :math:`(N_{mesh}, D_{pos})`.
            position_pivotal : torch.Tensor
                Per-graph pivotal positions of shape :math:`(N_{pivotal}, D_{pos})`.

            Returns
            -------
            torch.Tensor
                Decoded features in mesh space of shape
                :math:`(N_{nodes}^{batch}, D_{out}^{decode})`.
            """
            if not torch.compiler.is_compiling():
                if edge_features.ndim != 2:
                    raise ValueError(
                        f"Expected 2D edge features (N_edges, D_in) but got shape {tuple(edge_features.shape)}"
                    )
                if position_mesh.ndim != 2 or position_pivotal.ndim != 2:
                    raise ValueError(
                        f"Expected position tensors to be 2D, got {tuple(position_mesh.shape)} and {tuple(position_pivotal.shape)}"
                    )
            nodes_index = torch.arange(graph.batch_size).to(x.device)
            pyg = importlib.import_module("torch_geometric")
            if isinstance(graph, pyg.data.Data):
                batch_mesh = graph.batch
            else:
                raise ValueError(f"Unsupported graph type: {type(graph)}")
            position_mesh_batch = position_mesh.repeat(graph.batch_size, 1)
            position_pivotal_batch = position_pivotal.repeat(graph.batch_size, 1)
            batch_pivotal = nodes_index.repeat_interleave(
                torch.tensor([len(position_pivotal)] * graph.batch_size).to(x.device)
            )

            x, _, _, _ = self.knn_interpolate(
                x=x,
                pos_x=position_pivotal_batch,
                pos_y=position_mesh_batch,
                batch_x=batch_pivotal,
                batch_y=batch_mesh,
            )

            x = self.decoder_processor(x, edge_features, graph)
            return x