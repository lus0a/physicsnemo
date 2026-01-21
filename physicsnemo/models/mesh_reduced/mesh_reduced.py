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

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from jaxtyping import Float, Int

from physicsnemo import ModelMetaData, Module
from physicsnemo.core.version_check import check_version_spec
from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNet
from physicsnemo.nn.gnn_layers.graph_types import GraphType

Tensor = torch.Tensor

_TORCH_CLUSTER_AVAILABLE = check_version_spec("torch_cluster", "0.0.0", hard_fail=False)
_TORCH_SCATTER_AVAILABLE = check_version_spec("torch_scatter", "0.0.0", hard_fail=False)

# Optional PyG import for type checks
try:
    import torch_geometric as pyg  # type: ignore
except Exception:  # pragma: no cover
    pyg = None  # type: ignore

@dataclass
class MetaData(ModelMetaData):
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    onnx: bool = False
    func_torch: bool = True
    auto_grad: bool = True


class Mesh_Reduced(Module):
    r"""PbGMR-GMUS mesh-reduced architecture.

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
        Number of nearest neighbors for interpolation.
    aggregation : Literal["sum", "mean"], optional, default="mean"
        Message aggregation type. Allowed values are ``"sum"`` and ``"mean"``.

    Forward
    -------
    node_features : torch.Tensor
        Input node features of shape :math:`(N_{nodes}^{batch}, D_{in}^{node})`.
    edge_features : torch.Tensor
        Input edge features of shape :math:`(N_{edges}^{batch}, D_{in}^{edge})`.
    graph : :class:`~physicsnemo.nn.gnn_layers.utils.GraphType`
        Graph connectivity/topology container (PyG).
        Connectivity/topology only. Do not duplicate node or edge features on the graph;
        pass them via ``node_features`` and ``edge_features``. If present on
        the graph, they will be ignored by the model.
        ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
        ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
        The current :class:`~physicsnemo.nn.gnn_layers.graph_types.GraphType` resolves to
        PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
        :mod:`physicsnemo.nn.gnn_layers.graph_types` for the exact alias and requirements.
    position_mesh : torch.Tensor
        Per-graph reference mesh positions of shape :math:`(N_{mesh}, D_{pos})`.
        These positions are repeated internally across the batch.
    position_pivotal : torch.Tensor
        Per-graph pivotal positions of shape :math:`(N_{pivotal}, D_{pos})`.
        These positions are repeated internally across the batch.

    Outputs
    -------
    torch.Tensor
        Decoded node features of shape :math:`(N_{nodes}^{batch}, D_{out}^{decode})`.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from physicsnemo.models.mesh_reduced.mesh_reduced import Mesh_Reduced
    >>>
    >>> # Choose a consistent device
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>>
    >>> # Instantiate model
    >>> model = Mesh_Reduced(
    ...     input_dim_nodes=4,
    ...     input_dim_edges=3,
    ...     output_decode_dim=2,
    ... ).to(device)
    >>>
    >>> # Build a simple PyG graph
    >>> num_nodes, num_edges = 10, 15
    >>> edge_index = torch.randint(0, num_nodes, (2, num_edges))
    >>> graph = Data(edge_index=edge_index, num_nodes=num_nodes).to(device)
    >>> # For a single graph, set a batch vector of zeros
    >>> graph.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    >>>
    >>> # Node/edge features
    >>> node_features = torch.randn(num_nodes, 4, device=device)
    >>> edge_features = torch.randn(num_edges, 3, device=device)
    >>>
    >>> # Per-graph positions (repeated internally across the batch)
    >>> position_mesh = torch.randn(20, 3, device=device)       # (N_mesh, D_pos)
    >>> position_pivotal = torch.randn(5, 3, device=device)     # (N_pivotal, D_pos)
    >>>
    >>> # Forward pass
    >>> out = model(node_features, edge_features, graph, position_mesh, position_pivotal)
    >>> out.size()
    torch.Size([10, 2])

    Notes
    -----
    Reference: `Predicting physics in mesh-reduced space with temporal attention <https://arxiv.org/pdf/2201.09113>`.
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
        aggregation: Literal["sum", "mean"] = "mean",
    ):
        super().__init__(meta=MetaData())
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

        # Public constructor attributes for validation/serialization
        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_encode_dim = output_encode_dim
        self.output_decode_dim = output_decode_dim

    @torch.no_grad()
    def knn_interpolate(
        self,
        x: Float[Tensor, "n_x d_x"],
        pos_x: Float[Tensor, "n_x d_pos"],
        pos_y: Float[Tensor, "n_y d_pos"],
        batch_x: Optional[Int[Tensor, "n_x"]] = None,  # noqa: F821
        batch_y: Optional[Int[Tensor, "n_y"]] = None,  # noqa: F821
        k: int = 3,
        num_workers: int = 1,
    ) -> Tuple[
        Float[Tensor, "n_y d_x"],
        Int[Tensor, "k_n"],  # noqa: F821
        Int[Tensor, "k_n"],  # noqa: F821
        Float[Tensor, "k_n 1"],  # noqa: F821
    ]:
        r"""Perform k-nearest neighbor interpolation from ``pos_x`` to ``pos_y``.

        Parameters
        ----------
        x : torch.Tensor
            Source features of shape :math:`(N_x, D_x)`.
        pos_x : torch.Tensor
            Source positions of shape :math:`(N_x, D_{pos})`.
        pos_y : torch.Tensor
            Target positions of shape :math:`(N_y, D_{pos})`.
        batch_x : torch.Tensor, optional
            Batch indices for source positions.
        batch_y : torch.Tensor, optional
            Batch indices for target positions.
        k : int, optional, default=3
            Number of nearest neighbors to consider.
        num_workers : int, optional, default=1
            Number of workers for the KNN op.

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
        if not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE):
            raise RuntimeError(
                "kNN interpolation requires 'torch_cluster' and 'torch_scatter'. "
                "Install PyTorch Geometric dependencies following: "
                "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
            )
        import importlib

        torch_cluster = importlib.import_module("torch_cluster")
        torch_scatter = importlib.import_module("torch_scatter")

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

        y = torch_scatter.scatter(
            x[x_idx] * weights, y_idx, 0, dim_size=pos_y.size(0), reduce="sum"
        )
        y = y / torch_scatter.scatter(
            weights, y_idx, 0, dim_size=pos_y.size(0), reduce="sum"
        )

        return y.float(), x_idx, y_idx, weights

    def encode(
        self,
        x: Float[Tensor, "n_nodes input_dim_nodes"],
        edge_features: Float[Tensor, "n_edges input_dim_edges"],
        graph: GraphType,
        position_mesh: Float[Tensor, "n_mesh d_pos"],
        position_pivotal: Float[Tensor, "n_pivotal d_pos"],
    ) -> Float[Tensor, "n_pivotal_batched output_encode_dim"]:
        r"""Encode mesh features to pivotal space.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape :math:`(N_{nodes}^{batch}, D_{in}^{node})`.
        edge_features : torch.Tensor
            Edge features of shape :math:`(N_{edges}^{batch}, D_{in}^{edge})`.
        graph : :class:`~physicsnemo.nn.gnn_layers.utils.GraphType`
            Graph connectivity/topology container (PyG).
            Connectivity/topology only. Do not duplicate node or edge features on the graph;
            pass them via ``node_features`` and ``edge_features``. If present on
            the graph, they will be ignored by the model.
            ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
            ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
            The current :class:`~physicsnemo.nn.gnn_layers.graph_types.GraphType` resolves to
            PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
            :mod:`physicsnemo.nn.gnn_layers.graph_types` for the exact alias and requirements.
        position_mesh : torch.Tensor
            Per-graph mesh positions of shape :math:`(N_{mesh}, D_{pos})`.
        position_pivotal : torch.Tensor
            Per-graph pivotal positions of shape :math:`(N_{pivotal}, D_{pos})`.

        Returns
        -------
        torch.Tensor
            Encoded features in pivotal space of shape
            :math:`(N_{pivotal}^{batch}, D_{enc})`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or x.shape[1] != self.input_dim_nodes:
                raise ValueError(
                    f"Expected tensor of shape (N_nodes, {self.input_dim_nodes}) but got tensor of shape {tuple(x.shape)}"
                )
            if (
                edge_features.ndim != 2
                or edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_edges, {self.input_dim_edges}) but got tensor of shape {tuple(edge_features.shape)}"
                )
            if position_mesh.ndim != 2 or position_pivotal.ndim != 2:
                raise ValueError(
                    f"Expected position tensors to be 2D, got shapes {tuple(position_mesh.shape)} and {tuple(position_pivotal.shape)}"
                )

        # Encode on the mesh graph
        x = self.encoder_processor(x, edge_features, graph)
        x = self.PivotalNorm(x)  # (N_nodes^batch, D_enc)

        # Build batched positional tensors and batch ids for KNN interpolation
        if (pyg is not None) and isinstance(graph, pyg.data.Data):
            batch_mesh = graph.batch
            batch_size = (
                int(batch_mesh.max().item()) + 1 if batch_mesh.numel() > 0 else 1
            )
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")

        nodes_index = torch.arange(batch_size, device=x.device)
        position_mesh_batch = position_mesh.repeat(batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(
            torch.tensor([len(position_pivotal)] * batch_size, device=x.device)
        )

        # Interpolate from mesh to pivotal positions
        x, _, _, _ = self.knn_interpolate(
            x=x,
            pos_x=position_mesh_batch,
            pos_y=position_pivotal_batch,
            batch_x=batch_mesh,
            batch_y=batch_pivotal,
            k=self.k,
        )
        return x

    def decode(
        self,
        x: Float[Tensor, "n_pivotal_batched output_encode_dim"],
        edge_features: Float[Tensor, "n_edges input_dim_edges"],
        graph: GraphType,
        position_mesh: Float[Tensor, "n_mesh d_pos"],
        position_pivotal: Float[Tensor, "n_pivotal d_pos"],
    ) -> Float[Tensor, "n_nodes output_decode_dim"]:
        r"""Decode pivotal features back to mesh space.

        Parameters
        ----------
        x : torch.Tensor
            Input features in pivotal space of shape
            :math:`(N_{pivotal}^{batch}, D_{enc})`.
        edge_features : torch.Tensor
            Edge features of shape :math:`(N_{edges}^{batch}, D_{in}^{edge})`.
        graph : :class:`~physicsnemo.nn.gnn_layers.utils.GraphType`
            Graph connectivity/topology container (PyG).
            Connectivity/topology only. Do not duplicate node or edge features on the graph;
            pass them via ``node_features`` and ``edge_features``. If present on
            the graph, they will be ignored by the model.
            ``node_features.shape[0]`` must equal the number of nodes in the graph ``graph.num_nodes``.
            ``edge_features.shape[0]`` must equal the number of edges in the graph ``graph.num_edges``.
            The current :class:`~physicsnemo.nn.gnn_layers.graph_types.GraphType` resolves to
            PyTorch Geometric objects (``torch_geometric.data.Data`` or ``torch_geometric.data.HeteroData``). See
            :mod:`physicsnemo.nn.gnn_layers.graph_types` for the exact alias and requirements.
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
            if (
                edge_features.ndim != 2
                or edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_edges, {self.input_dim_edges}) but got tensor of shape {tuple(edge_features.shape)}"
                )
            if position_mesh.ndim != 2 or position_pivotal.ndim != 2:
                raise ValueError(
                    f"Expected position tensors to be 2D, got shapes {tuple(position_mesh.shape)} and {tuple(position_pivotal.shape)}"
                )

        if (pyg is not None) and isinstance(graph, pyg.data.Data):
            batch_mesh = graph.batch
            batch_size = (
                int(batch_mesh.max().item()) + 1 if batch_mesh.numel() > 0 else 1
            )
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")

        nodes_index = torch.arange(batch_size, device=x.device)
        position_mesh_batch = position_mesh.repeat(batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(
            torch.tensor([len(position_pivotal)] * batch_size, device=x.device)
        )

        # Interpolate from pivotal back to mesh positions
        x, _, _, _ = self.knn_interpolate(
            x=x,
            pos_x=position_pivotal_batch,
            pos_y=position_mesh_batch,
            batch_x=batch_pivotal,
            batch_y=batch_mesh,
            k=self.k,
        )

        # Decode on the mesh graph
        x = self.decoder_processor(x, edge_features, graph)
        return x

    def forward(
        self,
        node_features: Float[Tensor, "n_nodes input_dim_nodes"],
        edge_features: Float[Tensor, "n_edges input_dim_edges"],
        graph: GraphType,
        position_mesh: Float[Tensor, "n_mesh d_pos"],
        position_pivotal: Float[Tensor, "n_pivotal d_pos"],
    ) -> Float[Tensor, "n_nodes output_decode_dim"]:
        if not torch.compiler.is_compiling():
            if (
                node_features.ndim != 2
                or node_features.shape[1] != self.input_dim_nodes
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_nodes, {self.input_dim_nodes}) but got tensor of shape {tuple(node_features.shape)}"
                )
            if (
                edge_features.ndim != 2
                or edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_edges, {self.input_dim_edges}) but got tensor of shape {tuple(edge_features.shape)}"
                )
        enc = self.encode(
            node_features, edge_features, graph, position_mesh, position_pivotal
        )
        dec = self.decode(enc, edge_features, graph, position_mesh, position_pivotal)
        return dec
