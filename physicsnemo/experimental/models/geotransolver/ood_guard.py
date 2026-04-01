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

"""OOD (Out-of-Distribution) Guard for runtime anomaly detection.

Provides two complementary checks:
1. **Global parameter bounds** — per-dimension bounding box on global embeddings.
2. **Geometry context kNN** — k-nearest-neighbor distance in a latent geometry space.

During training, the guard collects calibration statistics.  During inference,
it compares incoming data against those statistics and emits warnings when
inputs fall outside the training distribution.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn


_RED = "\033[91m"
_RESET = "\033[0m"


class OODGuard(nn.Module):
    """Out-of-distribution guard using global-parameter bounds and geometry kNN.

    Parameters
    ----------
    buffer_size : int
        Capacity of the geometry embedding FIFO buffer (typically = training set size).
    global_dim : int | None
        Dimensionality of global embeddings.  ``None`` disables the global check.
    geometry_embed_dim : int | None
        Dimensionality of pooled geometry context vectors.  ``None`` disables the
        geometry kNN check.
    knn_k : int
        Number of nearest neighbours for the geometry distance check.
    """

    def __init__(
        self,
        buffer_size: int,
        global_dim: int | None = None,
        geometry_embed_dim: int | None = None,
        knn_k: int = 10,
    ) -> None:
        super().__init__()
        self.buffer_size = buffer_size

        # Global parameter bounds
        if global_dim is not None:
            self.register_buffer(
                "global_min", torch.full((global_dim,), float("inf"))
            )
            self.register_buffer(
                "global_max", torch.full((global_dim,), float("-inf"))
            )
        else:
            self.register_buffer("global_min", None)
            self.register_buffer("global_max", None)

        # Geometry kNN buffer
        if geometry_embed_dim is not None:
            self.register_buffer(
                "geo_embeddings", torch.zeros(buffer_size, geometry_embed_dim)
            )
            self.register_buffer("geo_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("knn_threshold", torch.tensor(float("inf")))
        else:
            self.register_buffer("geo_embeddings", None)
            self.register_buffer("geo_ptr", None)
            self.register_buffer("knn_threshold", None)

        self.register_buffer("knn_k", torch.tensor(knn_k, dtype=torch.long))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(
        self,
        global_embedding: torch.Tensor | None = None,
        geometry_context: torch.Tensor | None = None,
    ) -> None:
        """Accumulate calibration data (call during training).

        Parameters
        ----------
        global_embedding : Tensor | None
            Shape ``(B, N_g, C_g)`` — raw global embedding from the model.
        geometry_context : Tensor | None
            Shape ``(B, H, S, D)`` — geometry context from the context builder.
        """
        self._collect_global(global_embedding)
        self._collect_geometry(geometry_context)

    @torch.no_grad()
    def check(
        self,
        global_embedding: torch.Tensor | None = None,
        geometry_context: torch.Tensor | None = None,
    ) -> None:
        """Run OOD checks and emit warnings (call during inference).

        Parameters
        ----------
        global_embedding : Tensor | None
            Shape ``(B, N_g, C_g)`` — raw global embedding from the model.
        geometry_context : Tensor | None
            Shape ``(B, H, S, D)`` — geometry context from the context builder.
        """
        self._check_global(global_embedding)
        self._check_geometry(geometry_context)

    @torch.no_grad()
    def compute_threshold(self) -> None:
        """Compute the kNN threshold from the accumulated geometry buffer."""
        if self.geo_embeddings is None:
            return
        ptr = self.geo_ptr.item()
        if ptr == 0:
            return
        n_valid = min(ptr, self.buffer_size)
        store = self.geo_embeddings[:n_valid]
        store_norm = store / (store.norm(dim=-1, keepdim=True) + 1e-8)
        dists = torch.cdist(store_norm, store_norm)
        dists.fill_diagonal_(float("inf"))
        k = self.knn_k.item()
        kth_dists = dists.topk(k, largest=False).values[:, -1]
        threshold = torch.quantile(kth_dists, 0.99)
        self.knn_threshold.copy_(threshold)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_global(self, global_embedding: torch.Tensor | None) -> None:
        if global_embedding is None or self.global_min is None:
            return
        batch_min = global_embedding.detach().min(dim=0).values.min(dim=0).values
        batch_max = global_embedding.detach().max(dim=0).values.max(dim=0).values
        self.global_min.copy_(torch.minimum(self.global_min, batch_min))
        self.global_max.copy_(torch.maximum(self.global_max, batch_max))

    def _collect_geometry(self, geometry_context: torch.Tensor | None) -> None:
        if geometry_context is None or self.geo_embeddings is None:
            return
        pooled = geometry_context.detach().mean(dim=(1, 2))  # (B, D)
        ptr = self.geo_ptr.item()
        for i in range(pooled.shape[0]):
            self.geo_embeddings[ptr % self.buffer_size] = pooled[i]
            ptr += 1
        self.geo_ptr.fill_(ptr)

    def _check_global(self, global_embedding: torch.Tensor | None) -> None:
        if global_embedding is None or self.global_min is None:
            return
        if torch.isinf(self.global_min).any():
            return
        vals = global_embedding.detach()
        batch_min = vals.min(dim=0).values.min(dim=0).values
        batch_max = vals.max(dim=0).values.max(dim=0).values
        for d in range(batch_min.shape[0]):
            lo = self.global_min[d].item()
            hi = self.global_max[d].item()
            if batch_min[d].item() < lo:
                logging.warning(
                    f"{_RED}OOD Guard: global_embedding dim {d} value "
                    f"{batch_min[d].item():.4f} below training min {lo:.4f}{_RESET}"
                )
            if batch_max[d].item() > hi:
                logging.warning(
                    f"{_RED}OOD Guard: global_embedding dim {d} value "
                    f"{batch_max[d].item():.4f} above training max {hi:.4f}{_RESET}"
                )

    def _check_geometry(self, geometry_context: torch.Tensor | None) -> None:
        if geometry_context is None or self.geo_embeddings is None:
            return
        if torch.isinf(self.knn_threshold):
            return
        pooled = geometry_context.detach().mean(dim=(1, 2))  # (B, D)
        z = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
        store = self.geo_embeddings
        store_norm = store / (store.norm(dim=-1, keepdim=True) + 1e-8)
        dists = torch.cdist(z, store_norm)  # (B, buf_size)
        k = self.knn_k.item()
        kth_dists = dists.topk(k, largest=False).values[:, -1]  # (B,)
        threshold = self.knn_threshold.item()
        for i in range(kth_dists.shape[0]):
            dist_val = kth_dists[i].item()
            if dist_val > threshold:
                logging.warning(
                    f"{_RED}OOD Guard: geometry sample {i} kNN distance "
                    f"{dist_val:.4f} above threshold {threshold:.4f}{_RESET}"
                )
