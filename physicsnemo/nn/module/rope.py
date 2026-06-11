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

r"""Rotary position embedding (RoPE) primitives.

Parameter-free helpers for axial 2D RoPE, shared by attention modules (e.g.
:class:`~physicsnemo.nn.module.dit_layers.RopeNatten2DSelfAttention`). RoPE
encodes position as a relative rotation between query and key, so attention
becomes translation-equivariant within a window.

Math (axial 2D RoPE)
--------------------
``head_dim`` is split in half: the first half rotates by row index, the second
by column index. Each axis has ``head_dim/4`` rotation pairs sharing a frequency
:math:`\theta_k = \text{base}^{-2k/(head\_dim/2)}` for
:math:`k = 0 \ldots head\_dim/4 - 1`. For an adjacent channel pair
:math:`(x_a, x_b)` at angle :math:`\phi`, the rotation is
:math:`(x_a \cos\phi - x_b \sin\phi,\ x_a \sin\phi + x_b \cos\phi)`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def build_axial_rope_cos_sin(
    h: int,
    w: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Precompute axial 2D RoPE cos/sin tables for an :math:`h \times w` token grid.

    The first ``head_dim/2`` channels are rotated by the row index, the last
    ``head_dim/2`` by the column index. Within each axis-half, frequency
    :math:`\theta_k = \text{theta}^{-2k/(head\_dim/2)}` drives the adjacent
    channel pair ``(2k, 2k+1)``.

    Parameters
    ----------
    h : int
        Token grid height.
    w : int
        Token grid width.
    head_dim : int
        Per-head channel dimension. Must be divisible by 4 (half per axis, then
        adjacent pairs within each half).
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.
    device : torch.device, optional
        Device for the generated tables.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)``, each of shape :math:`(h, w, head\_dim)` in fp32.
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"head_dim={head_dim} must be divisible by 4 for axial 2D RoPE "
            f"(half per axis, then adjacent pairs within each half)."
        )
    half = head_dim // 2  # channels per axis

    # Frequencies for one axis: head_dim/4 unique values, each shared across an
    # adjacent channel pair via repeat_interleave below.
    k = torch.arange(0, half, 2, dtype=torch.float32, device=device)
    freqs = theta ** (-k / half)  # (head_dim/4,)

    row_idx = torch.arange(h, dtype=torch.float32, device=device)
    row_ang = row_idx[:, None] * freqs[None, :]  # (h, head_dim/4)
    col_idx = torch.arange(w, dtype=torch.float32, device=device)
    col_ang = col_idx[:, None] * freqs[None, :]  # (w, head_dim/4)

    # repeat_interleave(2) sends [a, b, c, ...] -> [a, a, b, b, c, c, ...] so that
    # the adjacent channel pair (2k, 2k+1) shares frequency theta_k.
    cos_row = row_ang.cos().repeat_interleave(2, dim=-1)  # (h, half)
    sin_row = row_ang.sin().repeat_interleave(2, dim=-1)
    cos_col = col_ang.cos().repeat_interleave(2, dim=-1)  # (w, half)
    sin_col = col_ang.sin().repeat_interleave(2, dim=-1)

    cos = torch.cat(
        [
            cos_row[:, None, :].expand(h, w, half),
            cos_col[None, :, :].expand(h, w, half),
        ],
        dim=-1,
    )  # (h, w, head_dim)
    sin = torch.cat(
        [
            sin_row[:, None, :].expand(h, w, half),
            sin_col[None, :, :].expand(h, w, half),
        ],
        dim=-1,
    )
    return cos.contiguous(), sin.contiguous()


def build_rope_cos_sin_1d(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Precompute 1D RoPE cos/sin tables for a length-``seq_len`` sequence.

    The standard sequence RoPE: every channel rotates by the token position,
    with ``head_dim/2`` frequencies :math:`\theta_k = \text{theta}^{-2k/head\_dim}`
    for :math:`k = 0 \ldots head\_dim/2 - 1`, each driving the adjacent channel
    pair ``(2k, 2k+1)``.

    Parameters
    ----------
    seq_len : int
        Number of positions in the sequence.
    head_dim : int
        Per-head channel dimension. Must be even (rotation acts on adjacent
        channel pairs).
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.
    device : torch.device, optional
        Device for the generated tables.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)``, each of shape :math:`(seq\_len, head\_dim)` in fp32.
    """
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim={head_dim} must be even for 1D RoPE "
            f"(rotation acts on adjacent channel pairs)."
        )

    # head_dim/2 unique frequencies, each shared across an adjacent channel pair
    # via repeat_interleave below.
    k = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    freqs = theta ** (-k / head_dim)  # (head_dim/2,)

    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    ang = pos[:, None] * freqs[None, :]  # (seq_len, head_dim/2)

    cos = ang.cos().repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = ang.sin().repeat_interleave(2, dim=-1)
    return cos.contiguous(), sin.contiguous()


def rotate_half_pairs(x: torch.Tensor) -> torch.Tensor:
    r"""Swap adjacent channel pairs with a sign flip.

    Maps ``(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)``. Used to compute
    ``q * cos + rotate_half_pairs(q) * sin``, the standard formulation of a 2D
    rotation on adjacent-pair channel encoding.

    Parameters
    ----------
    x : torch.Tensor
        Tensor whose last dimension is the (even) channel dimension.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``x`` with adjacent pairs rotated.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    r"""Apply a rotary position embedding to ``x`` given precomputed tables.

    Computes ``x * cos + rotate_half_pairs(x) * sin`` in fp32 (the sign-flipped
    term can accumulate error in low precision) and casts back to ``x``'s dtype.
    ``cos`` and ``sin`` must broadcast against ``x`` over its trailing
    ``(..., positions, head_dim)`` dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Query or key tensor of shape :math:`(\ldots, \text{positions}, head\_dim)`.
    cos, sin : torch.Tensor
        Rotation tables broadcastable to ``x`` (e.g. shape
        :math:`(\text{positions}, head\_dim)`), as produced by
        :func:`build_axial_rope_cos_sin`.

    Returns
    -------
    torch.Tensor
        Rotated tensor of the same shape and dtype as ``x``.
    """
    in_dtype = x.dtype
    x = x.float()
    return (x * cos + rotate_half_pairs(x) * sin).to(in_dtype)


class RotaryPositionEmbedding2D(nn.Module):
    r"""Axial 2D rotary position embedding as a reusable module.

    Precomputes cos/sin tables for an :math:`(h, w)` token grid and rotates
    query/key tensors shaped :math:`(\ldots, \text{seq}, head\_dim)` with
    ``seq == h * w`` in row-major order (height varies slowest). This is the
    standard :math:`(B, \text{heads}, N, head\_dim)` attention layout, so the
    module is a drop-in for any attention operating on a flattened 2D token
    sequence (e.g. a vision transformer).

    For NATTEN windowed attention, which operates on spatial
    :math:`(B, h, w, \text{heads}, head\_dim)` tensors, use
    :class:`~physicsnemo.nn.module.dit_layers.RopeNatten2DSelfAttention` instead.

    The cos/sin tables are stored as ``persistent=False`` buffers (they are
    deterministically rebuilt from ``(latent_hw, head_dim, theta)``). Building
    them at the global grid size lets ``distribute_module`` shard them along
    height for domain parallelism, with no explicit rank offset in model code.

    Parameters
    ----------
    head_dim : int
        Per-head channel dimension. Must be divisible by 4.
    latent_hw : Tuple[int, int]
        Spatial size :math:`(h, w)` of the token grid.
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.

    Forward
    -------
    q, k : torch.Tensor
        Query and key tensors of shape :math:`(\ldots, h \cdot w, head\_dim)`.
    latent_hw : Tuple[int, int], optional
        If given and different from the construction-time grid, the tables are
        rebuilt in place before rotating (off the ``torch.compile`` path).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The rotated ``(q, k)``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.rope import RotaryPositionEmbedding2D
    >>> rope = RotaryPositionEmbedding2D(head_dim=16, latent_hw=(4, 4))
    >>> q = torch.randn(2, 8, 16, 16)  # (B, heads, h*w, head_dim)
    >>> k = torch.randn(2, 8, 16, 16)
    >>> q_rot, k_rot = rope(q, k)
    >>> q_rot.shape
    torch.Size([2, 8, 16, 16])
    """

    def __init__(
        self,
        head_dim: int,
        latent_hw: Tuple[int, int],
        theta: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by 4 for axial 2D RoPE."
            )
        self.head_dim = int(head_dim)
        self.theta = float(theta)
        self._latent_hw: Tuple[int, int] = (int(latent_hw[0]), int(latent_hw[1]))
        cos, sin = build_axial_rope_cos_sin(
            *self._latent_hw, self.head_dim, theta=self.theta
        )
        # Flatten the spatial axes to (h*w, head_dim) so the tables broadcast
        # against any (..., seq, head_dim) attention layout.
        self.register_buffer("cos", cos.reshape(-1, self.head_dim), persistent=False)
        self.register_buffer("sin", sin.reshape(-1, self.head_dim), persistent=False)

    def _rebuild_for_shape(self, h: int, w: int) -> None:
        """Rebuild the cos/sin tables for a new latent shape (off the hot path)."""
        target_dtype = self.cos.dtype
        target_device = self.cos.device
        cos, sin = build_axial_rope_cos_sin(
            h, w, self.head_dim, theta=self.theta, device=target_device
        )
        self.register_buffer(
            "cos", cos.reshape(-1, self.head_dim).to(target_dtype), persistent=False
        )
        self.register_buffer(
            "sin", sin.reshape(-1, self.head_dim).to(target_dtype), persistent=False
        )
        self._latent_hw = (int(h), int(w))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if latent_hw is not None and (
            (int(latent_hw[0]), int(latent_hw[1])) != self._latent_hw
        ):
            self._rebuild_for_shape(int(latent_hw[0]), int(latent_hw[1]))

        n = self.cos.shape[0]
        if q.shape[-2] != n or k.shape[-2] != n:
            raise ValueError(
                f"q/k sequence length must be h*w={n} (latent_hw={self._latent_hw}), "
                f"but got q={q.shape[-2]}, k={k.shape[-2]}"
            )
        return apply_rotary_pos_emb(q, self.cos, self.sin), apply_rotary_pos_emb(
            k, self.cos, self.sin
        )


class RotaryPositionEmbedding1D(nn.Module):
    r"""1D rotary position embedding as a reusable module.

    The standard sequence RoPE used by most autoregressive / encoder
    transformers. Precomputes cos/sin tables for a length-``max_seq_len``
    sequence and rotates query/key tensors shaped
    :math:`(\ldots, \text{seq}, head\_dim)` (e.g. the
    :math:`(B, \text{heads}, N, head\_dim)` attention layout). Inputs shorter
    than ``max_seq_len`` are rotated with the leading positions, so a single
    module can serve any sequence length up to ``max_seq_len`` without a rebuild.

    The cos/sin tables are stored as ``persistent=False`` buffers (they are
    deterministically rebuilt from ``(max_seq_len, head_dim, theta)``).

    Parameters
    ----------
    head_dim : int
        Per-head channel dimension. Must be even.
    max_seq_len : int
        Maximum sequence length for which to precompute tables.
    theta : float, optional, default=10000.0
        Base used for the RoPE frequency schedule.

    Forward
    -------
    q, k : torch.Tensor
        Query and key tensors of shape :math:`(\ldots, \text{seq}, head\_dim)`
        with ``seq <= max_seq_len``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The rotated ``(q, k)``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn.module.rope import RotaryPositionEmbedding1D
    >>> rope = RotaryPositionEmbedding1D(head_dim=16, max_seq_len=128)
    >>> q = torch.randn(2, 8, 100, 16)  # (B, heads, seq, head_dim)
    >>> k = torch.randn(2, 8, 100, 16)
    >>> q_rot, k_rot = rope(q, k)
    >>> q_rot.shape
    torch.Size([2, 8, 100, 16])
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for 1D RoPE.")
        self.head_dim = int(head_dim)
        self.theta = float(theta)
        self.max_seq_len = int(max_seq_len)
        cos, sin = build_rope_cos_sin_1d(
            self.max_seq_len, self.head_dim, theta=self.theta
        )
        self.register_buffer("cos", cos, persistent=False)  # (max_seq_len, head_dim)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        if k.shape[-2] != seq_len:
            raise ValueError(
                f"q and k must share a sequence length; got q={seq_len}, "
                f"k={k.shape[-2]}"
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        # Slice the leading positions so the module serves any length <= max.
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)
