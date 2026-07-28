# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""U-FNO following Wen et al., Adv. Water Resour. 163, 104180 (2022) / arXiv:2109.03697.

Architecture (paper Sec. 3.2, Fig. 2):
  1) Lift  P:  a(x)  ->  v  (higher channel width)
  2) L  plain Fourier layers  +  M  U-Fourier layers
  3) Project Q:  v  ->  z(x)

Plain Fourier layer (Li et al.):
  v <- σ( K(v) + W(v) )

U-Fourier layer (Wen et al. Eq. 11):
  v <- σ( K(v) + U(v) + W(v) )
  where K = spectral integral operator (FFT · R · iFFT),
        U = two-step U-Net (local multi-scale conv),
        W = pointwise linear (1x1 conv).

Default L=M=half of ``num_fno_layers`` (paper: half Fourier + half U-Fourier).

I/O: (B, C_in, H, W) -> (B, C_out, H, W), same contract as PhysicsNeMo FNO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.fno import FNO


@dataclass
class MetaData(ModelMetaData):
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


def _as_modes2d(num_fno_modes: Union[int, Sequence[int]]) -> tuple[int, int]:
    if isinstance(num_fno_modes, (list, tuple)):
        if len(num_fno_modes) == 1:
            m = int(num_fno_modes[0])
            return m, m
        return int(num_fno_modes[0]), int(num_fno_modes[1])
    m = int(num_fno_modes)
    return m, m


def _gn(num_channels: int) -> nn.GroupNorm:
    groups = min(8, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class SpectralConv2d(nn.Module):
    """K: truncated Fourier integral operator (Li / Wen)."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        # Store as real (..., 2) = (re, im).  Not torch.cfloat Parameters:
        # GradScaler / AMP unscale does not support ComplexFloat grads.
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )

    @staticmethod
    def _as_complex(w: torch.Tensor) -> torch.Tensor:
        # w: (..., 2) real → complex64
        return torch.view_as_complex(w.float().contiguous())

    @staticmethod
    def _mul(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (B, Cin, H, Wf) x (Cin, Cout, H, Wf) -> (B, Cout, H, Wf)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cuFFT half/bfloat16 only allows power-of-two spatial sizes.
        # Elder padded grids (e.g. 64x256 + pad8 → 80x272) are not; under AMP
        # autocast would cast rfft2 to fp16 and crash. Always run FFT path in fp32.
        orig_dtype = x.dtype
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            b = x32.shape[0]
            h, w = x32.shape[-2], x32.shape[-1]
            x_ft = torch.fft.rfft2(x32, norm="ortho")
            out_ft = torch.zeros(
                b,
                self.out_channels,
                h,
                w // 2 + 1,
                dtype=torch.cfloat,
                device=x32.device,
            )
            m1 = min(self.modes1, h)
            m2 = min(self.modes2, w // 2 + 1)
            w1 = self._as_complex(self.weights1)
            w2 = self._as_complex(self.weights2)
            # top-left and bottom-left mode blocks (standard FNO-2d)
            out_ft[:, :, :m1, :m2] = self._mul(
                x_ft[:, :, :m1, :m2], w1[:, :, :m1, :m2]
            )
            out_ft[:, :, -m1:, :m2] = self._mul(
                x_ft[:, :, -m1:, :m2], w2[:, :, :m1, :m2]
            )
            out = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return out.to(dtype=orig_dtype)


class TwoStepUNet(nn.Module):
    """Two-step (2-level) U-Net used as U(·) inside each U-Fourier layer (Wen Fig. 2C)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            _gn(c),
            nn.GELU(),
        )
        self.down = nn.AvgPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            _gn(c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            _gn(c),
            nn.GELU(),
        )
        self.up_proj = nn.Conv2d(c, c, 1)
        self.dec = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1, bias=False),
            _gn(c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            _gn(c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        up = F.interpolate(e2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up_proj(up)
        return self.dec(torch.cat([up, e1], dim=1))


class FourierLayer(nn.Module):
    """Plain Fourier layer: σ(K(v) + W(v))."""

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.w(x))


class UFourierLayer(nn.Module):
    """U-Fourier layer (Wen Eq. 11): σ(K(v) + U(v) + W(v))."""

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.unet = TwoStepUNet(width)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.unet(x) + self.w(x))


class UFNO(Module):
    """U-FNO (Wen et al.): lift → Fourier×L → U-Fourier×M → project."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: str = "silu",  # unused; keep FNO-compatible kwargs
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: str = "gelu",  # unused (GELU fixed as paper σ)
        coord_features: bool = False,
        # U-FNO-specific (optional overrides)
        num_fourier_layers: Optional[int] = None,
        num_ufourier_layers: Optional[int] = None,
        unet_base: int = 32,  # ignored in Wen-faithful build; kept for config compat
        fusion: str = "sum",  # ignored
    ) -> None:
        if dimension != 2:
            raise ValueError("UFNO (Wen-style) currently supports dimension=2 only")
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = int(padding)
        self.padding_type = padding_type
        self.coord_features = bool(coord_features)
        width = int(latent_channels)
        modes1, modes2 = _as_modes2d(num_fno_modes)

        n_total = int(num_fno_layers)
        if num_fourier_layers is None and num_ufourier_layers is None:
            # paper: half Fourier + half U-Fourier
            L = n_total // 2
            M = n_total - L
        else:
            L = int(num_fourier_layers if num_fourier_layers is not None else 0)
            M = int(num_ufourier_layers if num_ufourier_layers is not None else max(0, n_total - L))
        self.num_fourier_layers = L
        self.num_ufourier_layers = M

        lift_in = in_channels + (2 if self.coord_features else 0)
        self.lift = nn.Conv2d(lift_in, width, kernel_size=1)

        self.fourier_layers = nn.ModuleList(
            [FourierLayer(width, modes1, modes2) for _ in range(L)]
        )
        self.ufourier_layers = nn.ModuleList(
            [UFourierLayer(width, modes1, modes2) for _ in range(M)]
        )

        # Project Q: width -> ... -> out_channels (paper: FC network Q)
        proj: list[nn.Module] = []
        if decoder_layers <= 1:
            proj.append(nn.Conv2d(width, out_channels, kernel_size=1))
        else:
            proj.append(nn.Conv2d(width, decoder_layer_size, kernel_size=1))
            proj.append(nn.GELU())
            for _ in range(decoder_layers - 2):
                proj.append(nn.Conv2d(decoder_layer_size, decoder_layer_size, kernel_size=1))
                proj.append(nn.GELU())
            proj.append(nn.Conv2d(decoder_layer_size, out_channels, kernel_size=1))
        self.project = nn.Sequential(*proj)
        # alias for residual zero-init helpers in train_elder_fno
        self.decoder_net = self.project

    def _add_coords(self, x: torch.Tensor) -> torch.Tensor:
        if not self.coord_features:
            return x
        b, _, h, w = x.shape
        ys = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._add_coords(x)
        if self.padding > 0:
            x = F.pad(
                x,
                [self.padding] * 4,
                mode=self.padding_type if self.padding_type in ("constant", "reflect", "replicate") else "constant",
            )
        v = self.lift(x)
        for layer in self.fourier_layers:
            v = layer(v)
        for layer in self.ufourier_layers:
            v = layer(v)
        if self.padding > 0:
            p = self.padding
            v = v[..., p:-p, p:-p]
        return self.project(v)


def build_model(cfg_model, num_fno_modes, in_channels: Optional[int] = None) -> Module:
    """Factory: ``cfg_model.arch`` in {fno, ufno} (default fno)."""
    from omegaconf import OmegaConf

    if hasattr(cfg_model, "get"):
        arch = str(cfg_model.get("arch", "fno")).lower()
        mdl = cfg_model
    else:
        arch = "fno"
        mdl = OmegaConf.create(cfg_model)

    if in_channels is None:
        in_channels = 2 + int(mdl.get("dt_channel", False))

    common = dict(
        in_channels=in_channels,
        out_channels=int(mdl.out_channels),
        decoder_layers=int(mdl.decoder_layers),
        decoder_layer_size=int(mdl.decoder_layer_size),
        dimension=int(mdl.dimension),
        latent_channels=int(mdl.latent_channels),
        num_fno_layers=int(mdl.num_fno_layers),
        num_fno_modes=num_fno_modes,
        padding=int(mdl.padding),
    )

    if arch in ("ufno", "u-fno", "u_fno", "wen", "wen_ufno"):
        kwargs = dict(common)
        # optional explicit L / M
        if mdl.get("num_fourier_layers", None) is not None:
            kwargs["num_fourier_layers"] = int(mdl.num_fourier_layers)
        if mdl.get("num_ufourier_layers", None) is not None:
            kwargs["num_ufourier_layers"] = int(mdl.num_ufourier_layers)
        kwargs["coord_features"] = bool(mdl.get("coord_features", False))
        return UFNO(**kwargs)

    if arch in ("fno", "vanilla", "vanilla_fno"):
        return FNO(**common)

    raise ValueError(f"Unknown model.arch={arch!r}; use 'fno' or 'ufno'")
