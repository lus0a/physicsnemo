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

"""On-the-fly datapipe for the 2-D time-dependent convection-diffusion equation.

A pure-PyTorch pseudo-spectral solver generates pairs ``(T_n, T_{n+1})`` on a
periodic domain together with a divergence-free velocity field ``(u, v)``.
No external data or NVIDIA Warp kernels are required.

Equation integrated by the reference solver::

    dT/dt = -u * T_x - v * T_y + D * (T_xx + T_yy)

Each yielded sample contains the scalar field at two consecutive times
(``T0`` -> ``T1``) over a macro time step ``dt``, plus the velocity components,
so the same pair can be used for both data-driven supervision and PDE-residual
(PINO) losses.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "ConvectionDiffusion2D"
    auto_device: bool = True
    cuda_graphs: bool = False  # FFT ops are not CUDA-graph friendly here
    ddp_sharding: bool = False


class ConvectionDiffusion2D(Datapipe):
    """Continuously generate 2-D convection-diffusion solution pairs.

    Parameters
    ----------
    resolution : int
        Grid size ``N`` (domain is ``N x N``).
    batch_size : int
        Number of independent simulations per yielded batch.
    D : float
        Diffusivity.
    dt_macro : float
        Time span between ``T0`` and ``T1``. The reference solver sub-steps
        this internally to stay within the CFL limit.
    nr_ic_modes : int
        Low-pass cutoff (in grid mode indices) for the random initial condition.
    nr_vel_modes : int
        Low-pass cutoff for the random streamfunction used to build the
        divergence-free velocity field.
    u_max : float
        Target maximum speed ``max|sqrt(u^2+v^2)|`` after scaling.
    substeps : int
        Minimum number of integration sub-steps; increased automatically if
        the CFL condition requires it.
    length : float
        Physical side length of the (square, periodic) domain.
    device : str or torch.device
        Device on which to generate data.
    """

    def __init__(
        self,
        resolution: int = 64,
        batch_size: int = 32,
        D: float = 1e-3,
        dt_macro: float = 0.1,
        nr_ic_modes: int = 5,
        nr_vel_modes: int = 4,
        u_max: float = 1.5,
        substeps: int = 25,
        length: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        super().__init__(meta=MetaData())

        self.resolution = resolution
        self.batch_size = batch_size
        self.D = float(D)
        self.dt_macro = float(dt_macro)
        self.nr_ic_modes = nr_ic_modes
        self.nr_vel_modes = nr_vel_modes
        self.u_max = float(u_max)
        self.substeps = int(substeps)
        self.length = float(length)
        self.device = torch.device(device)

        # Grid / spectral quantities (precomputed, moved to device lazily)
        self.dx = self.length / self.resolution
        k = 2.0 * np.pi * np.fft.fftfreq(self.resolution, d=self.dx)  # wavenumbers [N]
        kx = torch.tensor(k, dtype=torch.float32).view(1, 1, -1)  # along last dim
        ky = torch.tensor(k, dtype=torch.float32).view(1, -1, 1)  # along dim -2
        self._kx = kx.to(self.device)
        self._ky = ky.to(self.device)
        self._k2 = (self._kx**2 + self._ky**2).to(self.device)
        # Precomputed spectral operators (complex64) to avoid re-multiplying by
        # 1j / -k2 every integration sub-step.
        self._ikx = 1j * self._kx
        self._iky = 1j * self._ky
        self._neg_k2 = -self._k2

        # Low-pass masks for smooth IC / velocity. Compare on the integer mode
        # index |m| (NOT the wavenumber 2*pi*|m|/L, which is ~6x larger).
        m = np.fft.fftfreq(self.resolution, d=self.dx) * self.length  # mode indices
        m = torch.tensor(m, dtype=torch.float32)
        mask_ic = ((m.abs().view(1, 1, -1) <= self.nr_ic_modes) &
                   (m.abs().view(1, -1, 1) <= self.nr_ic_modes))
        mask_vel = ((m.abs().view(1, 1, -1) <= self.nr_vel_modes) &
                    (m.abs().view(1, -1, 1) <= self.nr_vel_modes))
        self._mask_ic = mask_ic.to(self.device)
        self._mask_vel = mask_vel.to(self.device)
        # spectral decay envelope 1/(1+|k|^2)^1.5 for smooth fields
        decay = 1.0 / (1.0 + self._k2**1.5)
        self._decay = decay

    # ------------------------------------------------------------------
    # Random field generation
    # ------------------------------------------------------------------
    def _random_spectral_field(self, mask: Tensor) -> Tensor:
        """Real random field from low-passed, decaying spectral coefficients."""
        B, N = self.batch_size, self.resolution
        amp = torch.randn(B, N, N, device=self.device)
        phs = torch.randn(B, N, N, device=self.device)
        coeff = (amp + 1j * phs) * mask * self._decay
        coeff[:, 0, 0] = 0.0  # zero mean
        field = torch.fft.ifft2(coeff, norm="ortho").real
        return field

    def _random_ic(self) -> Tensor:
        """Random smooth initial condition normalized to ``[0, 1]``."""
        field = self._random_spectral_field(self._mask_ic)
        fmin = field.amin(dim=(1, 2), keepdim=True)
        fmax = field.amax(dim=(1, 2), keepdim=True)
        field = (field - fmin) / (fmax - fmin + 1e-8)
        return field

    def _random_velocity(self) -> Tuple[Tensor, Tensor]:
        """Divergence-free velocity from a random streamfunction psi.

        ``u = d psi/dy``, ``v = -d psi/dx`` computed spectrally so that
        ``u_x + v_y = 0`` exactly. Scaled to ``max|speed| ~ u_max``.
        """
        psi = self._random_spectral_field(self._mask_vel)
        psi_hat = torch.fft.fft2(psi, norm="ortho")
        u_hat = self._iky * psi_hat
        v_hat = -self._ikx * psi_hat
        u = torch.fft.ifft2(u_hat, norm="ortho").real
        v = torch.fft.ifft2(v_hat, norm="ortho").real
        speed = torch.sqrt(u**2 + v**2 + 1e-12)
        scale = self.u_max / (speed.amax() + 1e-12)
        return u * scale, v * scale

    # ------------------------------------------------------------------
    # Reference pseudo-spectral solver
    # ------------------------------------------------------------------
    def _cfl_dt_max(self) -> float:
        """Largest stable explicit-Euler sub-step for current params."""
        diff_limit = 0.25 * self.dx**2 / max(self.D, 1e-12)
        adv_limit = self.dx / (self.u_max + 1e-12)
        return min(diff_limit, adv_limit)

    def _integrate(self, T0: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Integrate CD from ``T0`` over ``dt_macro`` -> ``T1``.

        Pseudo-spectral derivatives + explicit-Euler sub-steps. The k=0 (mean)
        mode is pinned to its initial value to conserve mass exactly.
        """
        # number of sub-steps honoring CFL
        n_sub = max(self.substeps,
                    int(np.ceil(self.dt_macro / self._cfl_dt_max())))
        dt = self.dt_macro / n_sub

        th = torch.fft.fft2(T0, norm="ortho")
        mean0 = th[:, 0, 0].clone()  # preserve mean (mass conservation)

        for _ in range(n_sub):
            # Pseudo-spectral derivatives of the advective term (physical space).
            Tx = torch.fft.ifft2(self._ikx * th, norm="ortho").real
            Ty = torch.fft.ifft2(self._iky * th, norm="ortho").real
            # RHS: advection transformed back to spectral + diffusion added
            # directly in spectral space (no extra ifft for the Laplacian).
            rhs_hat = torch.fft.fft2(-(u * Tx + v * Ty), norm="ortho")
            rhs_hat = rhs_hat + self.D * self._neg_k2 * th
            th = th + dt * rhs_hat
            th[:, 0, 0] = mean0  # keep mean fixed

        return torch.fft.ifft2(th, norm="ortho").real

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------
    def generate_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        T0 = self._random_ic()
        u, v = self._random_velocity()
        T1 = self._integrate(T0, u, v)
        return T0, T1, u, v

    def __iter__(self) -> Dict[str, Tensor]:
        """Yield batches of ``{T0, T1, u, v, dt}`` infinitely.

        Tensor shapes are ``[batch, 1, N, N]``; ``dt`` is the macro time step
        (a Python float) used to form the time derivative in the PDE residual.
        """
        while True:
            T0, T1, u, v = self.generate_batch()
            yield {
                "T0": T0.unsqueeze(1),
                "T1": T1.unsqueeze(1),
                "u": u.unsqueeze(1),
                "v": v.unsqueeze(1),
                "dt": self.dt_macro,
            }

    def __len__(self) -> int:
        return sys.maxsize
