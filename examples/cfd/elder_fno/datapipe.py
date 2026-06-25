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

"""On-the-fly datapipe for the Elder problem (variable-density, non-Boussinesq).

A pure-PyTorch reference solver generates single-step pairs
``(c_n, p_n) -> (c_{n+1}, p_{n+1})`` for the Elder problem in *primitive
variables* (concentration ``c`` and pressure ``p``), without the Boussinesq
approximation. This mirrors the UG4 Lua setup (``elder`` without Boussinesq):
variable density ``rho(c) = rho_f + drho * c``, Darcy velocity, and a
variable-density Darcy flow equation (with fluid-mass storage) coupled to a
conservative transport equation.

Governing equations (SI units; row index increases downward, ``z`` downward
positive, gravity ``g_vec = (0, +g)`` downward, ``rho = rho_f + drho*c``,
``c = 1`` dense)::

    Darcy velocity :  q = -(k/mu)(grad p - rho g_vec)
    flow (p)       :  d(phi rho)/dt + div(rho q) = 0      (mass_scale=0 drops dp/dt only;
                                                          storage d(phi rho)/dt = phi*drho*dc/dt)
    transport (c)  :  d(phi rho c)/dt + div(rho q c) = div(rho phi Dm grad c)

Pressure is worked in the **equivalent-freshwater-head gauge**
``h = p - p_hydro`` with ``p_hydro(z) = rho_f * g * z`` (fresh-water
hydrostatic, ``h = 0`` initially). In this gauge the Darcy velocity becomes
``q = -(k/mu) grad h + (k drho c / mu) g_vec``: buoyancy appears explicitly as a
body force proportional to ``c``. The flow equation (with the fluid-mass
storage ``d(phi rho)/dt = phi*drho*dc/dt`` moved to the RHS) turns into a
variable-coefficient Poisson equation for ``h``::

    -div( (rho k/mu) grad h ) = -phi*drho*dc/dt - g * d/dz( (rho k/mu) drho c )  (walls: no-flow)

solved by a batched dense direct solve (variable coefficient ``rho(c)``,
no-flow walls + a single Dirichlet gauge node ``h = 0`` at the top-left cell,
matching the Lua's ``p = 0`` at the top corners). Real pressure is recovered as
``p = h + p_hydro``.

The Darcy face fluxes are solved from ``(c_n, h_n)`` and frozen over one macro
time step while the conservative transport equation is advanced with
explicit-Euler sub-steps (full-upwind advection, central diffusion). This makes
``(c_{n+1}, p_{n+1})`` consistent with the single-step operator
``(c_n, p_n) -> (c_{n+1}, p_{n+1})`` the FNO learns; the buoyancy coupling is
captured *across* macro steps via trajectory sampling.

Boundary conditions (match the Lua):
* top wall: ``c = 1`` on the central source segment, zero-flux Neumann elsewhere;
* bottom wall: ``c = 0`` (Dirichlet);
* left/right walls: zero-flux Neumann (for both c and the flow);
* pressure gauge: ``h = 0`` (=> ``p = 0``) at the top-left cell;
* initial condition: ``p = p_hydro`` (=> ``h = 0``), ``c = 0`` except the source.

The grid *includes* the boundary nodes (shape ``[B, 1, Ny+2, Nx+2]``) so the
FNO sees the walls directly. All field math uses ellipsis (``...``) indexing so
it is independent of the leading batch/channel dimensions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    # Name used by the PhysicsNeMo datapipe registry/logging.
    name: str = "ElderProblem2D"
    # Let framework move tensors to runtime-selected device when needed.
    auto_device: bool = True
    # Disabled: iterative solves / control flow / in-place BC updates are not
    # CUDA-graph friendly.
    cuda_graphs: bool = False
    # Disabled: this generator keeps mutable trajectory state, so no DDP sharding.
    ddp_sharding: bool = False


class ElderProblem2D(Datapipe):
    """Continuously generate 2-D Elder-problem (c, p) solution pairs.

    Parameters
    ----------
    resolution : int
        Number of interior cells along the width (``Nx``). The full grid
        (including walls) is ``(Ny+2) x (Nx+2)``.
    batch_size : int
        Number of ``(c_n, p_n, c_{n+1}, p_{n+1})`` pairs per yielded batch.
    phi, Dm, permeability, viscosity, g, rho_f, drho : float
        Physical parameters (SI): porosity, molecular diffusion, permeability,
        dynamic viscosity, gravitational acceleration (downward, positive),
        fresh-water density, and density contrast ``rho(c) - rho_f``.
    W, H : float
        Domain width and height (``W/H`` sets the aspect ratio; square cells
        ``dx = dy`` are used, so ``Ny = round(Nx * H / W)``).
    source_frac : float
        Fraction of the top width occupied by the ``c = 1`` source segment.
    p_scale : float
        Characteristic pressure-head scale used to normalize ``h`` for the FNO
        input/output. ``None`` (default) => empirical: set to the observed
        ``max|h|`` over the pre-rolled trajectories, so ``h`` is O(1) and its
        MSE balances c's (the physical maximum ``drho * g * H`` is ~20x too
        large and starves the h loss). Set explicitly for exact resume
        reproducibility.
    dt_macro : float
        Time span between ``(c_n, p_n)`` and ``(c_{n+1}, p_{n+1})`` [s].
    flow_sign : float
        Sign of the downward gravity. With the convention here, ``+1.0`` makes
        dense (``c=1``) fluid sink. If fingers rise instead, flip to ``-1.0``
        (the buoyancy-direction unit test reports the correct value).
    substeps : int
        Minimum number of transport sub-steps; increased automatically to
        honor the CFL condition.
    max_substeps : int
        Cap on the auto-increased sub-step count.
    n_trajectories, rollout_steps : int
        Trajectory-sampling parameters. ``n_trajectories`` independent rollouts
        are advanced one macro step at a time; each is reset after
        ``rollout_steps`` steps. Trajectories are pre-rolled to staggered phases
        so a batch spans different stages of the fingering.
    device : str or torch.device
        Device on which to generate data.
    """

    def __init__(
        self,
        resolution: int = 64,
        batch_size: int = 32,
        phi: float = 0.1,
        Dm: float = 3.565e-6,
        permeability: float = 4.845e-13,
        viscosity: float = 1.0e-3,
        g: float = 9.81,
        rho_f: float = 1000.0,
        drho: float = 200.0,
        W: float = 600.0,
        H: float = 150.0,
        source_frac: float = 0.5,
        p_scale: float | None = None,
        dt_macro: float = 10.0 * 24 * 3600.0,
        flow_sign: float = 1.0,
        substeps: int = 4,
        max_substeps: int = 4000,
        n_trajectories: int = 16,
        rollout_steps: int = 128,
        device: str | torch.device = "cuda",
    ):
        super().__init__(meta=MetaData())

        # --- physical parameters (SI) ---------------------------------------
        self.resolution = int(resolution)
        self.batch_size = int(batch_size)
        self.phi = float(phi)
        self.Dm = float(Dm)
        self.permeability = float(permeability)
        self.viscosity = float(viscosity)
        self.g = float(g)
        self.rho_f = float(rho_f)
        self.drho = float(drho)
        self.W = float(W)
        self.H = float(H)
        self.source_frac = float(source_frac)
        self.dt_macro = float(dt_macro)
        self.flow_sign = float(flow_sign)
        self.substeps = int(substeps)
        self.max_substeps = int(max_substeps)
        self.n_trajectories = int(n_trajectories)
        self.rollout_steps = int(rollout_steps)
        self.device = torch.device(device)

        # Derived flow scalars.
        self.k_over_mu = self.permeability / self.viscosity
        # Signed downward gravity (flow_sign flips buoyancy coherently).
        self.gz = self.flow_sign * self.g

        # --- grid (square cells, wall nodes included) -----------------------
        self.Nx = int(self.resolution)            # interior x cells
        self.Ny = max(1, int(round(self.Nx * self.H / self.W)))  # interior y cells
        self.dx = self.W / self.Nx
        self.dy = self.H / self.Ny
        self.Nx_tot = self.Nx + 2
        self.Ny_tot = self.Ny + 2

        # source segment on the top wall (full-grid column indices).
        src_n = max(1, int(round(self.source_frac * self.Nx)))
        self.src_x0 = 1 + (self.Nx - src_n) // 2
        self.src_x1 = self.src_x0 + src_n

        # --- equivalent-freshwater-head hydrostatic reference ----------------
        # p_hydro(z) = rho_f * g * z, with z = row_index * dy (row 0 = top).
        z = torch.arange(self.Ny_tot, dtype=torch.float32) * self.dy
        self.p_hydro = (self.rho_f * self.g * z).view(-1, 1).expand(
            self.Ny_tot, self.Nx_tot
        ).contiguous()
        # Pressure-head scale for network normalization. If not given
        # explicitly it is set *after* the trajectory pre-roll from the observed
        # head amplitude (see the block below the pre-roll loop).
        self.p_scale = float(p_scale) if p_scale is not None else None
        self.p_hydro = self.p_hydro.to(self.device)

        # --- trajectory buffers (staggered phases) --------------------------
        # State is batched over trajectories so the (batched) CG flow solve and
        # the transport integrator advance all trajectories in parallel.
        self._traj_c = torch.zeros(
            self.n_trajectories, 1, self.Ny_tot, self.Nx_tot, device=self.device
        )
        self._traj_h = torch.zeros_like(self._traj_c)
        self._traj_step = torch.zeros(self.n_trajectories, dtype=torch.long, device=self.device)
        # Apply c-BC and solve the consistent initial head for all trajectories.
        self._apply_bc_c(self._traj_c)
        self._traj_h = self._embed_interior(self._flow_solve(self._interior(self._traj_c)))
        # Pre-roll trajectories to staggered phases for batch diversity. Round
        # r advances the subset of trajectories whose target phase exceeds r,
        # so the total work is max_target batched solves (not the sum).
        targets = [(t * self.rollout_steps) // max(1, self.n_trajectories)
                   for t in range(self.n_trajectories)]
        max_target = max(targets) if targets else 0
        for r in range(max_target):
            idx = torch.tensor(
                [t for t in range(self.n_trajectories) if targets[t] > r],
                dtype=torch.long, device=self.device,
            )
            if idx.numel():
                self._advance_subset(idx)

        # --- empirical head scale for network normalization ------------------
        # The naive scale ``drho*g*H`` is the *maximum possible* head (a fully
        # dense column, c=1 everywhere). The real Elder head perturbation
        # ``h = p - p_hydro`` is only ~5% of that, because the domain is mostly
        # fresh (mean c ~ 0.04): normalizing by ``drho*g*H`` crushes h to about
        # [-0.05, 0], making its MSE ~200x smaller than c's, so the joint data
        # loss ignores h and Pred h ends up inaccurate. Scaling by the observed
        # max|h| brings h to O(1) and balances the two channels.
        #
        # The reference solver is deterministic (no RNG), so max|h| over a
        # rollout is a reproducible function of the physics+grid -- this makes
        # p_scale consistent across instances (train, eval, resume). When the
        # staggered pre-roll ran (n_trajectories >= 2) we reuse _traj_h; when it
        # did not (e.g. n_trajectories == 1, used by rollout eval) we run a
        # deterministic calibration rollout so p_scale still matches training.
        if self.p_scale is None:
            h_amp = float(self._traj_h.abs().amax().item())
            if h_amp < 1.0:  # pre-roll did not advance _traj_h (still at IC)
                h_amp = self._calibrate_head_scale()
            floor = 0.01 * (self.drho * self.g * self.H)  # never below 1% of max
            self.p_scale = max(h_amp, floor)

    # ------------------------------------------------------------------
    # grid helpers
    # ------------------------------------------------------------------
    def _interior(self, full: Tensor) -> Tensor:
        """Return the interior ``[..., Ny, Nx]`` slice of a full-grid field."""
        return full[..., 1:-1, 1:-1]

    def _embed_interior(self, interior: Tensor) -> Tensor:
        """Embed an interior field into a full-grid field with no-flow
        (replicate) ghosts and the top-corner pressure gauge set to zero."""
        full = F.pad(interior, (1, 1, 1, 1), mode="replicate")
        full[..., 0, 0] = 0.0
        full[..., 0, -1] = 0.0
        return full

    # ------------------------------------------------------------------
    # variable-coefficient flow solve:  -div(alpha grad h) = f
    #   alpha = rho k / mu ,  f = -g * d/dz(alpha * drho * c)  (walls: no-flow)
    #   gauge: h = 0 at top-left interior cell (Dirichlet)
    # ------------------------------------------------------------------
    def _flow_rhs(self, c_int: Tensor, dc_dt: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        """Return ``(f, alpha)`` interior tensors for the flow solve.

        The flow equation is ``d(phi rho)/dt + div(rho q) = 0`` (the Lua keeps
        the fluid-mass storage ``d(phi rho)/dt = phi*drho*dc/dt`` and drops only
        the pressure-storage ``mass_scale*dp/dt``). In the head gauge this is
        ``M h = f`` with ``f = -phi*drho*dc/dt - g * d/dz(alpha*drho*c)``; the
        storage term ``dc_dt`` (interior, same shape as ``c_int``) is added when
        provided (forward difference over the macro step); ``None`` => 0
        (quasi-static, used for the initial head).
        """
        rho = self.rho_f + self.drho * c_int
        alpha = rho * self.k_over_mu
        # Buoyancy divergence over interior z-faces only (wall faces = 0).
        ac = alpha * self.drho * c_int                       # [..., Ny, Nx]
        face = 0.5 * (ac[..., :-1, :] + ac[..., 1:, :])      # [..., Ny-1, Nx]
        above = torch.cat([face, torch.zeros_like(face[..., :1, :])], dim=-2)
        below = torch.cat([torch.zeros_like(face[..., :1, :]), face], dim=-2)
        f = -self.gz * (above - below) / self.dy
        # Fluid-mass storage term  d(phi rho)/dt = phi*drho*dc/dt  (moved to RHS).
        # The c source/sink BCs change total mass, so this term has a nonzero spatial
        # mean; with no-flow walls + a single pressure gauge that excites the Poisson
        # null space and produces a spurious global head gradient. Project out the
        # uniform (null-space) mode by subtracting the per-sample spatial mean: the
        # removed constant is the gauge-absorbed global pressure, which has no effect on
        # the velocity (only gradients drive flow). Local mass conservation is preserved.
        if dc_dt is not None:
            dc_dev = dc_dt - dc_dt.mean(dim=(-2, -1), keepdim=True)
            f = f - self.phi * self.drho * dc_dev
        f[..., 0, 0] = 0.0                                   # gauge RHS
        return f, alpha

    def _flow_stencil(self, alpha: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Precompute the 5-point stencil face coefficients (``ae, aw, an, as_``)
        and the diagonal of M = -div(alpha grad .) for the no-flow (replicate)
        wall discretization. All outputs share the interior shape."""
        ap = F.pad(alpha, (1, 1, 1, 1), mode="replicate")
        ae = 0.5 * (ap[..., 1:-1, 1:-1] + ap[..., 1:-1, 2:])
        aw = 0.5 * (ap[..., 1:-1, 1:-1] + ap[..., 1:-1, :-2])
        an = 0.5 * (ap[..., 1:-1, 1:-1] + ap[..., 2:, 1:-1])
        as_ = 0.5 * (ap[..., 1:-1, 1:-1] + ap[..., :-2, 1:-1])
        diag = (ae + aw) / self.dx**2 + (an + as_) / self.dy**2
        return ae, aw, an, as_, diag

    def _flow_solve(self, c_int: Tensor, dc_dt: Tensor | None = None) -> Tensor:
        """Solve the flow equation ``M h = f`` for ``h`` (interior, same shape
        as ``c_int``).

        The 5-point variable-coefficient operator ``M = -div(alpha grad .)``
        (no-flow walls + a Dirichlet gauge at the top-left cell) is assembled
        as a batched dense matrix and solved directly with
        :func:`torch.linalg.solve`. For the grid sizes used here
        (``Nx*Ny`` up to ~1024) this is faster than a Python-loop iterative
        solver on both CPU and GPU (no per-iteration host overhead); it scales
        as ``O((Nx*Ny)^3)`` so raise resolution with care.

        ``dc_dt`` is the optional fluid-mass storage rate (see :meth:`_flow_rhs`).

        The solve is done in float64: the SI coefficients ``alpha/dx^2`` are
        ~1e-9, so the matrix condition (~``N^2``) exceeds float32 precision and
        a float32 solve returns an inaccurate head. The result is cast back to
        the input dtype.
        """
        out_dtype = c_int.dtype
        c64 = c_int.double()
        d64 = dc_dt.double() if dc_dt is not None else None
        f, alpha = self._flow_rhs(c64, d64)
        ae, aw, an, as_, _ = self._flow_stencil(alpha)
        B = c64.shape[0]
        Ny, Nx = self.Ny, self.Nx
        N = Ny * Nx
        dev = c64.device

        # Face conductances; zero the wall-face entries (no-flux).
        cE = ae / self.dx**2
        cW = aw / self.dx**2
        cN = an / self.dy**2
        cS = as_ / self.dy**2
        cE[..., -1] = 0.0
        cW[..., 0] = 0.0
        cN[..., -1, :] = 0.0
        cS[..., 0, :] = 0.0
        diag = (cE + cW + cN + cS).reshape(B, N)

        A = torch.diag_embed(diag)
        idx = torch.arange(N, device=dev)
        i_grid = torch.arange(Nx, device=dev).repeat(Ny)     # column index per flat cell
        j_grid = torch.arange(Ny, device=dev).repeat_interleave(Nx)
        cEf = cE.reshape(B, N)
        cWf = cW.reshape(B, N)
        cNf = cN.reshape(B, N)
        cSf = cS.reshape(B, N)

        # Off-diagonal couplings (-coeff at (cell, neighbor)); only where the
        # neighbor lies inside the interior (wall faces already zeroed).
        # NOTE: in ``_flow_stencil`` ``an`` is the +z (south, j+1) face and
        # ``as_`` is the -z (north, j-1) face, so the south neighbor (col +Nx)
        # takes ``cN`` and the north neighbor (col -Nx) takes ``cS``.
        mE = (i_grid < Nx - 1)
        mW = (i_grid > 0)
        mS = (j_grid < Ny - 1)        # +z (downward) neighbor
        mN = (j_grid > 0)             # -z (upward) neighbor
        A[:, idx[mE], idx[mE] + 1] = -cEf[:, mE]
        A[:, idx[mW], idx[mW] - 1] = -cWf[:, mW]
        A[:, idx[mS], idx[mS] + Nx] = -cNf[:, mS]
        A[:, idx[mN], idx[mN] - Nx] = -cSf[:, mN]

        # Dirichlet gauge at the top-left cell (flat index 0).
        A[:, 0, :] = 0.0
        A[:, 0, 0] = 1.0
        b = f.reshape(B, N).clone()
        b[:, 0] = 0.0

        h = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
        return h.reshape(B, 1, Ny, Nx).to(out_dtype)

    # ------------------------------------------------------------------
    # Darcy face fluxes (frozen over one macro step):  rho q at cell faces
    #   rho q = -alpha grad h + alpha * drho * c * (0, gz)
    #   wall faces are omitted (=> zero flux, no-flow).
    # ------------------------------------------------------------------
    def _face_fluxes(self, c_int: Tensor, h_int: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(Fx, Fz)`` interior face fluxes.

        ``Fx`` is the flux through the x-face between cells ``(i, i+1)``
        (shape ``[..., Ny, Nx-1]``); ``Fz`` through the z-face between rows
        ``(j, j+1)`` (shape ``[..., Ny-1, Nx]``).
        """
        rho = self.rho_f + self.drho * c_int
        alpha = rho * self.k_over_mu
        # x-faces (between i and i+1).
        ax = 0.5 * (alpha[..., :, :-1] + alpha[..., :, 1:])
        Fx = -ax * (h_int[..., :, 1:] - h_int[..., :, :-1]) / self.dx
        # z-faces (between j and j+1): rho qz = -alpha h_z + (alpha drho c) gz.
        # Use the cell-product-then-average for the buoyancy face value so the
        # face-flux divergence is exactly consistent with the flow RHS ``f``.
        az = 0.5 * (alpha[..., :-1, :] + alpha[..., 1:, :])
        ac = alpha * self.drho * c_int
        acz = 0.5 * (ac[..., :-1, :] + ac[..., 1:, :])
        Fz = -az * (h_int[..., 1:, :] - h_int[..., :-1, :]) / self.dy + acz * self.gz
        return Fx, Fz

    # ------------------------------------------------------------------
    # Boundary conditions for c (full grid, in-place)
    # ------------------------------------------------------------------
    def _apply_bc_c(self, c: Tensor) -> None:
        """In-place: Neumann mirror on left/right and top-outside-source, then
        Dirichlet ``c = 1`` on the top source segment and ``c = 0`` on the
        bottom wall."""
        # Neumann zero-flux mirrors.
        c[..., 0, : self.src_x0] = c[..., 1, : self.src_x0]
        c[..., 0, self.src_x1 :] = c[..., 1, self.src_x1 :]
        c[..., :, 0] = c[..., :, 1]
        c[..., :, -1] = c[..., :, -2]
        # Dirichlet bottom (c = 0) and top source (c = 1).
        c[..., -1, :] = 0.0
        c[..., 0, self.src_x0 : self.src_x1] = 1.0

    # ------------------------------------------------------------------
    # Reference transport solver (frozen face fluxes, explicit Euler sub-steps)
    # ------------------------------------------------------------------
    def _cfl_dt_max(self, Fx: Tensor, Fz: Tensor) -> float:
        """Largest stable explicit-Euler sub-step for the current face fluxes.

        Uses the max Darcy speed ``|q|`` (face flux / face density) and the
        2-D diffusion stability bound."""
        rho_ref = self.rho_f + self.drho * 1.0
        qmax = 0.0
        if Fx.numel():
            qmax = max(qmax, float(Fx.abs().amax().item()) / rho_ref)
        if Fz.numel():
            qmax = max(qmax, float(Fz.abs().amax().item()) / rho_ref)
        diff_limit = 0.5 / (self.Dm * (1.0 / self.dx**2 + 1.0 / self.dy**2))
        adv_limit = min(self.dx, self.dy) / (qmax + 1.0e-12)
        return min(diff_limit, adv_limit)

    def _integrate(self, c0_full: Tensor, Fx: Tensor, Fz: Tensor) -> Tensor:
        """Advance the conservative transport equation over ``dt_macro`` with
        frozen face fluxes, CFL-auto-increased explicit-Euler sub-steps."""
        n_sub = max(self.substeps, int(np.ceil(self.dt_macro / self._cfl_dt_max(Fx, Fz))))
        n_sub = min(n_sub, self.max_substeps)
        dt = self.dt_macro / n_sub

        c = c0_full.clone()
        for _ in range(n_sub):
            ci = self._interior(c)                            # [..., Ny, Nx]
            rho = self.rho_f + self.drho * ci
            adv = self._advective_divergence(ci, Fx, Fz)      # div(rho q c)
            diff = self._diffusive_divergence(c, rho)        # div(rho phi Dm grad c)
            # Conservative time term: d(phi rho c)/dt = phi (rho_f + 2 drho c) dc/dt.
            denom = self.phi * (self.rho_f + 2.0 * self.drho * ci)
            dc_dt = (diff - adv) / denom
            c[..., 1:-1, 1:-1] = ci + dt * dc_dt
            self._apply_bc_c(c)
        return c

    def _advective_divergence(self, ci: Tensor, Fx: Tensor, Fz: Tensor) -> Tensor:
        """``div(rho q c)`` on the interior with full-upwind c at faces."""
        Ny, Nx = self.Ny, self.Nx
        # x-direction: per-cell east/west face flux + upwind c (walls = 0).
        Fx_east = torch.zeros_like(ci)
        Fx_east[..., :, :-1] = Fx
        Fx_west = torch.zeros_like(ci)
        Fx_west[..., :, 1:] = Fx
        pos_x = (Fx > 0)
        c_east = torch.zeros_like(ci)
        c_east[..., :, :-1] = torch.where(pos_x, ci[..., :, :-1], ci[..., :, 1:])
        c_west = torch.zeros_like(ci)
        c_west[..., :, 1:] = torch.where(pos_x, ci[..., :, :-1], ci[..., :, 1:])
        adv_x = (Fx_east * c_east - Fx_west * c_west) / self.dx
        # z-direction: per-cell south(+z)/north(-z) face flux + upwind c.
        Fz_south = torch.zeros_like(ci)
        Fz_south[..., :-1, :] = Fz
        Fz_north = torch.zeros_like(ci)
        Fz_north[..., 1:, :] = Fz
        pos_z = (Fz > 0)
        c_south = torch.zeros_like(ci)
        c_south[..., :-1, :] = torch.where(pos_z, ci[..., :-1, :], ci[..., 1:, :])
        c_north = torch.zeros_like(ci)
        c_north[..., 1:, :] = torch.where(pos_z, ci[..., :-1, :], ci[..., 1:, :])
        adv_z = (Fz_south * c_south - Fz_north * c_north) / self.dy
        return adv_x + adv_z

    def _diffusive_divergence(self, c_full: Tensor, rho: Tensor) -> Tensor:
        """``div(rho phi Dm grad c)`` on the interior, central, walls = 0."""
        kd = rho * self.phi * self.Dm
        # 修改点：直接使用包含物理边界 c=1 的全网格 c_full
        cp = c_full 
        kdp = F.pad(kd, (1, 1, 1, 1), mode="replicate")
        cc = cp[..., 1:-1, 1:-1]
        ke = 0.5 * (kdp[..., 1:-1, 1:-1] + kdp[..., 1:-1, 2:])
        kw = 0.5 * (kdp[..., 1:-1, 1:-1] + kdp[..., 1:-1, :-2])
        kn = 0.5 * (kdp[..., 1:-1, 1:-1] + kdp[..., 2:, 1:-1])
        ks = 0.5 * (kdp[..., 1:-1, 1:-1] + kdp[..., :-2, 1:-1])
        return (
            (ke * (cp[..., 1:-1, 2:] - cc) - kw * (cc - cp[..., 1:-1, :-2])) / self.dx**2
            + (kn * (cp[..., 2:, 1:-1] - cc) - ks * (cc - cp[..., :-2, 1:-1])) / self.dy**2
        )

    # ------------------------------------------------------------------
    # Trajectory management (strategy B: sample along rollouts)
    # ------------------------------------------------------------------
    def _advance_all(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c0 = self._traj_c.clone()
        h0 = self._traj_h.clone()
        Fx, Fz = self._face_fluxes(self._interior(c0), self._interior(h0))
        c1 = self._integrate(c0, Fx, Fz)
        dc_dt = self._interior(c1 - c0) / self.dt_macro
        h1 = self._embed_interior(self._flow_solve(self._interior(c1), dc_dt))
        
        # 【修改点】：使用 clone() 避免后续 reset 污染返回的 c1, h1
        self._traj_c = c1.clone()
        self._traj_h = h1.clone()
        
        self._traj_step += 1
        reset = self._traj_step >= self.rollout_steps
        if bool(reset.any().item()):
            self._reset_masked(reset)
        p0 = h0 + self.p_hydro
        p1 = h1 + self.p_hydro
        return c0, p0, c1, p1

    def _reset_masked(self, mask: Tensor) -> None:
        """Reset the trajectories selected by the boolean ``mask`` to the
        initial condition (c = 0 + source, h = flow solve)."""
        n = int(mask.sum().item())
        if n == 0:
            return
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        self._traj_c[idx] = 0.0
        self._traj_h[idx] = 0.0
        self._apply_bc_c(self._traj_c[idx])
        self._traj_h[idx] = self._embed_interior(
            self._flow_solve(self._interior(self._traj_c[idx]))
        )
        self._traj_step[idx] = 0

    def _advance_round(self, n_steps: int) -> None:
        """Advance all trajectories by ``n_steps`` macro steps (pre-roll)."""
        for _ in range(n_steps):
            self._advance_all()

    def _advance_subset(self, idx: Tensor) -> None:
        c0 = self._traj_c[idx].clone()
        h0 = self._traj_h[idx].clone()
        Fx, Fz = self._face_fluxes(self._interior(c0), self._interior(h0))
        c1 = self._integrate(c0, Fx, Fz)
        dc_dt = self._interior(c1 - c0) / self.dt_macro
        h1 = self._embed_interior(self._flow_solve(self._interior(c1), dc_dt))
        
        # 【修改点】：同样使用 clone()
        self._traj_c[idx] = c1.clone()
        self._traj_h[idx] = h1.clone()
        
        self._traj_step[idx] += 1

    def _calibrate_head_scale(self) -> float:
        """Deterministic estimate of ``max|h|`` over a full rollout from the IC.

        Used when the staggered pre-roll did not run (e.g. ``n_trajectories ==
        1``), so that ``p_scale`` matches the value training computed. The
        reference solver has no randomness, so this is reproducible across
        instances (train / eval / resume). Returns the running max of ``|h|``
        over ``rollout_steps`` macro steps from the initial condition.
        """
        c = torch.zeros(1, 1, self.Ny_tot, self.Nx_tot, device=self.device)
        h = torch.zeros_like(c)
        self._apply_bc_c(c)
        h = self._embed_interior(self._flow_solve(self._interior(c)))
        h_amp = float(h.abs().amax().item())
        for _ in range(max(1, self.rollout_steps)):
            c_prev = c
            Fx, Fz = self._face_fluxes(self._interior(c), self._interior(h))
            c = self._integrate(c, Fx, Fz)
            dc_dt = self._interior(c - c_prev) / self.dt_macro
            h = self._embed_interior(self._flow_solve(self._interior(c), dc_dt))
            h_amp = max(h_amp, float(h.abs().amax().item()))
        return h_amp

    def generate_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Collect ``batch_size`` samples by advancing trajectories in rounds.

        Each round advances all ``n_trajectories`` in parallel (one batched
        flow solve) and yields that many samples; rounds repeat until the
        batch is full, then the concatenation is trimmed to ``batch_size``.
        """
        c0s, p0s, c1s, p1s, t0s = [], [], [], [], []
        n = self.batch_size
        have = 0
        while have < n:
            # 记录推进前的物理时间 (将秒转换为天)
            t0 = self._traj_step.clone() * self.dt_macro / (24 * 3600.0)
            c0, p0, c1, p1 = self._advance_all()
            c0s.append(c0)
            p0s.append(p0)
            c1s.append(c1)
            p1s.append(p1)
            t0s.append(t0)
            have += c0.shape[0]
        return (
            torch.cat(c0s, dim=0)[:n],
            torch.cat(p0s, dim=0)[:n],
            torch.cat(c1s, dim=0)[:n],
            torch.cat(p1s, dim=0)[:n],
            torch.cat(t0s, dim=0)[:n],
        )

    def __iter__(self) -> Dict[str, Tensor]:
        """Yield batches of ``{c0, p0, c1, p1, t0, dt}`` infinitely.

        Tensor shapes are ``[batch, 1, Ny+2, Nx+2]`` (walls included); ``dt``
        is the macro time step (Python float) used to form the time derivative
        in the PDE residual.
        """
        while True:
            c0, p0, c1, p1, t0 = self.generate_batch()
            yield {
                "c0": c0,
                "p0": p0,
                "c1": c1,
                "p1": p1,
                "t0": t0,
                "dt": self.dt_macro,
            }


    def __len__(self) -> int:
        return sys.maxsize
