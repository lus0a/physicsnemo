# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026
# SPDX-License-Identifier: Apache-2.0
"""Elder FV residual matching unisolver ``Elder::FormFunction`` (structured 2D).

Discrete residual is **per cell volume**, same assembly as C++:

* accum (flow):  ``φ * drho * (c - c_old) * rec_dt``  with ``rec_dt = 1/dt``
* accum (trans): ``φ * ρ * (c - c_old) * rec_dt``
* diffusion: face-harmonic φ, arithmetic ρ, interior + Dirichlet boundary faces
* advection: interior faces only, ``qA = T*((P_l-P_r) + ρ_f * g·(x_r-x_l))``,
  mass flux ``ρ_f*qA``, upwind ``c``, assemble ``±flux/V``
* pin: top-left & top-right cells replace flow residual by ``P - p_gauge``

Grid layout (same as ``vtu_dataset`` / C++ FNO buffers):
``c, P`` shaped ``[B, 1, Nz, Nx]`` or ``[B, Nz, Nx]``, **iz=0 is top** (min depth z).

All ops are pure torch → differentiable w.r.t. ``c, P`` for FNO training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class ElderPhysics:
    phi: float = 0.1
    perm: float = 4.845e-13
    visc: float = 1.0e-3
    Dm: float = 3.565e-6
    rho_f: float = 1000.0
    drho: float = 200.0
    g: float = 9.81
    W: float = 600.0
    H: float = 150.0
    # extruded y-thickness of the single-layer mesh (elder.cpgrid dy = 1 m)
    dy: float = 1.0
    # BC (match Elder JSON / elder_Shuai.lua)
    c_top: float = 1.0
    c_bottom: float = 0.0
    x_src_min: float = 150.0
    x_src_max: float = 450.0
    p_gauge: float = 0.0
    pin_enable: bool = True

    def density(self, c: Tensor) -> Tensor:
        return self.rho_f + self.drho * c


def _as_B1HW(x: Tensor) -> Tensor:
    """Ensure shape [B, 1, Nz, Nx]."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError(f"expected channel dim 1, got {tuple(x.shape)}")
        return x
    raise ValueError(f"unsupported shape {tuple(x.shape)}")


def grid_spacings(Nz: int, Nx: int, phy: ElderPhysics) -> Tuple[float, float, float]:
    """Cell sizes dx (x), dz (depth), dy (thickness)."""
    dx = phy.W / float(Nx)
    dz = phy.H / float(Nz)
    return dx, dz, phy.dy


def form_function_elder(
    c: Tensor,
    P: Tensor,
    c_old: Tensor,
    P_old: Tensor,
    dt: Union[float, Tensor],
    phy: Optional[ElderPhysics] = None,
    *,
    return_terms: bool = False,
) -> Union[Tuple[Tensor, Tensor], Dict[str, Tensor]]:
    """Compute per-volume residual fields matching unisolver Elder FormFunction.

    Parameters
    ----------
    c, P : predicted state at t^{n+1}, shape [B,1,Nz,Nx] or [B,Nz,Nx]
    c_old, P_old : state at t^n (P_old unused in accum but kept for API symmetry)
    dt : macro step [s] (>0). Internally uses rec_dt = 1/dt like unisolver.
    phy : physical / BC parameters
    return_terms : if True, also return accum/diff/adv splits (diagnostic)

    Returns
    -------
    F_p, F_c : residual [B,1,Nz,Nx]  (flow / transport rows)
    or dict with keys F_p, F_c, accum_p, accum_c, diff_c, adv_p, adv_c
    """
    phy = phy or ElderPhysics()
    c = _as_B1HW(c)
    P = _as_B1HW(P)
    c_old = _as_B1HW(c_old)
    # P_old unused in Elder accum (ρ independent of P) but accept for API
    _ = P_old

    B, _, Nz, Nx = c.shape
    device, dtype = c.device, c.dtype
    dx, dz, dy = grid_spacings(Nz, Nx, phy)
    V = dx * dy * dz
    inv_V = 1.0 / V
    rec_dt = 1.0 / dt if not torch.is_tensor(dt) else 1.0 / dt

    # cell-center x (for top Dirichlet source strip)
    ix = torch.arange(Nx, device=device, dtype=dtype)
    x_c = (ix + 0.5) * dx  # [Nx]
    top_src = (x_c >= phy.x_src_min) & (x_c <= phy.x_src_max)  # [Nx]

    rho = phy.density(c)
    rho_old_term = phy.density(c)  # for transport accum uses current ρ(c)

    # ----- accumulation (all cells) -----
    dc = c - c_old
    accum_p = phy.phi * phy.drho * dc * rec_dt
    accum_c = phy.phi * rho_old_term * dc * rec_dt

    F_p = accum_p.clone()
    F_c = accum_c.clone()
    diff_c = torch.zeros_like(F_c)
    adv_p = torch.zeros_like(F_p)
    adv_c = torch.zeros_like(F_c)

    # ----- diffusion: interior faces -----
    # x-faces between (:, j) and (:, j+1)
    # poro_f harmonic; rho_f arithmetic; coef = poro_f*Dm*rho_f*(area/dist)/V
    # contribution to cell: -coef*(c_nb - c_loc)  [matches GetT_Cell_Diffusion]
    area_x = dy * dz
    dist_x = dx
    area_z = dx * dy
    dist_z = dz
    poro = phy.phi
    # uniform rock → harmonic = poro
    poro_f = poro

    # x-interior faces
    c_l = c[:, :, :, :-1]
    c_r = c[:, :, :, 1:]
    rho_f_x = 0.5 * (phy.density(c_l) + phy.density(c_r))
    coef_x = poro_f * phy.Dm * rho_f_x * (area_x / dist_x) * inv_V
    # at left cell loc, nb=right: -coef*(c_r - c_l)
    # at right cell loc, nb=left: -coef*(c_l - c_r) = +coef*(c_r - c_l)
    dflux_x = coef_x * (c_r - c_l)
    diff_c[:, :, :, :-1] = diff_c[:, :, :, :-1] - dflux_x
    diff_c[:, :, :, 1:] = diff_c[:, :, :, 1:] + dflux_x

    # z-interior faces (l = top / smaller iz, r = bottom / larger iz)
    c_t = c[:, :, :-1, :]
    c_b = c[:, :, 1:, :]
    rho_f_z = 0.5 * (phy.density(c_t) + phy.density(c_b))
    coef_z = poro_f * phy.Dm * rho_f_z * (area_z / dist_z) * inv_V
    dflux_z = coef_z * (c_b - c_t)
    diff_c[:, :, :-1, :] = diff_c[:, :, :-1, :] - dflux_z
    diff_c[:, :, 1:, :] = diff_c[:, :, 1:, :] + dflux_z

    # ----- diffusion: boundary Dirichlet faces -----
    # top faces of iz=0: dist to face = dz/2, area = area_z
    # only where top_src
    dist_bc = 0.5 * dz
    coef_top = poro * phy.Dm * phy.density(c[:, :, 0:1, :]) * (area_z / dist_bc) * inv_V
    c_bc_top = torch.where(
        top_src.view(1, 1, 1, Nx),
        torch.full((), phy.c_top, device=device, dtype=dtype),
        c[:, :, 0:1, :],  # no Dirichlet → zero flux ⇒ skip; use c so (c_bc-c)=0
    )
    # only apply where top_src
    mask_top = top_src.view(1, 1, 1, Nx).to(dtype)
    diff_c[:, :, 0:1, :] = diff_c[:, :, 0:1, :] - mask_top * coef_top * (
        c_bc_top - c[:, :, 0:1, :]
    )

    # bottom faces of iz=Nz-1: c_bc = c_bottom everywhere
    coef_bot = poro * phy.Dm * phy.density(c[:, :, -1:, :]) * (area_z / dist_bc) * inv_V
    c_bc_bot = torch.full_like(c[:, :, -1:, :], phy.c_bottom)
    diff_c[:, :, -1:, :] = diff_c[:, :, -1:, :] - coef_bot * (
        c_bc_bot - c[:, :, -1:, :]
    )

    F_c = F_c + diff_c

    # ----- advection: interior faces (GetT_RR_Flux) -----
    # trans = perm * area / (visc * dist)
    # qA = trans * ((P_l - P_r) + rho_f * gdot)
    # mass = rho_f * qA; adv = mass * c_up
    # assemble: F_l += flux/V, F_r -= flux/V; pin skips flow row
    trans_x = phy.perm * area_x / (phy.visc * dist_x)
    trans_z = phy.perm * area_z / (phy.visc * dist_z)

    # x-faces: gdot = 0
    P_l = P[:, :, :, :-1]
    P_r = P[:, :, :, 1:]
    c_l = c[:, :, :, :-1]
    c_r = c[:, :, :, 1:]
    rho_f_x = 0.5 * (phy.density(c_l) + phy.density(c_r))
    qA_x = trans_x * ((P_l - P_r) + rho_f_x * 0.0)
    mass_x = rho_f_x * qA_x
    c_up_x = torch.where(qA_x > 0, c_l, c_r)
    adv_flux_x = mass_x * c_up_x
    inv = inv_V
    adv_p[:, :, :, :-1] = adv_p[:, :, :, :-1] + mass_x * inv
    adv_p[:, :, :, 1:] = adv_p[:, :, :, 1:] - mass_x * inv
    adv_c[:, :, :, :-1] = adv_c[:, :, :, :-1] + adv_flux_x * inv
    adv_c[:, :, :, 1:] = adv_c[:, :, :, 1:] - adv_flux_x * inv

    # z-faces: loc_l = top (iz), loc_r = bottom (iz+1)
    # gdot = g * (z_r - z_l) = g * dz  (z increases downward)
    gdot_z = phy.g * dz
    P_t = P[:, :, :-1, :]
    P_b = P[:, :, 1:, :]
    c_t = c[:, :, :-1, :]
    c_b = c[:, :, 1:, :]
    rho_f_z = 0.5 * (phy.density(c_t) + phy.density(c_b))
    qA_z = trans_z * ((P_t - P_b) + rho_f_z * gdot_z)
    mass_z = rho_f_z * qA_z
    c_up_z = torch.where(qA_z > 0, c_t, c_b)
    adv_flux_z = mass_z * c_up_z
    adv_p[:, :, :-1, :] = adv_p[:, :, :-1, :] + mass_z * inv
    adv_p[:, :, 1:, :] = adv_p[:, :, 1:, :] - mass_z * inv
    adv_c[:, :, :-1, :] = adv_c[:, :, :-1, :] + adv_flux_z * inv
    adv_c[:, :, 1:, :] = adv_c[:, :, 1:, :] - adv_flux_z * inv

    F_p = F_p + adv_p
    F_c = F_c + adv_c

    # ----- pressure pin: top-left & top-right cells -----
    if phy.pin_enable:
        # overwrite flow residual (do not add adv on those cells for flow —
        # C++ skips adv on flow row for pinned; we zero then set)
        # Easiest: recompute F_p at pins as P - p_gauge, discard flow accum+adv
        F_p = F_p.clone()
        F_p[:, :, 0, 0] = P[:, :, 0, 0] - phy.p_gauge
        F_p[:, :, 0, -1] = P[:, :, 0, -1] - phy.p_gauge
        # C++ never adds adv to flow at pin; we already added then overwrote — OK
        # But accum was also overwritten — matches IsPinned covering flow row

    if return_terms:
        return {
            "F_p": F_p,
            "F_c": F_c,
            "accum_p": accum_p,
            "accum_c": accum_c,
            "diff_c": diff_c,
            "adv_p": adv_p,
            "adv_c": adv_c,
        }
    return F_p, F_c


def residual_l2_norms(
    F_p: Tensor, F_c: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Global L2 norms matching unisolver ComputeResidualNorm / Newton total.

    Returns (norm_p, norm_c, norm_tot) as scalars (mean over batch if B>1:
    each is L2 over space, then mean over batch).
    """
    # sum over spatial dims, keep batch
    sp = tuple(range(1, F_p.ndim))
    n_p = torch.sqrt((F_p * F_p).sum(dim=sp) + 1e-300)
    n_c = torch.sqrt((F_c * F_c).sum(dim=sp) + 1e-300)
    n_t = torch.sqrt(n_p * n_p + n_c * n_c)
    return n_p.mean(), n_c.mean(), n_t.mean()


def residual_loss_elder(
    c: Tensor,
    P: Tensor,
    c_old: Tensor,
    P_old: Tensor,
    dt: Union[float, Tensor],
    phy: Optional[ElderPhysics] = None,
    *,
    w_c: float = 1.0,
    reduction: str = "l2",
) -> Tensor:
    """Scalar loss for training: differentiable w.r.t. c, P.

    reduction:
      * ``l2``: sqrt(mean F_p^2 + mean F_c^2) style → use mean of squares
        actually: (||F_p||_2^2 + w_c ||F_c||_2^2) / N  averaged over batch
      * ``mean_sq``: mean(F_p^2) + w_c mean(F_c^2)
    """
    F_p, F_c = form_function_elder(c, P, c_old, P_old, dt, phy)
    if reduction == "mean_sq":
        return (F_p * F_p).mean() + w_c * (F_c * F_c).mean()
    # default: match Newton-ish total energy / Ncells
    sp = tuple(range(1, F_p.ndim))
    ncells = 1
    for d in sp:
        ncells *= F_p.shape[d]
    loss = ((F_p * F_p).sum(dim=sp) + w_c * (F_c * F_c).sum(dim=sp)) / ncells
    return loss.mean()


def compare_to_previous(
    c_pred: Tensor,
    P_pred: Tensor,
    c_old: Tensor,
    P_old: Tensor,
    dt: Union[float, Tensor],
    phy: Optional[ElderPhysics] = None,
) -> Dict[str, float]:
    """||F(pred)|| vs ||F(old)|| — whether FNO beats previous-state guess."""
    Fp, Fc = form_function_elder(c_pred, P_pred, c_old, P_old, dt, phy)
    np_, nc, nt = residual_l2_norms(Fp, Fc)
    Fp0, Fc0 = form_function_elder(c_old, P_old, c_old, P_old, dt, phy)
    # For old as guess of next step: FormFunction(u_n, u_n) — C++ no-AI start
    # uses field_=field_old_=u_n, so same
    np0, nc0, nt0 = residual_l2_norms(Fp0, Fc0)
    return {
        "norm_p_pred": float(np_.detach()),
        "norm_c_pred": float(nc.detach()),
        "norm_tot_pred": float(nt.detach()),
        "norm_p_prev": float(np0.detach()),
        "norm_c_prev": float(nc0.detach()),
        "norm_tot_prev": float(nt0.detach()),
        "ratio_tot": float((nt / (nt0 + 1e-30)).detach()),
    }
