# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026
# SPDX-License-Identifier: Apache-2.0
"""Verify torch FV residual vs unisolver-produced snapshots.

Uses consecutive VTU pairs (c0,P0)->(c1,P1) from DataSet:
* ||F(c1,P1; c0, dt)|| should be *small* if residual matches unisolver
  (true next state nearly satisfies F=0).
* ||F(c0,P0; c0, dt)|| is the no-AI initial residual (previous state as guess).
* Optional FNO prediction residual for comparison.

Usage:
  python verify_elder_residual_fv.py
  python verify_elder_residual_fv.py --checkpoint outputs_phys_w1e-4/checkpoints
  python verify_elder_residual_fv.py --indices 0,3,6,12
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from elder_residual_fv import (
    ElderPhysics,
    compare_to_previous,
    form_function_elder,
    residual_l2_norms,
)


def _load_dataset(cfg, device):
    from vtu_dataset import VtuElderDataset

    phy = cfg.physics
    dp = VtuElderDataset(
        cfg.data.train_dir,
        1,
        device,
        phy.phi,
        phy.Dm,
        phy.permeability,
        phy.viscosity,
        phy.g,
        phy.rho_f,
        phy.drho,
        phy.W,
        phy.H,
        phy.dt_macro,
        flow_sign=float(phy.get("flow_sign", 1.0)),
        file_stride=int(cfg.data.get("file_stride", 1)),
    )
    return dp


def _maybe_fno(cfg, dp, checkpoint, device):
    if not checkpoint:
        return None, False
    import glob

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.models.fno import FNO

    from train_elder_fno import _resolve_fno_modes, _resolve_in_channels, build_invar

    DistributedManager.initialize()
    mdl = cfg.model
    modes = _resolve_fno_modes(
        OmegaConf.to_container(mdl, resolve=True)["num_fno_modes"], dp, mdl.padding
    )
    model = FNO(
        in_channels=_resolve_in_channels(mdl),
        out_channels=mdl.out_channels,
        decoder_layers=mdl.decoder_layers,
        decoder_layer_size=mdl.decoder_layer_size,
        dimension=mdl.dimension,
        latent_channels=mdl.latent_channels,
        num_fno_layers=mdl.num_fno_layers,
        num_fno_modes=modes,
        padding=mdl.padding,
    ).to(device)
    mdlus = sorted(glob.glob(os.path.join(checkpoint, "*.mdlus")))
    if mdlus:
        def _ep(f):
            try:
                return int(os.path.basename(f).rsplit(".", 2)[-2])
            except ValueError:
                return 0
        mdlus.sort(key=_ep)
        model.load(mdlus[-1])
        print(f"loaded {mdlus[-1]}")
    else:
        from physicsnemo.utils import load_checkpoint

        load_checkpoint(path=checkpoint, models=model, device=device)
    model.eval()
    return model, bool(cfg.training.get("residual", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default=None, help="optional FNO dir for pred residual")
    ap.add_argument("--indices", default="0,3,6,12,30", help="snapshot indices k (pair k->k+stride)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    dp = _load_dataset(cfg, device)
    phy_cfg = cfg.physics
    phy = ElderPhysics(
        phi=float(phy_cfg.phi),
        perm=float(phy_cfg.permeability),
        visc=float(phy_cfg.viscosity),
        Dm=float(phy_cfg.Dm),
        rho_f=float(phy_cfg.rho_f),
        drho=float(phy_cfg.drho),
        g=float(phy_cfg.g),
        W=float(phy_cfg.W),
        H=float(phy_cfg.H),
    )
    dt = float(dp.dt_macro)
    stride = int(dp.file_stride)
    print(
        f"grid Nz={dp.Ny_tot} Nx={dp.Nx_tot} dx={dp.dx:.6g} "
        f"dt={dt:.6g}s ({dt/86400:.1f}d) stride={stride} n_files={dp.n_files}"
    )
    print(f"physics: phi={phy.phi} k={phy.perm} Dm={phy.Dm} drho={phy.drho}")

    model, use_res = _maybe_fno(cfg, dp, args.checkpoint, device)
    p_hydro = dp.p_hydro  # [Nz,Nx] or broadcastable
    p_scale = float(dp.p_scale)

    indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    data = dp.data  # [N,2,Nz,Nx]

    print(
        f"\n{'idx':>4} {'t0_d':>7} | {'||F(true)||':>12} {'||F(prev)||':>12} "
        f"{'||F(fno)||':>12} | true/prev  fno/prev"
    )
    print("-" * 90)

    for i in indices:
        j = i + stride
        if j >= data.shape[0]:
            print(f"{i}: skip (j={j} out of range)")
            continue
        c0 = data[i : i + 1, 0:1].float()
        P0 = data[i : i + 1, 1:2].float()
        c1 = data[j : j + 1, 0:1].float()
        P1 = data[j : j + 1, 1:2].float()

        # true next state residual (should be small if FV matches unisolver)
        Ft_p, Ft_c = form_function_elder(c1, P1, c0, P0, dt, phy)
        _, _, nt_true = residual_l2_norms(Ft_p, Ft_c)

        # previous-state guess (no AI)
        stats_prev = compare_to_previous(c0, P0, c0, P0, dt, phy)
        # compare_to_previous(pred=old) gives F(old; old) — same as no-AI
        nt_prev = stats_prev["norm_tot_prev"]

        nt_fno = float("nan")
        if model is not None:
            with torch.no_grad():
                h0 = (P0 - p_hydro) / p_scale
                invar = build_invar(c0, h0, dt,
                                    bool(cfg.model.get("dt_channel", False)),
                                    float(cfg.model.get("dt_ref_s", 2.592e6)))
                raw = model(invar)
                if use_res:
                    c_p = c0 + raw[:, 0:1]
                    h_p = h0 + raw[:, 1:2]
                else:
                    c_p, h_p = raw[:, 0:1], raw[:, 1:2]
                P_p = h_p * p_scale + p_hydro
            Fp, Fc = form_function_elder(c_p, P_p, c0, P0, dt, phy)
            _, _, nt_fno_t = residual_l2_norms(Fp, Fc)
            nt_fno = float(nt_fno_t)

        t0 = i * (dp.file_interval / 86400.0)
        r_tp = float(nt_true) / (nt_prev + 1e-30)
        r_fp = nt_fno / (nt_prev + 1e-30) if nt_fno == nt_fno else float("nan")
        print(
            f"{i:4d} {t0:7.1f} | {float(nt_true):12.4e} {nt_prev:12.4e} "
            f"{nt_fno:12.4e} | {r_tp:8.3e}  {r_fp:8.3e}"
        )

        # component breakdown for first index
        if i == indices[0]:
            np_t, nc_t, _ = residual_l2_norms(Ft_p, Ft_c)
            print(
                f"      true components: ResNorm[0]={float(np_t):.4e} "
                f"ResNorm[1]={float(nc_t):.4e}"
            )
            terms = form_function_elder(c1, P1, c0, P0, dt, phy, return_terms=True)
            for k in ("accum_p", "accum_c", "diff_c", "adv_p", "adv_c"):
                t = terms[k]
                n = float(torch.sqrt((t * t).sum()))
                print(f"      true Fsplit {k}: L2={n:.4e}")

    print(
        "\nInterpretation:\n"
        "  ||F(true)|| << ||F(prev)||  → FV residual consistent with unisolver data.\n"
        "  ||F(fno)||  <  ||F(prev)||  → FNO better Newton initial guess (this metric).\n"
        "  If ||F(true)|| is large, check dx/dz, rec_dt=1/dt, pin/BC, or face signs."
    )


if __name__ == "__main__":
    # run from elder_fno directory
    if not os.path.exists("config.yaml") and os.path.exists(
        "/mnt/c/Users/lushuai/physicsnemo/examples/cfd/elder_fno/config.yaml"
    ):
        os.chdir("/mnt/c/Users/lushuai/physicsnemo/examples/cfd/elder_fno")
    main()
