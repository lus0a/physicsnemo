# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive rollout evaluation for the trained Elder FNO.

Pure inference / read-only: loads a checkpoint, drives the single-step model
forward N macro steps by feeding its own prediction back in, and compares
against the reference-solver trajectory from the datapipe. Writes:

* ``rollout_error.png``      - per-step RMSE of c and h (log y) vs step.
* ``rollout_field_stepT.png``- True / Pred / |error| fields for c and h at a
                              few selected steps.

Does NOT touch training, weights, or the datapipe used by training.

Run:
    python eval_rollout.py --steps 50 --checkpoint outputs_elder_fno/checkpoints
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils import load_checkpoint

from datapipe import ElderProblem2D
from train_elder_fno import _resolve_fno_modes


def _plot_fields(c_true, c_pred, h_true, h_pred, title, out_path):
    """2x3 True/Pred/|error| comparison for c (row 0) and h (row 1)."""
    c_true, c_pred, h_true, h_pred = (np.asarray(x) for x in (c_true, c_pred, h_true, h_pred))
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.4)
    ax = [[fig.add_subplot(gs[0, c]), fig.add_subplot(gs[1, c])] for c in range(3)]
    ax = [[ax[c][0] for c in range(3)], [ax[c][1] for c in range(3)]]  # ax[row][col]
    fig.suptitle(title, fontsize=15, fontweight="bold")

    titles = [("True c", "Pred c", "|c error|"), ("True h", "Pred h", "|h error|")]
    fields = [(c_true, c_pred, np.abs(c_pred - c_true)),
              (h_true, h_pred, np.abs(h_pred - h_true))]
    for row in range(2):
        true_f = fields[row][0]
        if row == 0:
            row_vmin, row_vmax = 0.0, 1.0
        else:
            row_vmin, row_vmax = float(true_f.min()), float(true_f.max())
        for col in range(3):
            f = fields[row][col]
            if col < 2:
                vmin, vmax = row_vmin, row_vmax
            else:
                vmin, vmax = 0.0, float(f.max())
            im = ax[row][col].imshow(f, origin="upper", vmin=vmin, vmax=vmax)
            ax[row][col].set_title(titles[row][col])
            plt.colorbar(im, ax=ax[row][col])
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", default="outputs_elder_fno/checkpoints",
                   help="checkpoint dir (loads latest) or a .pt file")
    p.add_argument("--steps", type=int, default=50, help="number of macro steps to roll out")
    p.add_argument("--out_dir", default="rollout_eval")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    DistributedManager.initialize()

    # --- datapipe: a single fresh trajectory, no reset during the rollout ---
    N = int(args.steps)
    phy = cfg.physics
    dat = cfg.data
    dp = ElderProblem2D(
        resolution=dat.resolution,
        batch_size=1,
        phi=phy.phi, Dm=phy.Dm, permeability=phy.permeability, viscosity=phy.viscosity,
        g=phy.g, rho_f=phy.rho_f, drho=phy.drho, W=phy.W, H=phy.H,
        source_frac=dat.source_frac, p_scale=phy.get("p_scale", None),
        dt_macro=phy.dt_macro, flow_sign=phy.get("flow_sign", 1.0),
        substeps=dat.substeps, max_substeps=dat.max_substeps,
        n_trajectories=1, rollout_steps=N + 10,   # avoid reset mid-rollout
        device=device,
    )
    p_hydro, p_scale = dp.p_hydro, dp.p_scale
    dt_days = dp.dt_macro / (24 * 3600.0)

    # --- model (architecture must match training) ---
    mdl = cfg.model
    model = FNO(
        in_channels=mdl.in_channels, out_channels=mdl.out_channels,
        decoder_layers=mdl.decoder_layers, decoder_layer_size=mdl.decoder_layer_size,
        dimension=mdl.dimension, latent_channels=mdl.latent_channels,
        num_fno_layers=mdl.num_fno_layers,
        num_fno_modes=_resolve_fno_modes(
            OmegaConf.to_container(mdl, resolve=True)["num_fno_modes"], dp, mdl.padding),
        padding=mdl.padding,
    ).to(device)
    load_checkpoint(path=args.checkpoint, models=model, device=device)
    model.eval()
    print(f"loaded checkpoint from {args.checkpoint}")

    # --- ground-truth trajectory: advance the reference solver N steps ---
    true_c = [None] * (N + 1)
    true_p = [None] * (N + 1)
    for t in range(N):
        c0, p0, c1, p1 = dp._advance_all()
        if t == 0:
            true_c[0], true_p[0] = c0.detach(), p0.detach()
        true_c[t + 1], true_p[t + 1] = c1.detach(), p1.detach()

    # --- model rollout: feed own prediction back in (normalized h space) ---
    cur_c = true_c[0]
    cur_h = (true_p[0] - p_hydro) / p_scale
    pred_c, pred_h = [], []
    with torch.no_grad():
        for t in range(N):
            invar = torch.cat([cur_c, cur_h], dim=1)
            out = model(invar)
            cur_c, cur_h = out[:, 0:1], out[:, 1:2]
            pred_c.append(cur_c.detach())
            pred_h.append(cur_h.detach())

    # --- per-step errors ---
    rmse_c, rmse_h = [], []
    c_min, c_max = 1e9, -1e9
    for t in range(N):
        tc = true_c[t + 1]
        th = (true_p[t + 1] - p_hydro) / p_scale
        rmse_c.append(float(torch.sqrt(((pred_c[t] - tc) ** 2).mean())))
        rmse_h.append(float(torch.sqrt(((pred_h[t] - th) ** 2).mean())))
        c_min = min(c_min, float(pred_c[t].min()))
        c_max = max(c_max, float(pred_c[t].max()))

    # --- error curve ---
    steps = np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(steps, rmse_c, "o-", ms=4, label="RMSE c")
    ax.semilogy(steps, rmse_h, "s-", ms=4, label="RMSE h")
    ax.axhline(0.05, color="r", ls="--", alpha=0.6, label="0.05 divergence threshold")
    ax.set_xlabel(f"rollout step (1 step = {dt_days:.1f} days)")
    ax.set_ylabel("RMSE (log)")
    ax.set_title(f"Autoregressive rollout error vs step ({N} steps)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "rollout_error.png"), dpi=120)
    plt.close(fig)

    # --- field comparisons at a few steps ---
    for t in [1, N // 2, N]:
        if t < 1 or t > N:
            continue
        tc = true_c[t][0, 0].cpu().numpy()
        pc = pred_c[t - 1][0, 0].cpu().numpy()
        th = ((true_p[t] - p_hydro) / p_scale)[0, 0].cpu().numpy()
        ph = pred_h[t - 1][0, 0].cpu().numpy()
        days = t * dt_days
        _plot_fields(tc, pc, th, ph,
                     f"Rollout step {t} (~{days:.0f} days)",
                     os.path.join(args.out_dir, f"rollout_field_step{t:03d}.png"))

    # --- summary ---
    div_c = next((t + 1 for t, r in enumerate(rmse_c) if r > 0.05), None)
    print("\n=== rollout summary ===")
    print(f"steps              : {N}  ({N*dt_days:.0f} days)")
    print(f"RMSE c  step1/last : {rmse_c[0]:.3e} / {rmse_c[-1]:.3e}")
    print(f"RMSE h  step1/last : {rmse_h[0]:.3e} / {rmse_h[-1]:.3e}")
    print(f"pred c range       : [{c_min:.3f}, {c_max:.3f}]  (truth in [0,1])")
    print(f"c diverges (>0.05) : {'step ' + str(div_c) if div_c else 'never within rollout'}")
    print(f"plots written to   : {args.out_dir}/")


if __name__ == "__main__":
    main()
