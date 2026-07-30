# SPDX-License-Identifier: Apache-2.0
"""按 stride (=dt) 拆分 val 单步误差: 每档步长 (10/20/40/80/160d) 的 MSE_c / MSE_h
+ 随物理时间的曲线。复用 FnoEngine, 与推理/训练验证完全一致 (per-pair 真实 dt)。

用法:
  python per_stride_error.py --checkpoint outputs_elder_fno/checkpoints --arch fno
  python per_stride_error.py --checkpoint outputs_elder_ufno/checkpoints --arch ufno
"""
from __future__ import annotations
import argparse, os
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from fno_infer_core import FnoEngine
from vtu_dataset import VtuElderDataset
from train_elder_fno import build_invar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="outputs_elder_fno/checkpoints")
    ap.add_argument("--arch", default=None, help="fno|ufno (覆盖 config)")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--out_dir", default="per_stride_error")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    phy = cfg.physics
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine = FnoEngine.load(args.config, args.checkpoint, device=str(dev), arch=args.arch)

    dp = VtuElderDataset(
        cfg.data.get("train_dir", "DataSet"), batch_size=1, device=dev,
        phi=phy.phi, Dm=phy.Dm, permeability=phy.permeability, viscosity=phy.viscosity,
        g=phy.g, rho_f=phy.rho_f, drho=phy.drho, W=phy.W, H=phy.H, dt_macro=phy.dt_macro,
        flow_sign=phy.get("flow_sign", 1.0), split=args.split,
        val_frac=float(cfg.data.get("val_frac", 0.2)),
        n_val_blocks=int(cfg.data.get("n_val_blocks", 8)),
        val_gap=int(cfg.data.get("val_gap", 2)),
        file_stride=int(cfg.data.get("file_stride", 1)),
        file_strides=list(cfg.data.get("file_strides",
                                       [int(cfg.data.get("file_stride", 1))])),
    )
    ph, ps = engine.p_hydro, engine.p_scale
    resid, dt_aware, dt_ref = engine.residual, engine.dt_aware, engine.dt_ref
    fi = dp.file_interval  # 原始文件间隔 (秒, ~10d)

    groups = defaultdict(list)
    for j, s in enumerate(dp.pair_strides):
        groups[int(s)].append(j)

    rows = []  # (stride, days, t_days, mse_c, mse_h, old_c, old_h)
    with torch.no_grad():
        for s in sorted(groups):
            js = groups[s]
            k = torch.as_tensor([dp.pair_indices[j] for j in js], device=dev, dtype=torch.long)
            c0 = dp.data[k, 0:1]; P0 = dp.data[k, 1:2]
            c1 = dp.data[k + s, 0:1]; P1 = dp.data[k + s, 1:2]
            h0 = (P0 - ph) / ps; h1 = (P1 - ph) / ps
            dt = s * fi  # 这一档的真实 dt (秒)
            invar = build_invar(c0, h0, dt, dt_aware, dt_ref)
            raw = engine.model(invar)
            pc = c0 + raw[:, 0:1] if resid else raw[:, 0:1]
            ph_ = h0 + raw[:, 1:2] if resid else raw[:, 1:2]
            mse_c = ((pc - c1) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            mse_h = ((ph_ - h1) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            old_c = ((c0 - c1) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            old_h = ((h0 - h1) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            t = dp._times_days[k + s].cpu().numpy()
            for ii in range(len(js)):
                rows.append((s, s * fi / 86400.0, float(t[ii]), float(mse_c[ii]),
                             float(mse_h[ii]), float(old_c[ii]), float(old_h[ii])))

    print(f"\n=== {args.checkpoint}  arch={engine.arch}  split={args.split}  "
          f"{len(rows)} pairs ===")
    print(f"{'stride':>6} {'days':>6} {'n':>5} {'MSE_c':>11} {'MSE_h':>11} "
          f"{'old_c':>11} {'old_h':>11} {'AI/old(c)':>9}")
    for s in sorted(groups):
        r = [x for x in rows if x[0] == s]
        mc = np.mean([x[3] for x in r]); mh = np.mean([x[4] for x in r])
        oc = np.mean([x[5] for x in r]); oh = np.mean([x[6] for x in r])
        print(f"{s:>6} {s*fi/86400:>6.0f} {len(r):>5} {mc:>11.2e} {mh:>11.2e} "
              f"{oc:>11.2e} {oh:>11.2e} {mc/oc:>9.2f}")

    # 存 + 画 (按 stride 着色)
    np.savez(os.path.join(args.out_dir, "per_stride.npz"),
             **{f"s{s}": np.array([(x[2], x[3], x[4], x[5], x[6])
                                  for x in sorted([r for r in rows if r[0] == s],
                                                  key=lambda z: z[2])])
                for s in sorted(groups)})
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for col, (name, idx) in enumerate([("MSE_c (concentration)", 3), ("MSE_h (head)", 4)]):
        for s in sorted(groups):
            r = sorted([x for x in rows if x[0] == s], key=lambda z: z[2])
            ax[col].semilogy([x[2] for x in r], [x[idx] for x in r], "o-", ms=3,
                             label=f"{s*fi/86400:.0f}d (stride {s}, n={len(r)})")
        ax[col].set_title(name); ax[col].set_xlabel("target snapshot time (days)")
        ax[col].set_ylabel("MSE (log)"); ax[col].grid(True, which="both", alpha=0.3)
        ax[col].legend(fontsize=7, loc="best")
    fig.suptitle(f"{args.checkpoint}  arch={engine.arch}  per-stride val single-step error",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(args.out_dir, "per_stride.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\nwritten {args.out_dir}/per_stride.png + per_stride.npz")


if __name__ == "__main__":
    main()
