# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""逐对单步误差: 把 val 图里那个平均 AI MSE 拆成每个相邻快照对一条点。

对每个 pair k = 快照 k -> 快照 k+file_stride, 用训练好的 FNO 做一次单步前向
(与 validation_step 完全一致: h=(P-p_hydro)/p_scale 归一化, 残差模式则重建),
单独算这一步的 MSE_c / MSE_h, 以及 "不变基线" old MSE (直接拿输入当预测)。
画成随物理时间变化的曲线 (log y), 并存 .npz / .csv。

纯推理 / 只读: 不碰训练、权重、训练用的 datapipe。

用法:
    python per_step_error.py --checkpoint outputs_baseline30days/checkpoints
    python per_step_error.py --checkpoint outputs_baseline10days/checkpoints --split val
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

from vtu_dataset import VtuElderDataset
from train_elder_fno import _resolve_fno_modes


def _build_model(cfg, dp, checkpoint, device):
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
    load_checkpoint(path=checkpoint, models=model, device=device)   # 自动取目录里最新 ckpt
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", default="outputs_baseline30days/checkpoints",
                   help="checkpoint 目录 (取最新) 或 .pt 文件")
    p.add_argument("--train_dir", default=None,
                   help="VTU 数据目录 (默认 cfg.data.train_dir)")
    p.add_argument("--split", default="all", choices=["all", "train", "val"],
                   help="算哪些配对: all=全部相邻对; val/train=只算验证/训练集配对")
    p.add_argument("--out_dir", default="per_step_error")
    p.add_argument("--chunk", type=int, default=64, help="一次前向的配对数 (显存控制)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    phy = cfg.physics
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    DistributedManager.initialize()                     # 单进程也无妨; load_checkpoint 需要

    # --- 数据集 (split 控制算哪些配对; file_stride 须与训练一致) ---
    dp = VtuElderDataset(
        args.train_dir or cfg.data.get("train_dir", "DataSet"),
        batch_size=1, device=device,
        phi=phy.phi, Dm=phy.Dm, permeability=phy.permeability, viscosity=phy.viscosity,
        g=phy.g, rho_f=phy.rho_f, drho=phy.drho, W=phy.W, H=phy.H, dt_macro=phy.dt_macro,
        flow_sign=phy.get("flow_sign", 1.0),
        split=args.split,
        val_frac=float(cfg.data.get("val_frac", 0.2)),
        n_val_blocks=int(cfg.data.get("n_val_blocks", 8)),
        val_gap=int(cfg.data.get("val_gap", 2)),
        file_stride=int(cfg.data.get("file_stride", 1)),
    )
    p_hydro, p_scale = dp.p_hydro, dp.p_scale           # 与训练同 (p_scale 由数据自动算, 全程一个标量)
    s = dp.file_stride
    step_days = dp.dt_macro / 86400.0                   # 每个 macro 步折合多少天 (画图用)

    model = _build_model(cfg, dp, args.checkpoint, device)
    use_residual = bool(cfg.training.get("residual", False))
    print(f"loaded checkpoint from {args.checkpoint} | {step_days:.0f}-day/step | "
          f"residual={use_residual} | split={args.split} -> {len(dp.pair_indices)} pairs")

    # --- 逐 chunk 前向, 累计每个 pair 的 MSE ---
    idx = dp.pair_indices                                # [P] 起始快照下标 k (pair = k -> k+s)
    times, mse_c, mse_h, old_c, old_h = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), args.chunk):
            k = torch.as_tensor(idx[i:i + args.chunk], device=device, dtype=torch.long)
            c0 = dp.data[k, 0:1]                         # [B,1,Ny,Nx]
            p0 = dp.data[k, 1:2]
            c1 = dp.data[k + s, 0:1]                     # 真值浓度 (目标)
            p1 = dp.data[k + s, 1:2]
            h0 = (p0 - p_hydro) / p_scale                # 归一化水头
            h1 = (p1 - p_hydro) / p_scale
            invar = torch.cat([c0, h0], dim=1)
            raw = model(invar)                           # 单步前向 (直接值 或 残差模式下的增量)
            if use_residual:                             # c_{n+1}=c_n+Δc, h_{n+1}=h_n+Δh
                pred_c = c0 + raw[:, 0:1]
                pred_h = h0 + raw[:, 1:2]
            else:
                pred_c, pred_h = raw[:, 0:1], raw[:, 1:2]
            # 每对一条 MSE: 在 [1,Ny,Nx] 全网格 (含边界 cell) 上取均值, 与 validation_step 一致
            times.append((dp._times_days[k + s]).cpu().numpy())           # x = 目标快照时间 (天)
            mse_c.append(((pred_c - c1) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
            mse_h.append(((pred_h - h1) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
            old_c.append(((c0 - c1) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())   # 不变基线
            old_h.append(((h0 - h1) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())

    times = np.concatenate(times)
    mse_c, mse_h = np.concatenate(mse_c), np.concatenate(mse_h)
    old_c, old_h = np.concatenate(old_c), np.concatenate(old_h)
    order = np.argsort(times)                            # 按时间排序 (idx 未必时间有序)
    times, mse_c, mse_h, old_c, old_h = (
        a[order] for a in (times, mse_c, mse_h, old_c, old_h))

    # --- 存 .npz / .csv ---
    np.savez(os.path.join(args.out_dir, "per_step_error.npz"),
             time_days=times, mse_c=mse_c, mse_h=mse_h, old_c=old_c, old_h=old_h,
             step_days=np.float32(step_days))
    rows = np.stack([times, mse_c, mse_h, old_c, old_h], axis=1)
    np.savetxt(os.path.join(args.out_dir, "per_step_error.csv"), rows,
               delimiter=",", header="time_days,mse_c,mse_h,old_c,old_h", comments="")

    # --- 画图: 左 c / 右 h, 各自叠 old 基线; 水平虚线 = 全程平均 (=val 图里那个标量) ---
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    for j, (name, ai, ol) in enumerate([("c (concentration, [0,1])", mse_c, old_c),
                                        ("h (normalized head, dimensionless)", mse_h, old_h)]):
        a = ax[j]
        a.semilogy(times, ai, "o-", ms=3, label="AI MSE")
        a.semilogy(times, ol, "x--", ms=3, alpha=0.7, label="old MSE (persistence)")
        a.axhline(ai.mean(), color="C0", ls=":", alpha=0.6,
                  label=f"AI mean = {ai.mean():.2e}")
        a.set_title(name)
        a.set_xlabel("physical time of target snapshot (days)")
        a.set_ylabel("MSE (log)")
        a.grid(True, which="both", alpha=0.3)
        a.legend(loc="best", fontsize=8)
    fig.suptitle(f"Per-step single-step error vs time  "
                 f"({step_days:.0f}-day/step, {len(times)} pairs, split={args.split})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(args.out_dir, "per_step_error.png"), dpi=120)
    plt.close(fig)

    # --- 摘要 ---
    beat_c = float(np.mean(mse_c < old_c))               # AI 跑赢不变基线的 pair 占比
    beat_h = float(np.mean(mse_h < old_h))
    print("\n=== per-step single-step error summary ===")
    print(f"pairs              : {len(times)}  (split={args.split})")
    print(f"time span          : {times[0]:.0f} .. {times[-1]:.0f} days")
    print(f"AI  MSE_c  mean/min/max : {mse_c.mean():.2e} / {mse_c.min():.2e} / {mse_c.max():.2e}")
    print(f"AI  MSE_h  mean/min/max : {mse_h.mean():.2e} / {mse_h.min():.2e} / {mse_h.max():.2e}")
    print(f"old MSE_c  mean         : {old_c.mean():.2e}")
    print(f"old MSE_h  mean         : {old_h.mean():.2e}")
    print(f"AI < old  (c / h)       : {beat_c*100:.0f}% / {beat_h*100}%  of pairs")
    print(f"written to              : {args.out_dir}/  (per_step_error.png/.npz/.csv)")


if __name__ == "__main__":
    main()
