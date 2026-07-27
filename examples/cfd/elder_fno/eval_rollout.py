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
from physicsnemo.utils import load_checkpoint

from ufno import build_model
from vtu_dataset import VtuElderDataset
from train_elder_fno import _resolve_fno_modes, _resolve_in_channels, build_invar


def _plot_fields(c_true, c_pred, h_true, h_pred, title, out_path):
    """2x3 True/Pred/|error| comparison for c (row 0) and h (row 1)."""
    c_true, c_pred, h_true, h_pred = (np.asarray(x) for x in (c_true, c_pred, h_true, h_pred))  # 统一转 numpy
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.4)        # 2 行 3 列; 行=c/h, 列=True/Pred/error
    ax = [[fig.add_subplot(gs[0, c]), fig.add_subplot(gs[1, c])] for c in range(3)]
    ax = [[ax[c][0] for c in range(3)], [ax[c][1] for c in range(3)]]  # 重排成 ax[row][col]
    fig.suptitle(title, fontsize=15, fontweight="bold")

    titles = [("True c", "Pred c", "|c error|"), ("True h", "Pred h", "|h error|")]
    fields = [(c_true, c_pred, np.abs(c_pred - c_true)),   # c 行: 真值/预测/绝对误差
              (h_true, h_pred, np.abs(h_pred - h_true))]   # h 行
    for row in range(2):
        true_f = fields[row][0]
        if row == 0:
            row_vmin, row_vmax = 0.0, 1.0          # c 用物理范围 [0,1]
        else:
            row_vmin, row_vmax = float(true_f.min()), float(true_f.max())   # h 用真值范围统一 True/Pred
        for col in range(3):
            f = fields[row][col]
            if col < 2:
                vmin, vmax = row_vmin, row_vmax    # True/Pred 共享范围便于对比
            else:
                vmin, vmax = 0.0, float(f.max())   # 误差列单独缩放
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
    p.add_argument("--train_dir", default=None,
                   help="VTU data dir for ground-truth trajectory (default: cfg.data.train_dir)")
    p.add_argument("--out_dir", default="rollout_eval")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)              # 读训练用的 config (保证模型/物理参数一致)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)       # 输出目录 (rollout_error.png 等)
    DistributedManager.initialize()                # physicsnemo 分布式初始化 (单进程也无妨)

    # --- 真值轨迹: 直接读 unisolver VTU 快照 (高精度, 训练用的同一批数据) ---
    N = int(args.steps)                             # 要推演的 macro 步数
    phy = cfg.physics
    train_dir = args.train_dir or cfg.data.get("train_dir", "DataSet")
    dp = VtuElderDataset(
        train_dir=train_dir, batch_size=1, device=device,
        phi=phy.phi, Dm=phy.Dm, permeability=phy.permeability, viscosity=phy.viscosity,
        g=phy.g, rho_f=phy.rho_f, drho=phy.drho, W=phy.W, H=phy.H,
        dt_macro=phy.dt_macro, flow_sign=phy.get("flow_sign", 1.0),
        file_stride=int(cfg.data.get("file_stride", 1)),
    )
    if N + 1 > dp.n_files:                          # VTU 只有 n_files 个快照, rollout 不能超出
        print(f"WARNING: steps {N} 超过 VTU 快照数 {dp.n_files}-1, 截断到 {dp.n_files - 1}")
        N = dp.n_files - 1
    p_hydro, p_scale = dp.p_hydro, dp.p_scale       # 必须与训练一致 (p_scale 由数据自动算)
    dt_days = dp.dt_macro / (24 * 3600.0)           # 每个 macro 步折合多少天 (画图用)

    # --- 模型 (结构必须与训练一致, 否则权重加载不上) ---
    mdl = cfg.model
    modes = _resolve_fno_modes(
        OmegaConf.to_container(mdl, resolve=True)["num_fno_modes"], dp, mdl.padding)
    model = build_model(mdl, num_fno_modes=modes, in_channels=_resolve_in_channels(mdl)).to(device)
    load_checkpoint(path=args.checkpoint, models=model, device=device)   # 加载训练好的权重
    model.eval()                                    # 推理模式
    print(f"loaded checkpoint from {args.checkpoint} (arch={mdl.get('arch', 'fno')})")

    # --- 真值轨迹: VTU 快照 0..N (通道 0=c, 1=P), 每个 [1,1,Ny_tot,Nx_tot] ---
    true_c = [dp.data[t:t + 1, 0:1] for t in range(N + 1)]
    true_p = [dp.data[t:t + 1, 1:2] for t in range(N + 1)]

    # --- 模型 rollout: 把自身预测喂回当输入 (在归一化 h 空间里循环) ---
    dt_aware = bool(cfg.model.get("dt_channel", False))
    dt_ref = float(cfg.model.get("dt_ref_s", 2.592e6))
    cur_c = true_c[0]                               # 从真值初值出发
    cur_h = (true_p[0] - p_hydro) / p_scale         # 归一化初值水头
    pred_c, pred_h = [], []
    use_residual = bool(cfg.training.get("residual", False))   # 与训练一致的残差预测
    with torch.no_grad():                           # 纯推理
        for t in range(N):
            invar = build_invar(cur_c, cur_h, dp.dt_macro, dt_aware, dt_ref)  # (c_t, h_t[, dt]) 拼输入
            out = model(invar)                          # 单步输出 (直接值 或 残差模式下的增量)
            if use_residual:                            # c_{t+1}=c_t+Δc, h_{t+1}=h_t+Δh
                cur_c = cur_c + out[:, 0:1]
                cur_h = cur_h + out[:, 1:2]
            else:
                cur_c, cur_h = out[:, 0:1], out[:, 1:2]
            pred_c.append(cur_c.detach())               # 记录预测 (喂回下一步)
            pred_h.append(cur_h.detach())

    # --- 逐步误差 (预测第 t 步 vs 真值第 t+1 步) ---
    rmse_c, rmse_h = [], []
    c_min, c_max = 1e9, -1e9                        # 跟踪预测 c 的范围 (看是否越界 [0,1])
    for t in range(N):
        tc = true_c[t + 1]                          # 对应的真值
        th = (true_p[t + 1] - p_hydro) / p_scale    # 归一化真值水头
        rmse_c.append(float(torch.sqrt(((pred_c[t] - tc) ** 2).mean())))   # c 的 RMSE
        rmse_h.append(float(torch.sqrt(((pred_h[t] - th) ** 2).mean())))   # h 的 RMSE
        c_min = min(c_min, float(pred_c[t].min()))   # 更新预测 c 的最小/最大值
        c_max = max(c_max, float(pred_c[t].max()))

    # --- 误差曲线 (log y) ---
    steps = np.arange(1, N + 1)                     # x 轴: 步数 1..N
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(steps, rmse_c, "o-", ms=4, label="RMSE c")     # c 误差 (对数 y)
    ax.semilogy(steps, rmse_h, "s-", ms=4, label="RMSE h")     # h 误差
    ax.axhline(0.05, color="r", ls="--", alpha=0.6, label="0.05 divergence threshold")   # 发散阈值
    ax.set_xlabel(f"rollout step (1 step = {dt_days:.1f} days)")
    ax.set_ylabel("RMSE (log)")
    ax.set_title(f"Autoregressive rollout error vs step ({N} steps)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "rollout_error.png"), dpi=120)
    plt.close(fig)

    # --- 选几个步数画场图对比 (早 / 中 / 晚) ---
    for t in [1, N // 2, N]:
        if t < 1 or t > N:
            continue
        tc = true_c[t][0, 0].cpu().numpy()          # 真值 c (去 batch/channel 维, 转 numpy)
        pc = pred_c[t - 1][0, 0].cpu().numpy()      # 预测 c (pred[t-1] 对应第 t 步)
        th = ((true_p[t] - p_hydro) / p_scale)[0, 0].cpu().numpy()   # 真值归一化 h
        ph = pred_h[t - 1][0, 0].cpu().numpy()      # 预测 h
        days = t * dt_days                          # 该步对应的物理天数
        _plot_fields(tc, pc, th, ph,
                     f"Rollout step {t} (~{days:.0f} days)",
                     os.path.join(args.out_dir, f"rollout_field_step{t:03d}.png"))

    # --- 摘要 ---
    div_c = next((t + 1 for t, r in enumerate(rmse_c) if r > 0.05), None)   # 首个 c RMSE 超 0.05 的步
    print("\n=== rollout summary ===")
    print(f"steps              : {N}  ({N*dt_days:.0f} days)")
    print(f"RMSE c  step1/last : {rmse_c[0]:.3e} / {rmse_c[-1]:.3e}")   # 首步/末步误差
    print(f"RMSE h  step1/last : {rmse_h[0]:.3e} / {rmse_h[-1]:.3e}")
    print(f"pred c range       : [{c_min:.3f}, {c_max:.3f}]  (truth in [0,1])")   # 越界检查
    print(f"c diverges (>0.05) : {'step ' + str(div_c) if div_c else 'never within rollout'}")
    print(f"plots written to   : {args.out_dir}/")


if __name__ == "__main__":
    main()
