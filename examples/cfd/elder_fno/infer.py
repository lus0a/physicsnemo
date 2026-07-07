# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""单步推理: 用训练好的 Elder FNO 预测一个 macro 步 (c_n, P_n) -> (c_{n+1}, P_{n+1})。

加载 checkpoint, 喂入一个输入状态 (c_n, P_n), 输出 30 天 (或训练用的 file_stride×10 天)
后的预测 (c_{n+1}, P_{n+1})。结果存 .npz (数组) 和可选 .vtu (ParaView 可视化)。

输入须在与训练相同的 64×256 cell 网格上 (c∈[0,1], P 为真实压力 Pa)。模型只做单步。

用法:
    # 用数据集第 0 个快照作输入 (会自动和第 0+file_stride 个真值对比)
    python infer.py --checkpoint outputs_baseline30days/checkpoints --index 0

    # 用任意 VTU 文件作输入 (须与训练同网格)
    python infer.py --checkpoint outputs_baseline30days/checkpoints \\
        --input DataSet/progame_name-0.0000000000.vtu --out_vtu pred.vtu
"""

from __future__ import annotations

import argparse

import numpy as np
import pyvista as pv
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
    load_checkpoint(path=checkpoint, models=model, device=device)   # 自动取目录里最新的 ckpt
    model.eval()
    return model


def _read_vtu_field(path, order, ny, nx):
    """读一个 .vtu 的 c/P, 按 cell 中心 order 还原成 [ny, nx] 规则网格。"""
    g = pv.read(path)
    c = g.cell_data["c"][order].reshape(ny, nx)
    P = g.cell_data["P"][order].reshape(ny, nx)
    return c, P, g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs_baseline30days/checkpoints",
                   help="checkpoint 目录 (取最新) 或 .pt 文件")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--input", default=None, help="输入 VTU (c_n, P_n); 与 --index 二选一")
    p.add_argument("--index", type=int, default=None, help="用数据集第 N 个快照作输入")
    p.add_argument("--out", default="pred_step.npz", help="输出 .npz (c, P 数组 [64,256])")
    p.add_argument("--out_vtu", default=None, help="可选: 输出 VTU (含 c_pred/P_pred, 原 cell 顺序)")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    phy = cfg.physics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DistributedManager.initialize()                     # 单进程也无妨; load_checkpoint 需要

    # 数据集: 只为拿网格 / p_hydro / p_scale / _order (file_stride 须与训练一致)
    dp = VtuElderDataset(
        cfg.data.train_dir, 1, device, phy.phi, phy.Dm, phy.permeability, phy.viscosity,
        phy.g, phy.rho_f, phy.drho, phy.W, phy.H, phy.dt_macro,
        file_stride=int(cfg.data.get("file_stride", 1)),
    )
    p_hydro, p_scale = dp.p_hydro, dp.p_scale
    order = dp._order
    step_days = dp.dt_macro / 86400.0

    model = _build_model(cfg, dp, args.checkpoint, device)
    use_residual = bool(cfg.training.get("residual", False))
    print(f"loaded checkpoint from {args.checkpoint} | {step_days:.0f}-day/step | residual={use_residual}")

    # --- 取输入 (c_n, P_n) ---
    if args.index is not None:
        c_n = dp.data[args.index:args.index + 1, 0:1]          # [1,1,Ny_tot,Nx_tot]
        P_n = dp.data[args.index:args.index + 1, 1:2]
        in_label = f"snapshot {args.index}"
    elif args.input:
        c_np, P_np, _ = _read_vtu_field(args.input, order, dp.Ny_tot, dp.Nx_tot)
        c_n = torch.from_numpy(c_np[None, None]).to(device)
        P_n = torch.from_numpy(P_np[None, None]).to(device)
        in_label = args.input
    else:
        raise SystemExit("须指定 --input <vtu> 或 --index <N>")

    # --- 单步前向 (与训练同: h=(P-p_hydro)/p_scale 归一化, 残差模式则重建) ---
    h_n = (P_n - p_hydro) / p_scale
    invar = torch.cat([c_n, h_n], dim=1)
    with torch.no_grad():
        raw = model(invar)
    if use_residual:
        c_pred = c_n + raw[:, 0:1]
        h_pred = h_n + raw[:, 1:2]
    else:
        c_pred = raw[:, 0:1]
        h_pred = raw[:, 1:2]
    P_pred = h_pred * p_scale + p_hydro                       # 还原真实压力 [Pa]

    # --- 保存 .npz ---
    c_pred_np = c_pred[0, 0].cpu().numpy()
    P_pred_np = P_pred[0, 0].cpu().numpy()
    np.savez(args.out, c=c_pred_np, P=P_pred_np,
             p_hydro=p_hydro[0].cpu().numpy(), p_scale=np.float32(p_scale))
    print(f"\n输入: {in_label}")
    print(f"预测 (+{step_days:.0f} 天): c range [{c_pred_np.min():.3f}, {c_pred_np.max():.3f}] "
          f"(物理 [0,1]); P range [{P_pred_np.min():.3e}, {P_pred_np.max():.3e}] Pa")
    print(f"saved {args.out}  (c, P 形状 {c_pred_np.shape})")

    # --- 若有真值, 对比 ---
    if args.index is not None and args.index + dp.file_stride < dp.n_files:
        j = args.index + dp.file_stride
        c_true = dp.data[j, 0].cpu().numpy()
        P_true = dp.data[j, 1].cpu().numpy()
        mse_c = float(np.mean((c_pred_np - c_true) ** 2))
        mse_P = float(np.mean((P_pred_np - P_true) ** 2))
        # 单步 old 基线 (输入当输出)
        old_c = float(np.mean((c_n[0, 0].cpu().numpy() - c_true) ** 2))
        print(f"\nvs 真值 snapshot {j}:")
        print(f"  AI   MSE_c={mse_c:.2e}  MSE_P={mse_P:.2e}")
        print(f"  old  MSE_c={old_c:.2e}  (不变基线)")
        print(f"  AI/old c = {mse_c / max(old_c, 1e-12):.2f}x  (<1 表示跑赢不变基线)")

    # --- 可选: 写 VTU (把预测写回输入网格的原始 cell 顺序, ParaView 可直接看) ---
    if args.out_vtu:
        if not args.input:
            print("(--out_vtu 需要 --input 指定一个 VTU 作网格模板, 跳过)")
        else:
            g = pv.read(args.input)
            c_orig = np.empty(g.n_cells, dtype=np.float32)
            P_orig = np.empty(g.n_cells, dtype=np.float32)
            c_orig[order] = c_pred_np.ravel()                 # sorted-position -> 原 cell 下标
            P_orig[order] = P_pred_np.ravel()
            g.cell_data["c_pred"] = c_orig
            g.cell_data["P_pred"] = P_orig
            g.save(args.out_vtu)
            print(f"saved {args.out_vtu} (cell_data: c_pred, P_pred)")


if __name__ == "__main__":
    main()
