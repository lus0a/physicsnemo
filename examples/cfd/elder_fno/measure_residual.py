# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""量 PDE 残差大小 (训练 loss 尺度): 回答 "AI 预测满不满足 Elder 方程", 并为选 physics_weight 提供依据。

对每个 pair 用 _residuals_own_fd (与训练完全一致的有限差分残差) 算:
* AI 预测 的归一化平均绝对残差 μ_c, μ_p (输运 / 连续性);
* 参考解 (真值 c1,p1) 自己的残差 μ_c_ref, μ_p_ref —— 有限差分截断误差地板,
  代表 "残差最小能到多少"。

残差被 scale_c/scale_p 归一化到 O(1); 返回的是 mean-|R| (L1 风格), 与训练 loss_pde 同口径。
注: 训练里 batch_size>1 时 _residuals_own_fd 返回值会 ×B, 这里一律除回 B 报 per-sample 值。

纯推理 / 只读。用法:
    python measure_residual.py --checkpoint outputs_baseline30days/checkpoints
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils import load_checkpoint

from vtu_dataset import VtuElderDataset
from train_elder_fno import _resolve_fno_modes, _residuals_own_fd, _build_residual_mask


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
    load_checkpoint(path=checkpoint, models=model, device=device)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", default="outputs_baseline30days/checkpoints")
    p.add_argument("--train_dir", default=None)
    p.add_argument("--split", default="all", choices=["all", "train", "val"])
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    phy = cfg.physics
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    DistributedManager.initialize()

    dp = VtuElderDataset(
        args.train_dir or cfg.data.get("train_dir", "DataSet"),
        batch_size=1, device=device,
        phi=phy.phi, Dm=phy.Dm, permeability=phy.permeability, viscosity=phy.viscosity,
        g=phy.g, rho_f=phy.rho_f, drho=phy.drho, W=phy.W, H=phy.H, dt_macro=phy.dt_macro,
        flow_sign=phy.get("flow_sign", 1.0), split=args.split,
        val_frac=float(cfg.data.get("val_frac", 0.2)),
        n_val_blocks=int(cfg.data.get("n_val_blocks", 8)),
        val_gap=int(cfg.data.get("val_gap", 2)),
        file_stride=int(cfg.data.get("file_stride", 1)),
    )
    p_hydro, p_scale = dp.p_hydro, dp.p_scale
    s = dp.file_stride
    continuity_weight = float(cfg.training.get("continuity_weight", 1.0))
    mask = _build_residual_mask(dp, int(cfg.training.get("mask_top_rows", 2)), device)

    model = _build_model(cfg, dp, args.checkpoint, device)
    use_residual = bool(cfg.training.get("residual", False))
    print(f"loaded {args.checkpoint} | split={args.split} -> {len(dp.pair_indices)} pairs | device={device}")

    idx = dp.pair_indices
    ai_c, ai_p, ref_c, ref_p, data_c = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), args.chunk):
            k = torch.as_tensor(idx[i:i + args.chunk], device=device, dtype=torch.long)
            B = k.numel()
            c0 = dp.data[k, 0:1].float()
            p0 = dp.data[k, 1:2].float()
            c1 = dp.data[k + s, 0:1].float()
            p1 = dp.data[k + s, 1:2].float()
            h0 = (p0 - p_hydro) / p_scale
            invar = torch.cat([c0, h0], dim=1)
            raw = model(invar)
            if use_residual:
                pred_c = c0 + raw[:, 0:1]
                pred_h = h0 + raw[:, 1:2]
            else:
                pred_c, pred_h = raw[:, 0:1], raw[:, 1:2]
            pred_p = pred_h * p_scale + p_hydro           # 还原真实压力 (残差用 P, 不是 h)

            # AI 预测的残差 (返回值 ×B, 除回 B 得 per-sample)
            lc, lp = _residuals_own_fd(pred_c.float(), pred_p.float(), c0, dp.dt_macro, dp, mask)
            ai_c.append((lc / B).item()); ai_p.append((lp / B).item())
            # 参考解自己的残差 (地板): 把真值当 pred
            lc2, lp2 = _residuals_own_fd(c1.float(), p1.float(), c0, dp.dt_macro, dp, mask)
            ref_c.append((lc2 / B).item()); ref_p.append((lp2 / B).item())
            # 顺带数据 MSE (per-sample)
            data_c.append((((pred_c - c1) ** 2).mean(dim=(1, 2, 3))).mean().item())

    ai_c, ai_p = np.mean(ai_c), np.mean(ai_p)
    ref_c, ref_p = np.mean(ref_c), np.mean(ref_p)
    data = np.mean(data_c)
    ai_tot = ai_c + continuity_weight * ai_p
    ref_tot = ref_c + continuity_weight * ref_p

    print("\n=== 归一化平均绝对残差 (per-sample, 与训练 loss_pde 同口径) ===")
    print(f"{'':22s}{'AI 预测':>14s}{'参考解(地板)':>16s}{'AI/参考':>10s}")
    print(f"{'输运残差 μ_c':<22s}{ai_c:>14.3e}{ref_c:>16.3e}{ai_c/max(ref_c,1e-12):>10.2f}")
    print(f"{'连续性残差 μ_p':<22s}{ai_p:>14.3e}{ref_p:>16.3e}{ai_p/max(ref_p,1e-12):>10.2f}")
    print(f"{'合计 μ_c+wμ_p':<22s}{ai_tot:>14.3e}{ref_tot:>16.3e}")
    print(f"\n参考解残差地板 = {ref_tot:.2e}  (FD 截断误差, 残差最小能到这)")
    print(f"AI 残差         = {ai_tot:.2e}  ({ai_tot/max(ref_tot,1e-12):.1f}× 地板)")
    print(f"\n数据 MSE (loss_data 量级, per-sample) = {data:.2e}")
    print(f"\n--- 选 physics_weight 的参考 ---")
    print(f"若想让物理项在【训练初期】(loss_data~0.028) 与数据项相当:")
    print(f"  physics_weight ~ 0.028 / (B*μ_total),  B=训练 batch_size=128")
    w_early = 0.028 / (128 * max(ai_tot, 1e-12))
    print(f"  => ~ {w_early:.2e}")
    print(f"若想让物理项在【收敛后】(loss_data~2e-6) 与数据项相当:")
    w_late = 2e-6 / (128 * max(ai_tot, 1e-12))
    print(f"  => ~ {w_late:.2e}  (很小; 残差 O(1) vs 数据 1e-6 本身差很多个量级)")


if __name__ == "__main__":
    main()
