# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""量 PDE 残差大小 (训练 loss 尺度): 回答 "AI 预测满不满足 Elder 方程", 并为选 physics_weight 提供依据。

支持两个残差后端 (与 train_elder_fno.py 完全同口径; 由 config.training.physics_backend 决定,
可用 --backend 覆盖):

* own_fd       — 原中心差分, masked L1 + scale 归一化 (残差 O(1))。返回 per-sample mean-|R|;
                 训练里 loss_pde = mean-|R| 但 **未除 batch**, 即 ×B。
* unisolver_fv — 与 Elder::FormFunction 对齐的 FV 残差 (SI 单位, 无归一化)。
                 返回 mean(F^2), batch-invariant; 训练里 loss_pde=(Fp*Fp).mean()+w_c*(Fc*Fc).mean()。

对每个 pair 算:
* AI 预测 的残差;
* 参考解 (真值 c1,p1) 自己的残差 —— 截断误差地板, 代表 "残差最小能到多少"。

并据此给 physics_weight 的参考值 (让物理项 ~ 数据项, 二者梯度同量级)。

注意两个后端的 c/p 权重角色相反 (与训练一致):
  own_fd:       loss_pde = R_c + continuity_weight * R_p   (输运 w=1, 连续性 w=continuity_weight)
  unisolver_fv: loss_pde = F_p + continuity_weight * F_c   (flow w=1, transport w=continuity_weight=w_c)

纯推理 / 只读。用法:
    python measure_residual.py --checkpoint outputs_elder_fno/checkpoints
    python measure_residual.py --backend unisolver_fv --checkpoint outputs_elder_fno/checkpoints
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
from elder_residual_fv import ElderPhysics, form_function_elder


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


def _build_elder_physics(cfg, pin_enable=True):
    """从 config.physics 构造 ElderPhysics (与 train_elder_fno.py 同样)。
    pin_enable=False 时关掉左上/右上角的压力 pin (诊断 pin 是否主导流动残差)。"""
    phy = cfg.physics
    return ElderPhysics(
        phi=float(phy.phi), perm=float(phy.permeability), visc=float(phy.viscosity),
        Dm=float(phy.Dm), rho_f=float(phy.rho_f), drho=float(phy.drho),
        g=float(phy.g), W=float(phy.W), H=float(phy.H),
        pin_enable=pin_enable,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", default="outputs_elder_fno/checkpoints")
    p.add_argument("--backend", default=None,
                   help="残差后端: own_fd | unisolver_fv (默认读 config.training.physics_backend)")
    p.add_argument("--train_dir", default=None)
    p.add_argument("--split", default="all", choices=["all", "train", "val"])
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--no-pin", action="store_true",
                   help="关掉 ElderPhysics 压力 pin (诊断 pin 是否主导流动残差)")
    p.add_argument("--pin", action="store_true",
                   help="强制开 pin (测 unisolver Newton 实际看到的含 pin FormFunction 范数)")
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
    dt = dp.dt_macro
    B_train = int(cfg.data.get("batch_size", 128))   # 训练 batch (own_fd 的 w 公式要用)

    backend = (args.backend or str(cfg.training.get("physics_backend", "own_fd"))).lower()
    if backend not in ("own_fd", "unisolver_fv"):
        backend = "own_fd"

    mask = None
    elder_phy = None
    if backend == "unisolver_fv":
        # pin 默认读 config.training.pin_enable (与训练一致); --no-pin / --pin 显式覆盖
        if args.no_pin:
            pin_enable = False
        elif args.pin:
            pin_enable = True
        else:
            pin_enable = bool(cfg.training.get("pin_enable", False))
        elder_phy = _build_elder_physics(cfg, pin_enable=pin_enable)
        print(f"backend = unisolver_fv  (Elder FormFunction, mean-square, SI 单位 | pin {'ON' if pin_enable else 'OFF'})")
    else:
        mask = _build_residual_mask(dp, int(cfg.training.get("mask_top_rows", 2)), device)
        print(f"backend = own_fd  (masked L1 + scale, 归一化到 O(1))")

    model = _build_model(cfg, dp, args.checkpoint, device)
    use_residual = bool(cfg.training.get("residual", False))
    print(f"loaded {args.checkpoint} | split={args.split} -> {len(dp.pair_indices)} pairs | device={device}")
    print(f"continuity_weight={continuity_weight} | dt_macro={dt:.1f}s | train batch_size={B_train}")

    idx = dp.pair_indices
    # a = 训练里权重为 1 的残差项; b = 权重为 continuity_weight 的项 (两个后端都成立)
    ai_a, ai_b, ref_a, ref_b, data_c = [], [], [], [], []
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

            if backend == "unisolver_fv":
                # mean-square 残差 (batch-invariant), 与训练 loss_pde 同口径
                # a = flow (Fp, w=1), b = transport (Fc, w=continuity_weight)
                Fp, Fc = form_function_elder(pred_c, pred_p, c0, p0, dt, elder_phy)
                ai_a.append((Fp * Fp).mean().item())
                ai_b.append((Fc * Fc).mean().item())
                # 参考地板: 真值 (c1,p1) 代入
                Fp2, Fc2 = form_function_elder(c1, p1, c0, p0, dt, elder_phy)
                ref_a.append((Fp2 * Fp2).mean().item())
                ref_b.append((Fc2 * Fc2).mean().item())
            else:
                # own_fd: 返回值 ×B, 除回 B 得 per-sample mean-|R|
                # a = 输运 R_c (w=1), b = 连续性 R_p (w=continuity_weight)
                lc, lp = _residuals_own_fd(pred_c, pred_p, c0, dt, dp, mask)
                ai_a.append((lc / B).item()); ai_b.append((lp / B).item())
                lc2, lp2 = _residuals_own_fd(c1, p1, c0, dt, dp, mask)
                ref_a.append((lc2 / B).item()); ref_b.append((lp2 / B).item())

            # 数据 MSE (loss_data 量级, per-sample)
            data_c.append((((pred_c - c1) ** 2).mean(dim=(1, 2, 3))).mean().item())

    ai_a, ai_b = np.mean(ai_a), np.mean(ai_b)
    ref_a, ref_b = np.mean(ref_a), np.mean(ref_b)
    data = np.mean(data_c)
    ai_tot = ai_a + continuity_weight * ai_b      # 训练里 loss_pde 的 per-sample 量 (own_fd 未含 ×B)
    ref_tot = ref_a + continuity_weight * ref_b

    unit = "mean(F^2)" if backend == "unisolver_fv" else "mean-|R|"
    if backend == "unisolver_fv":
        lab_a, lab_b = "流动 Fp (w=1)", "输运 Fc (w=w_c)"
    else:
        lab_a, lab_b = "输运 R_c (w=1)", "连续性 R_p (w=w_c)"

    print(f"\n=== 残差 ({backend}, {unit}, per-sample, 与训练 loss_pde 同口径) ===")
    print(f"{'':22s}{'AI 预测':>14s}{'参考解(地板)':>16s}{'AI/参考':>10s}")
    print(f"{lab_a:<22s}{ai_a:>14.3e}{ref_a:>16.3e}{ai_a/max(ref_a,1e-300):>10.2f}")
    print(f"{lab_b:<22s}{ai_b:>14.3e}{ref_b:>16.3e}{ai_b/max(ref_b,1e-300):>10.2f}")
    print(f"{'合计 loss_pde':<22s}{ai_tot:>14.3e}{ref_tot:>16.3e}")
    print(f"\n参考解地板 = {ref_tot:.2e}  (截断误差, 残差最小能到这)")
    print(f"AI 残差     = {ai_tot:.2e}  ({ai_tot/max(ref_tot,1e-300):.1f}× 地板)")
    print(f"\n数据 MSE (loss_data 量级, per-sample) = {data:.2e}")

    print(f"\n--- 选 physics_weight 的参考 (让物理项 ≈ 数据项, 二者梯度同量级) ---")
    # 训练里总 loss = loss_data + physics_weight * loss_pde
    # 想让 physics_weight * loss_pde ≈ loss_data  =>  w ≈ loss_data / loss_pde
    if backend == "unisolver_fv":
        # mean-square 已 batch-invariant, 直接除 (不乘 batch)
        denom = max(ai_tot, 1e-300)
        print(f"unisolver_fv 用 mean-square (不乘 batch_size):")
        print(f"  训练初期 (loss_data~0.028): physics_weight ~ 0.028 / {ai_tot:.2e} = {0.028/denom:.2e}")
        print(f"  收敛后   (loss_data~2e-6):  physics_weight ~ 2e-6  / {ai_tot:.2e} = {2e-6/denom:.2e}")
    else:
        # own_fd 训练里 ×B (未除 batch), 故 w 公式分母要乘 batch_size
        denom = max(B_train * ai_tot, 1e-300)
        print(f"own_fd 训练里 loss_pde 未除 batch (×B_train={B_train}):")
        print(f"  训练初期 (loss_data~0.028): physics_weight ~ 0.028 / ({B_train}·{ai_tot:.2e}) = {0.028/denom:.2e}")
        print(f"  收敛后   (loss_data~2e-6):  physics_weight ~ 2e-6  / ({B_train}·{ai_tot:.2e}) = {2e-6/denom:.2e}")
    print("(以上仅为量级参考; 实际可在此附近做 1/3×~3× 扫参, 看 val MSE 是否优于纯数据基线。)")


if __name__ == "__main__":
    main()
