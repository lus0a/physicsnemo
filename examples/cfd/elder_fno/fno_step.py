# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""单步 FNO 推理 wrapper (C++ Elder  <->  Python PhysicsNeMo FNO).

C++ 端 (Elder::FnoPredict) 每个 macro 步:
  1) 把当前 (c, P) 写成 raw binary: float32 [NZ*NX] (c) + float32 [NZ*NX] (P), 行主序;
  2) 调用本脚本;
  3) 读回 fno_out.bin 的 (c_pred, P_pred) 覆盖 field_ 作为 Newton 初值.

归一化 (p_hydro, p_scale) 与训练完全一致: 首次建 VtuElderDataset 算一次并缓存到
fno_norm.npz (含 p_hydro / p_scale / num_fno_modes / 网格尺寸); 后续直接读缓存,
跳过 DataSet 全量加载, 只重新 load model checkpoint. residual 模式从 config 读,
由本脚本处理 (C++ 端无需感知).

C++ 拼好的命令 (在 workdir 下执行):
  python fno_step.py --in fno_in.bin --out fno_out.bin \\
                     --checkpoint outputs_baseline30days/checkpoints
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils import load_checkpoint

from vtu_dataset import VtuElderDataset
from train_elder_fno import _resolve_fno_modes, _resolve_in_channels, build_invar


NORM_CACHE = "fno_norm.npz"


def _load_or_build_norm(cfg, device):
    """返回 (p_hydro[Ny,Nx] tensor, p_scale float, modes list, NY, NX).

    首次建 VtuElderDataset (慢, 读全量 DataSet) 算 p_hydro/p_scale/modes 并缓存;
    后续读 fno_norm.npz 直接复用, 不再加载 DataSet."""
    if os.path.exists(NORM_CACHE):
        z = np.load(NORM_CACHE)
        p_hydro = torch.from_numpy(z["p_hydro"]).to(device)
        p_scale = float(z["p_scale"])
        modes = [int(m) for m in z["modes"]]
        return p_hydro, p_scale, modes, int(z["NY"]), int(z["NX"])

    phy = cfg.physics
    dp = VtuElderDataset(
        cfg.data.train_dir, 1, device, phy.phi, phy.Dm, phy.permeability, phy.viscosity,
        phy.g, phy.rho_f, phy.drho, phy.W, phy.H, phy.dt_macro,
        file_stride=int(cfg.data.get("file_stride", 1)),
    )
    modes = _resolve_fno_modes(
        OmegaConf.to_container(cfg.model, resolve=True)["num_fno_modes"], dp, cfg.model.padding)
    np.savez(
        NORM_CACHE,
        p_hydro=dp.p_hydro.cpu().numpy(),
        p_scale=np.float32(dp.p_scale),
        modes=np.asarray(modes, dtype=np.int64),
        NY=np.int64(dp.Ny_tot), NX=np.int64(dp.Nx_tot),
    )
    print(f"fno_step: built {NORM_CACHE} (p_scale={float(dp.p_scale):.1f}, "
          f"modes={modes}, {dp.Ny_tot}x{dp.Nx_tot})")
    return dp.p_hydro, float(dp.p_scale), modes, int(dp.Ny_tot), int(dp.Nx_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="C++ 写来的 raw binary 输入")
    ap.add_argument("--out", required=True, help="写回 C++ 的 raw binary 输出")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="outputs_baseline30days/checkpoints",
                    help="checkpoint 目录 (取最新) 或 .pt 文件")
    ap.add_argument("--dt", type=float, default=None,
                    help="本时间步 Δt (秒); dt_channel=true 时必传以匹配训练, 不传则报错。")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DistributedManager.initialize()                       # load_checkpoint 需要

    p_hydro, p_scale, modes, NY, NX = _load_or_build_norm(cfg, device)
    residual = bool(cfg.training.get("residual", False))

    mdl = cfg.model
    model = FNO(
        in_channels=_resolve_in_channels(mdl), out_channels=mdl.out_channels,
        decoder_layers=mdl.decoder_layers, decoder_layer_size=mdl.decoder_layer_size,
        dimension=mdl.dimension, latent_channels=mdl.latent_channels,
        num_fno_layers=mdl.num_fno_layers,
        num_fno_modes=modes,                              # 用缓存里解析好的 modes, 不再依赖 dp
        padding=mdl.padding,
    ).to(device)
    # 直接加载 .mdlus (含完整模型); 跳过 load_checkpoint 会额外读的 1.5GB .pt
    # (.pt 是 optimizer/scheduler 状态, 推理用不到), 大幅加速每步 reload
    import glob as _glob
    _mdlus = sorted(_glob.glob(os.path.join(args.checkpoint, "*.mdlus")))
    if _mdlus:
        def _epoch_of(_f):
            try:
                return int(os.path.basename(_f).rsplit(".", 2)[-2])
            except ValueError:
                return 0
        _mdlus.sort(key=_epoch_of)
        model.load(_mdlus[-1])
        print(f"fno_step: loaded model {_mdlus[-1]}", flush=True)
    else:
        load_checkpoint(path=args.checkpoint, models=model, device=device)
    model.eval()

    # --- 读 raw binary: c[NY*NX] + P[NY*NX] float32 行主序 ---
    raw = np.fromfile(args.inp, dtype=np.float32)
    n = NY * NX
    if raw.size < 2 * n:
        print(f"fno_step: input too short ({raw.size} floats < {2*n} needed)", file=sys.stderr)
        sys.exit(1)
    c_n = torch.from_numpy(raw[:n].reshape(NY, NX)[None, None]).to(device)
    P_n = torch.from_numpy(raw[n:2 * n].reshape(NY, NX)[None, None]).to(device)

    # --- 归一化 + 单步前向 (与 infer.py 完全一致) ---
    dt_aware = bool(cfg.model.get("dt_channel", False))
    dt_ref = float(cfg.model.get("dt_ref_s", 2.592e6))
    if dt_aware and args.dt is None:
        print("fno_step: dt_channel=true 需要传 --dt <秒> (本时间步 Δt)", file=sys.stderr)
        sys.exit(1)
    dt_val = args.dt if args.dt is not None else 0.0
    h_n = (P_n - p_hydro) / p_scale
    invar = build_invar(c_n, h_n, dt_val, dt_aware, dt_ref)
    with torch.no_grad():
        raw_out = model(invar)
    if residual:
        c_pred = c_n + raw_out[:, 0:1]
        h_pred = h_n + raw_out[:, 1:2]
    else:
        c_pred = raw_out[:, 0:1]
        h_pred = raw_out[:, 1:2]
    P_pred = h_pred * p_scale + p_hydro                   # 还原真实压力 [Pa]

    # --- 写回 raw binary: c_pred[NY*NX] + P_pred[NY*NX] ---
    c_arr = c_pred[0, 0].cpu().numpy().astype(np.float32).ravel()
    P_arr = P_pred[0, 0].cpu().numpy().astype(np.float32).ravel()
    with open(args.out, "wb") as f:
        f.write(c_arr.tobytes())
        f.write(P_arr.tobytes())
    print(f"fno_step: in c[{float(c_n.min()):.3f},{float(c_n.max()):.3f}] "
          f"P[{float(P_n.min()):.3e},{float(P_n.max()):.3e}] -> "
          f"out c[{float(c_arr.min()):.3f},{float(c_arr.max()):.3f}] "
          f"P[{float(P_arr.min()):.3e},{float(P_arr.max()):.3e}] "
          f"(residual={residual}, {device})")


if __name__ == "__main__":
    main()
