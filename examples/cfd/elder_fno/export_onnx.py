# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""把训练好的 Elder FNO 导出成 ONNX, 供 C++ / ONNX Runtime 等部署。

导出内容:
  elder_fno.onnx     纯场算子 [1,2,64,256] (c, h_normalized) -> [1,2,64,256] (c_{n+1}, h_{n+1})
                    残差重建 (若 residual=true) 已包进图里, 故 ONNX 永远输出完整场。
                    归一化 P<->h 不在图里 (网格相关, 留给宿主代码做)。
  p_hydro.npy        静水压参考场 [64,256] (Pa), 宿主做 h=(P-p_hydro)/p_scale 用。
  onnx_meta.json     p_scale / 网格尺寸 / dt / 归一化公式 / 输入输出约定。

宿主 (C++) 调用流程:
    h = (P - p_hydro) / p_scale          # 输入归一化
    [c1, h1] = ONNX([c, h])              # 单步前向 (静态形状 1x2x64x256)
    P1 = h1 * p_scale + p_hydro          # 输出还原

用法:
    python export_onnx.py --checkpoint outputs_baseline30days/checkpoints --out elder_fno_30d.onnx
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils import load_checkpoint

from vtu_dataset import VtuElderDataset
from train_elder_fno import _resolve_fno_modes


class InferenceWrapper(torch.nn.Module):
    """把 FNO 包成"总是输出完整场"的算子 (残差模式自动重建), 简化宿主侧调用。"""

    def __init__(self, fno: FNO, residual: bool):
        super().__init__()
        self.fno = fno
        self.residual = bool(residual)

    def forward(self, ch: torch.Tensor) -> torch.Tensor:
        raw = self.fno(ch)                       # [B,2,Ny,Nx] 直接值 或 增量(残差)
        if self.residual:                        # 重建完整场 c_{n+1}=c_n+Δc, h_{n+1}=h_n+Δh
            return torch.cat([ch[:, 0:1] + raw[:, 0:1], ch[:, 1:2] + raw[:, 1:2]], dim=1)
        return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs_baseline30days/checkpoints")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="elder_fno_30d.onnx")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    phy = cfg.physics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DistributedManager.initialize()

    # 数据集只为拿网格 / p_hydro / p_scale (file_stride 须与训练一致)
    dp = VtuElderDataset(
        cfg.data.train_dir, 1, device, phy.phi, phy.Dm, phy.permeability, phy.viscosity,
        phy.g, phy.rho_f, phy.drho, phy.W, phy.H, phy.dt_macro,
        file_stride=int(cfg.data.get("file_stride", 1)),
    )
    mdl = cfg.model
    fno = FNO(
        in_channels=mdl.in_channels, out_channels=mdl.out_channels,
        decoder_layers=mdl.decoder_layers, decoder_layer_size=mdl.decoder_layer_size,
        dimension=mdl.dimension, latent_channels=mdl.latent_channels,
        num_fno_layers=mdl.num_fno_layers,
        num_fno_modes=_resolve_fno_modes(
            OmegaConf.to_container(mdl, resolve=True)["num_fno_modes"], dp, mdl.padding),
        padding=mdl.padding,
    ).to(device)
    load_checkpoint(path=args.checkpoint, models=fno, device=device)
    fno.eval()

    use_residual = bool(cfg.training.get("residual", False))
    model = InferenceWrapper(fno, use_residual).to(device).eval()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    base = os.path.splitext(os.path.basename(args.out))[0]

    # --- 导出 ONNX (静态形状 1x2x64x256) ---
    dummy = torch.randn(1, mdl.in_channels, dp.Ny_tot, dp.Nx_tot, device=device)
    torch.onnx.export(
        model, (dummy,), args.out,
        input_names=["ch_in"], output_names=["ch_out"],
        opset_version=args.opset,
        dynamic_axes=None,                       # 固定形状, C++ 端最省心
    )
    print(f"exported ONNX: {args.out}")

    # --- sidecar: p_hydro + meta ---
    p_hydro_path = os.path.join(out_dir, f"{base}_p_hydro.npy")
    meta_path = os.path.join(out_dir, f"{base}_meta.json")
    np.save(p_hydro_path, dp.p_hydro.cpu().numpy())
    meta = {
        "p_scale": float(dp.p_scale),
        "Ny_tot": int(dp.Ny_tot), "Nx_tot": int(dp.Nx_tot),
        "dt_macro_s": float(dp.dt_macro), "step_days": float(dp.dt_macro / 86400.0),
        "file_stride": int(dp.file_stride),
        "residual": use_residual,
        "input": "ch_in [1,2,Ny_tot,Nx_tot] = concat(c[0,1], h_normalized), h=(P-p_hydro)/p_scale",
        "output": "ch_out [1,2,Ny_tot,Nx_tot] = concat(c_{n+1}, h_{n+1}_normalized); P_{n+1}=h*p_scale+p_hydro",
        "p_hydro_file": os.path.basename(p_hydro_path),
        "note": "h 是等淡水水头归一化; p_hydro=ρf·g·z 每行 (向下 z 增). c∈[0,1], P 单位 Pa.",
    }
    json.dump(meta, open(meta_path, "w"), indent=2, ensure_ascii=False)
    print(f"sidecar: {p_hydro_path}, {meta_path}")

    # --- 校验: ONNX 输出 vs PyTorch 输出 (需 onnxruntime) ---
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n(onnxruntime 未装, 跳过数值校验。装一下可验证: pip install onnxruntime)")
        return
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    x = dummy.cpu().numpy()
    onnx_out = sess.run(None, {"ch_in": x})[0]
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()
    diff = float(np.abs(onnx_out - pt_out).max())
    print(f"\n校验: max|onnx - torch| = {diff:.2e}  ({'OK ✓' if diff < 1e-3 else '⚠ 偏大, 检查 FFT 导出'})")
    print(f"  输入形状 {x.shape} -> 输出形状 {onnx_out.shape}")


if __name__ == "__main__":
    main()
