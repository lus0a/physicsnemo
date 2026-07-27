# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""单步 FNO/U-FNO 推理 (oneshot: 每调用加载模型一次).

推荐生产 hybrid 使用持久化服务 (模型只 load 一次):
  python fno_server.py --checkpoint outputs_elder_ufno/checkpoints --port 8765
C++ FnoPredictor.mode = \"persist\" 经 TCP 连接, 不再每步 system(python).

Oneshot 用法 (调试 / 无 server 时的回退):
  python fno_step.py --in fno_in.bin --out fno_out.bin \\
                     --checkpoint outputs_elder_ufno/checkpoints --dt 864000
"""
from __future__ import annotations

import argparse
import warnings

from fno_infer_core import FnoEngine


def main():
    warnings.filterwarnings(
        "ignore",
        message="Could not initialize using ENV, SLURM or OPENMPI methods",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="C++ raw binary 输入")
    ap.add_argument("--out", required=True, help="写回 C++ 的 raw binary")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--checkpoint",
        default="outputs_elder_ufno/checkpoints",
        help="checkpoint 目录 (取最新 .mdlus) 或 load_checkpoint 路径",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="本时间步 Δt (秒); dt_channel=true 时必传",
    )
    ap.add_argument("--device", default=None, help="cuda|cpu|auto")
    args = ap.parse_args()

    engine = FnoEngine.load(args.config, args.checkpoint, device=args.device)
    if engine.dt_aware and args.dt is None:
        raise SystemExit("fno_step: dt_channel=true 需要 --dt <秒>")
    engine.predict_files(args.inp, args.out, float(args.dt if args.dt is not None else 0.0))


if __name__ == "__main__":
    main()
