# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""给安装版 physicsnemo 的 SpectralConv 打 AMP 补丁 (运行时猴补丁, 不改库文件)。

问题: cuFFT 在 half 精度下只支持 2 的幂的信号尺寸。Elder 网格经 padding 后是
非 2^N (如 [72, 264]), 在 use_amp=true 时 physicsnemo 自带 SpectralConv2d.forward
里直接 ``torch.fft.rfft2(x)`` 会抛:
    RuntimeError: cuFFT only supports dimensions whose sizes are powers of two
    when computing in half precision ...

修复 (与源码仓库 spectral_layers.py 一致): 把整个 spectral conv 关掉 autocast、
在 fp32 下跑 FFT, 再把结果转回原 dtype, 让网络其余部分继续走 fp16。

适用范围:
- vanilla FNO (arch=fno, 用 physicsnemo 的 SpectralConv2d) -> 需要本补丁
- U-FNO (arch=ufno, 用本仓 ufno.py 自带的 SpectralConv2d, 已内联修复) -> 不受影响,
  打了也无害 (多套一层 autocast-off + float, 结果等价)。

用法: train_elder_fno.py 顶部已 ``import patch_spectral_amp`` (导入即生效)。
也可手动 ``from patch_spectral_amp import apply; apply()``。
"""
from __future__ import annotations

import torch

try:
    from physicsnemo.nn.module.spectral_layers import SpectralConv2d as _SpectralConv2d
except Exception:  # pragma: no cover - physicsnemo 未装时静默
    _SpectralConv2d = None

_applied = False


def apply() -> bool:
    """包住 SpectralConv2d.forward: 关 autocast + 输入转 fp32, 输出转回原 dtype。

    返回 True 表示打了补丁; False 表示无 SpectralConv2d 可打 (环境里没有)。
    幂等: 重复调用不会重复包裹。
    """
    global _applied
    if _applied or _SpectralConv2d is None:
        return _applied or False

    orig_forward = _SpectralConv2d.forward

    def _patched_forward(self, x):
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = orig_forward(self, x.float())          # FFT 全程 fp32
        return out.to(orig_dtype)                        # 转回 fp16 让网络继续走 AMP

    _SpectralConv2d.forward = _patched_forward           # 猴补丁: 替换类方法
    _applied = True
    return True


apply()   # 导入即生效
