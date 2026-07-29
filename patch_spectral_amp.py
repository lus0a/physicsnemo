#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""一次性补丁: 让 physicsnemo 的 SpectralConv2d.forward 在 AMP(fp16) 下用 fp32 跑 FFT。

问题: cuFFT 在半精度下只接受 2 的幂的信号尺寸, 而 Elder 网格 padding 后是 72x264
      (非 2 的幂), 在 vanilla FNO + use_amp=true 时报:
      RuntimeError: cuFFT only supports dimensions whose sizes are powers of two
      when computing in half precision ...

修法: 整个 spectral conv 用 autocast(enabled=False) + x.float() 跑 fp32,
      输出 .to(orig_dtype) 回 fp16, 网络其余部分仍走半精度。UFNO 用本地 SpectralConv
      不受影响, 无需打。

用法 (在 xjjsserver 训练机上, 激活 physicsnemo 环境后):
    python patch_spectral_amp.py            # 打补丁
    python patch_spectral_amp.py --revert   # 还原 (用 .bak_amp 备份)

幂等: 已打过会自动跳过。
"""
import argparse, shutil, sys
from pathlib import Path

try:
    import physicsnemo
except Exception as e:
    sys.exit(f"import physicsnemo 失败: {e}\n请先激活含 physicsnemo 的 conda 环境")

TARGET = Path(physicsnemo.__file__).resolve().parent / "nn" / "module" / "spectral_layers.py"
SENTINEL = "AMP fix: cuFFT in half precision"  # 新代码里的标记串

OLD = '''    def forward(
        self, x: Float[Tensor, "batch in_channels h w"]
    ) -> Float[Tensor, "batch out_channels h w"]:
        x_ft = torch.fft.rfft2(x)  # (batch, in_channels, h, w//2+1) complex
        h, w = x_ft.size(-2), x_ft.size(-1)  # h=h, w=w//2+1

        # Initialize output in frequency space
        out_ft = torch.zeros(
            x.size(0), self.out_channels, h, w, dtype=torch.cfloat, device=x.device
        )  # (batch, out_channels, h, w) complex

        # Accumulate Fourier modes. Use .contiguous() on sliced complex tensors and
        # padding (not slice assignment) for torch.compile compatibility.
        # Slice assignment causes gradient stride issues in the Inductor backward pass.
        # Pad format: (left, right, top, bottom) for last 2 dims
        out_ft = out_ft + F.pad(
            self.compl_mul2d(
                x_ft[:, :, : self.modes1, : self.modes2].contiguous(), self.weights1
            ),
            (0, w - self.modes2, 0, h - self.modes1),
        )
        out_ft = out_ft + F.pad(
            self.compl_mul2d(
                x_ft[:, :, -self.modes1 :, : self.modes2].contiguous(), self.weights2
            ),
            (0, w - self.modes2, h - self.modes1, 0),
        )

        # Return to physical space
        return torch.fft.irfft2(
            out_ft, s=(x.size(-2), x.size(-1))
        )  # (batch, out_channels, h, w) real
'''

NEW = '''    def forward(
        self, x: Float[Tensor, "batch in_channels h w"]
    ) -> Float[Tensor, "batch out_channels h w"]:
        # AMP fix: cuFFT in half precision requires power-of-2 signal sizes, which
        # fails on non-power-of-2 grids (e.g. padded 72x264). Disable autocast and
        # run the whole spectral conv in fp32, then cast the result back so the rest
        # of the network keeps running in fp16 under AMP.
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            x_ft = torch.fft.rfft2(x)  # (batch, in_channels, h, w//2+1) complex
            h, w = x_ft.size(-2), x_ft.size(-1)  # h=h, w=w//2+1

            # Initialize output in frequency space
            out_ft = torch.zeros(
                x.size(0), self.out_channels, h, w, dtype=torch.cfloat, device=x.device
            )  # (batch, out_channels, h, w) complex

            # Accumulate Fourier modes. Use .contiguous() on sliced complex tensors and
            # padding (not slice assignment) for torch.compile compatibility.
            # Slice assignment causes gradient stride issues in the Inductor backward pass.
            # Pad format: (left, right, top, bottom) for last 2 dims
            out_ft = out_ft + F.pad(
                self.compl_mul2d(
                    x_ft[:, :, : self.modes1, : self.modes2].contiguous(), self.weights1
                ),
                (0, w - self.modes2, 0, h - self.modes1),
            )
            out_ft = out_ft + F.pad(
                self.compl_mul2d(
                    x_ft[:, :, -self.modes1 :, : self.modes2].contiguous(), self.weights2
                ),
                (0, w - self.modes2, h - self.modes1, 0),
            )

            # Return to physical space
            out = torch.fft.irfft2(
                out_ft, s=(x.size(-2), x.size(-1))
            )  # (batch, out_channels, h, w) real
        return out.to(orig_dtype)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="用 .bak_amp 还原")
    args = ap.parse_args()

    if not TARGET.exists():
        sys.exit(f"找不到目标文件: {TARGET}")
    print(f"target: {TARGET}")

    if args.revert:
        bak = TARGET.with_suffix(".py.bak_amp")
        if not bak.exists():
            sys.exit(f"找不到备份: {bak}")
        shutil.copy2(bak, TARGET)
        print("已还原 (from .bak_amp)")
        return

    text = TARGET.read_text()
    if SENTINEL in text:
        print("已经打过补丁, 跳过")
        return
    if OLD not in text:
        sys.exit(
            "未找到原始 forward 代码块 (版本不匹配?)。请检查 physicsnemo 版本是否为 2.2.0a0, "
            "或手动给 SpectralConv2d.forward 套 autocast(enabled=False)+x.float()。"
        )
    bak = TARGET.with_suffix(".py.bak_amp")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"补丁成功 (备份: {bak})")


if __name__ == "__main__":
    main()
