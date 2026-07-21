# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physics-informed FNO (PINO) training for the Elder problem (c, p).

The model learns the single-step joint operator ``(c_n, p_n) -> (c_{n+1}, p_{n+1})``
over a macro time step ``dt`` for the variable-density, non-Boussinesq Elder
problem. Pressure is presented to the network in the equivalent-freshwater-head
gauge ``h = p - p_hydro`` (normalized by ``p_scale``); real pressure is recovered
for the physics residual. Training combines:

* a **data loss** (MSE on c and on the normalized head h vs. the reference), and
* a **PDE-residual loss** with two residuals evaluated on the prediction:
  - the conservative transport residual in c,
  - the flow / continuity residual ``div(rho q)`` in p (via the Darcy velocity
    ``q = -(k/mu)(grad p - rho g)`` reconstructed from the predicted p and c).

The Elder domain is *non-periodic* (wall-bounded), so the ``own_fd`` backend
computes the residuals with hand-written non-periodic central finite
differences (correct all the way to the walls).

GPU readiness knobs (in ``config.yaml``):
* ``training.use_amp``  - mixed precision (autocast + GradScaler).
* ``training.tf32``     - enable TF32 matmul/conv on Ampere+.
* ``seed``              - reproducibility (per-rank seeding under DDP).
"""

from __future__ import annotations

import glob
import json
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from datapipe import ElderProblem2D
from elder_residual_fv import ElderPhysics, form_function_elder
from vts_dataset import VtsElderDataset
from vtu_dataset import VtuElderDataset


def _enable_tf32() -> None:
    """Allow TF32 for matmul/conv on Ampere+ GPUs (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _zero_init_last_linear(module: torch.nn.Module) -> None:
    """把模块里最后一个 Linear 层的权重和偏置置零 (残差预测用)。

    残差模式下 FNO 输出应解释为增量 Δ; 零初始化 decoder 末层 => 初始 Δ=0 =>
    c_{n+1}=c_n, h_{n+1}=h_n (即"不变"基线), 训练从此起步只会更好。
    """
    last_lin = None
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            last_lin = m
    if last_lin is not None:
        torch.nn.init.zeros_(last_lin.weight)
        if last_lin.bias is not None:
            torch.nn.init.zeros_(last_lin.bias)


def _resolve_fno_modes(raw, datapipe, padding):
    """Resolve ``num_fno_modes`` to an explicit ``[modes_y, modes_x]`` list.

    A float in ``(0, 1]`` (or a list of such, one per axis) is interpreted as a
    *fraction* of the Fourier modes available to the spectral convolution,
    which operates on the padded grid ``(Ny_tot + pad, Nx_tot + pad)``. This
    auto-scales with resolution. Integers pass through unchanged.

    The per-axis cap is ``n // 2`` (n = padded dim): the y axis keeps both low-
    and high-frequency halves (so 2*modes_y <= n), and using ``n//2`` for x is a
    conservative choice that avoids the rfft Nyquist edge. Fractions are floored
    and clamped to >= 1.
    """
    vals = [raw, raw] if isinstance(raw, (int, float)) else list(raw)
    pad = int(padding)
    out = []
    for v, n_tot in zip(vals, (datapipe.Ny_tot + pad, datapipe.Nx_tot + pad)):
        if isinstance(v, float) and 0.0 < v <= 1.0:
            out.append(max(1, int(v * (n_tot // 2))))
        else:
            out.append(int(v))
    return out


def _load_loss_history(out_dir):
    """Load persisted train/val loss history so the loss curve continues across resume."""
    path = os.path.join(out_dir, "loss_history.json")
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        d = json.load(f)
    return d.get("train", []), d.get("val", [])


def _save_loss_history(out_dir, train_history, val_history):
    """Persist train/val loss history (called every epoch)."""
    path = os.path.join(out_dir, "loss_history.json")
    with open(path, "w") as f:
        json.dump({"train": list(train_history), "val": list(val_history)}, f)


def _build_residual_mask(datapipe, mask_top_rows: int, device) -> torch.Tensor:
    """Mask of shape ``[Ny, Nx]`` (interior) for the PDE residual.

    Zeros out the top ``mask_top_rows`` interior rows. The top boundary layer
    (the imposed ``c = 1`` source segment and the stiff diffusion front just
    below it) does not satisfy the transport PDE in the same way and is excluded
    from the residual; the fingering region deeper in the interior provides the
    physics signal.
    """
    Ny, Nx = datapipe.Ny, datapipe.Nx
    mask = torch.ones(Ny, Nx, device=device)
    n = max(0, int(mask_top_rows))
    if n:
        mask[:n, :] = 0.0
    return mask


def _grads(field, dx, dy):
    """First derivatives of a full-grid field at the interior cells."""
    f_x = (field[..., 1:-1, 2:] - field[..., 1:-1, :-2]) / (2.0 * dx)
    f_z = (field[..., 2:, 1:-1] - field[..., :-2, 1:-1]) / (2.0 * dy)
    return f_x, f_z


def _div(fx, fz, dx, dy):
    """Divergence of a full-grid flux field at the interior cells."""
    fx_x = (fx[..., 1:-1, 2:] - fx[..., 1:-1, :-2]) / (2.0 * dx)
    fz_z = (fz[..., 2:, 1:-1] - fz[..., :-2, 1:-1]) / (2.0 * dy)
    return fx_x + fz_z


def _residuals_own_fd(pred_c, pred_p, c0, dt, dp, mask):
    """Non-periodic finite-difference transport + continuity residuals.

    All quantities are evaluated at the interior cells (the full grid includes
    walls). The gradient flows through the predictions ``(pred_c, pred_p)``;
    ``c0`` is the reference initial concentration. ``dp`` is the datapipe
    (providing physical parameters and grid spacing).
    """
    rho_f, drho = dp.rho_f, dp.drho       # 淡水密度, 密度差
    phi, Dm = dp.phi, dp.Dm               # 孔隙度, 分子扩散系数
    kom, gz = dp.k_over_mu, dp.gz         # k/mu, 带符号重力 (flow_sign*g)
    dx, dy = dp.dx, dp.dy                 # 网格间距 (方形网格 dx=dy)

    # 物理尺度, 使 SI 单位的残差归一化到 O(1): 输运残差按累积率 phi*rho_f/dt 缩放,
    # 连续性残差按浮力驱动质量通量率 rho_f*q_ref/H 缩放 (q_ref = k*drho*g/mu)。
    scale_c = phi * rho_f / dt            # 输运残差的归一化尺度
    q_ref = kom * drho * abs(gz)          # 浮力驱动的特征 Darcy 速度
    scale_p = rho_f * q_ref / dp.H        # 连续性残差的归一化尺度

    # 由预测的 (c, p) 在全网格上构造密度与 Darcy 速度。
    rho = rho_f + drho * pred_c           # 变密度 rho(c)
    p_x, p_z = _grads(pred_p, dx, dy)     # 压力的 x/z 一阶导 (内部 cell)
    qx = -kom * p_x                       # Darcy 速度 x 分量 q = -(k/mu) dp/dx
    qz = -kom * (p_z - rho[..., 1:-1, 1:-1] * gz)   # z 分量含浮力: q_z = -(k/mu)(dp/dz - rho*g)
    # 把速度提升回全网格 (壁面镜像) 以便求散度。
    qx_full = F.pad(qx, (1, 1, 1, 1), mode="replicate")
    qz_full = F.pad(qz, (1, 1, 1, 1), mode="replicate")
    rhoq_x = rho * qx_full                # 质量通量 rho*q 的 x 分量
    rhoq_z = rho * qz_full                # 质量通量 rho*q 的 z 分量

    # 连续性残差: d(phi rho)/dt + div(rho q)
    #   其中 d(phi rho)/dt = phi*drho*dc/dt (rho = rho_f + drho*c, phi 常数)。
    pc = pred_c[..., 1:-1, 1:-1]          # 预测浓度的内部切片
    c0i = c0[..., 1:-1, 1:-1]             # 初始 (上一步) 浓度的内部切片
    storage = phi * drho * (pc - c0i) / dt   # 流体质量存储项 d(phi rho)/dt
    R_p = storage + _div(rhoq_x, rhoq_z, dx, dy)   # 连续性残差

    # 输运残差: d(phi rho c)/dt + div(rho q c) - div(rho phi Dm grad c)
    # 其中 d(phi rho c)/dt = phi (rho_f + 2 drho c) (c_pred - c0)/dt。
    time_term = phi * (rho_f + 2.0 * drho * pc) * (pc - c0i) / dt   # 守恒时间项
    adv = _div(rhoq_x * pred_c, rhoq_z * pred_c, dx, dy)           # 对流通量散度 div(rho q c)
    # 扩散: div(rho phi Dm grad c), 用面系数中心差分, 壁面=0。
    kd = rho * phi * Dm                   # 扩散系数场 rho*phi*Dm
    cp = pred_c                           # 浓度全网格
    cc = cp[..., 1:-1, 1:-1]              # 当前 cell 浓度
    ke = 0.5 * (kd[..., 1:-1, 1:-1] + kd[..., 1:-1, 2:])    # 东界面扩散系数
    kw = 0.5 * (kd[..., 1:-1, 1:-1] + kd[..., 1:-1, :-2])   # 西界面
    kn = 0.5 * (kd[..., 1:-1, 1:-1] + kd[..., 2:, 1:-1])    # 北界面 (+z 下侧)
    ks = 0.5 * (kd[..., 1:-1, 1:-1] + kd[..., :-2, 1:-1])   # 南界面 (-z 上侧)
    diff = (
        (ke * (cp[..., 1:-1, 2:] - cc) - kw * (cc - cp[..., 1:-1, :-2])) / dx**2     # x 方向扩散散度
        + (kn * (cp[..., 2:, 1:-1] - cc) - ks * (cc - cp[..., :-2, 1:-1])) / dy**2   # z 方向扩散散度
    )
    R_c = time_term + adv - diff          # 输运残差

    m = mask.view(1, 1, *mask.shape)      # 残差掩码 reshape 成 [1,1,Ny,Nx] 便于广播
    n = m.sum().clamp(min=1.0)            # 掩码内 cell 数 (至少 1, 防除零)
    loss_c = (R_c.abs() * m).sum() / (n * scale_c)   # 输运残差: 掩码内平均 |R_c|, 按尺度归一化
    loss_p = (R_p.abs() * m).sum() / (n * scale_p)   # 连续性残差: 同理
    return loss_c, loss_p


def validation_step(model, datapipe, p_hydro, p_scale, num_iters, epoch, out_dir, device, use_amp,
                    val_history=None, train_history=None, residual=False):
    """Evaluate MSE on fresh trajectory samples and save a comparison plot.

    ``val_history`` / ``train_history`` are mutable lists the caller passes in;
    this call appends its result to ``val_history`` and plots both curves at
    the bottom of the figure (log y-axis). Pass ``None`` to skip the curve.
    """
    model.eval()
    total_c, total_h, count = 0.0, 0.0, 0       # AI 预测的 MSE 累计 (|AI_pred - new|)
    total_old_c, total_old_h = 0.0, 0.0          # old 基线 MSE 累计 (|old - new|, 即 |c0-c1|)
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    last = None
    best_t0 = -1.0                          # 跟踪物理时间最长 (指进最发育) 的样本用于展示
    dt_days = datapipe.dt_macro / (24 * 3600.0)
    with torch.no_grad():                          # 验证不需要梯度
        for batch, _ in zip(datapipe, range(num_iters)):
            c0 = batch["c0"]                       # 初始 (上一步) 浓度 [B,1,Ny+2,Nx+2]
            h0 = (batch["p0"] - p_hydro) / p_scale  # 归一化水头 h = (p - p_hydro)/p_scale
            c1 = batch["c1"]                       # 目标浓度 (真值)
            h1 = (batch["p1"] - p_hydro) / p_scale  # 目标归一化水头 (真值)
            invar = torch.cat([c0, h0], dim=1)     # 拼成 2 通道输入
            with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                raw = model(invar)               # 单步前向: 直接预测值 (或残差模式下的增量 Δ)
            if residual:                         # 残差预测: c_{n+1}=c_n+Δc, h_{n+1}=h_n+Δh
                pred_c = c0 + raw[:, 0:1]
                pred_h = h0 + raw[:, 1:2]
            else:                                # 直接预测全场
                pred_c, pred_h = raw[:, 0:1], raw[:, 1:2]
            total_c += F.mse_loss(pred_c, c1).item()      # 累加 AI 的 c MSE (|AI_pred - c1|)
            total_h += F.mse_loss(pred_h, h1).item()      # 累加 AI 的 h MSE
            total_old_c += F.mse_loss(c0, c1).item()      # 累加 old 基线 c MSE (|c0 - c1|)
            total_old_h += F.mse_loss(h0, h1).item()      # 累加 old 基线 h MSE (|h0 - h1|)
            count += 1
            t0 = batch.get("t0")                   # 每个样本的物理时间 (天), 可能为 None
            mi = int(torch.argmax(t0).item()) if t0 is not None else 0   # batch 内时间最长的样本下标
            t_max = float(t0[mi].item()) if t0 is not None else 0.0      # 该样本的时间
            if t_max > best_t0:                    # 跨 batch 保留时间最长 (指进最发育) 的样本用于画图
                best_t0 = t_max
                # 单步 preconditioner 视角: old=c0(输入), AI_pred=pred, new=c1(真值)。
                last = (c1[mi, 0], c0[mi, 0], pred_c[mi, 0], h1[mi, 0], h0[mi, 0], pred_h[mi, 0])

    model.train()                                  # 恢复训练模式 (验证时切到了 eval)
    mean_c = total_c / max(count, 1)               # 平均 AI c MSE
    mean_h = total_h / max(count, 1)               # 平均 AI h MSE
    mean_old_c = total_old_c / max(count, 1)       # 平均 old 基线 c MSE
    mean_old_h = total_old_h / max(count, 1)       # 平均 old 基线 h MSE

    if last is not None:
        # last = (c_new, c_old, c_pred, h_new, h_old, h_pred), 转 numpy 画图。
        c_true, c_old, c_pred, h_true, h_old, h_pred = (t.detach().cpu().numpy() for t in last)
        # 3 行 GridSpec: 上两行 c/h 各 5 列 (True|True old|Pred|AI误差|old误差), 第三行跨 5 列画 loss 曲线。
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.5], hspace=0.5, wspace=0.35)
        ax = [[fig.add_subplot(gs[0, c]) for c in range(5)],
              [fig.add_subplot(gs[1, c]) for c in range(5)]]
        ax_loss = fig.add_subplot(gs[2, :])

        t1_days = best_t0 + dt_days
        # 单步 preconditioner 指标: AI MSE < old MSE 即 AI 比输入(old)更靠近真值(new)。
        fig.suptitle(
            f"Elder FNO Validation - Epoch {epoch} | Physical Time: {t1_days:.0f} days\n"
            f"AI  MSE_c = {mean_c:.3e}   old MSE_c = {mean_old_c:.3e}   |   "
            f"AI MSE_h = {mean_h:.3e}   old MSE_h = {mean_old_h:.3e}",
            fontsize=15, fontweight="bold",
        )
        col_titles = ["True (new)", "True old (input)", "Pred (AI)", "|Pred-True|", "|Old-True|"]
        for row in range(2):
            if row == 0:                                  # c 行
                true_f, old_f, pred_f = c_true, c_old, c_pred
                field_vmin, field_vmax = 0.0, 1.0         # c 用物理范围 [0,1]
            else:                                         # h 行
                true_f, old_f, pred_f = h_true, h_old, h_pred
                field_vmin, field_vmax = float(true_f.min()), float(true_f.max())   # h 用真值范围
            err_pred = np.abs(pred_f - true_f)            # AI 误差
            err_old = np.abs(old_f - true_f)              # old 基线误差
            err_vmax = max(float(err_pred.max()), float(err_old.max()))   # 两误差列共用上界, 可直接对比
            fields = [true_f, old_f, pred_f, err_pred, err_old]
            for col in range(5):
                f = fields[col]
                vmin, vmax = (field_vmin, field_vmax) if col < 3 else (0.0, err_vmax)
                im = ax[row][col].imshow(f, origin="upper", vmin=vmin, vmax=vmax)
                ax[row][col].set_title(("c: " if row == 0 else "h: ") + col_titles[col], fontsize=10)
                plt.colorbar(im, ax=ax[row][col], fraction=0.046)

        # --- loss 曲线 (log y): AI vs old 基线 ---
        if val_history is not None:
            val_history.append((epoch, mean_c + mean_h, mean_c, mean_h, mean_old_c + mean_old_h))
        if val_history:
            ve = [h[0] for h in val_history]
            ax_loss.semilogy(ve, [h[1] for h in val_history], "o-", label="val AI total")
            ax_loss.semilogy(ve, [h[2] for h in val_history], ".--", alpha=0.7, label="val AI MSE_c")
            ax_loss.semilogy(ve, [h[3] for h in val_history], ".--", alpha=0.7, label="val AI MSE_h")
            if len(val_history[0]) > 4:                   # 有 old 基线时画出
                ax_loss.semilogy(ve, [h[4] for h in val_history], "x--", alpha=0.7, label="val old total (baseline)")
        if train_history:
            te = [h[0] for h in train_history]
            ax_loss.semilogy(te, [h[1] for h in train_history], "-", alpha=0.8, label="train loss_data")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss (log)")
        ax_loss.set_title("Loss curves (AI vs old baseline)")
        ax_loss.legend(loc="best", fontsize=8)
        ax_loss.grid(True, which="both", alpha=0.3)

        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        fig.savefig(os.path.join(out_dir, f"val_{epoch:04d}.png"))
        plt.close(fig)
    return mean_c + mean_h


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    seed = int(cfg.get("seed", 0))
    if seed:
        torch.manual_seed(seed + dist.rank)
        np.random.seed(seed + dist.rank)

    if cfg.training.get("tf32", True):
        _enable_tf32()

    log = PythonLogger(name="elder_fno")           # 控制台 + 文件日志
    log.file_logging()                             # 同时写 train_elder_fno.log
    LaunchLogger.initialize()                      # physicsnemo 的训练日志器

    use_amp = bool(cfg.training.get("use_amp", False)) and device.type == "cuda"   # 仅 CUDA 启用混合精度
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"   # autocast 设备类型
    scaler = torch.amp.GradScaler(device=autocast_dev, enabled=use_amp)   # AMP 梯度缩放器

    # --- data ----------------------------------------------------------------
    dp_kwargs = dict(
        resolution=cfg.data.resolution,
        batch_size=cfg.data.batch_size,
        phi=cfg.physics.phi,
        Dm=cfg.physics.Dm,
        permeability=cfg.physics.permeability,
        viscosity=cfg.physics.viscosity,
        g=cfg.physics.g,
        rho_f=cfg.physics.rho_f,
        drho=cfg.physics.drho,
        W=cfg.physics.W,
        H=cfg.physics.H,
        source_frac=cfg.data.source_frac,
        p_scale=cfg.physics.get("p_scale", None),
        dt_macro=cfg.physics.dt_macro,
        flow_sign=cfg.physics.get("flow_sign", 1.0),
        substeps=cfg.data.substeps,
        max_substeps=cfg.data.max_substeps,
        n_trajectories=cfg.data.n_trajectories,
        rollout_steps=cfg.data.rollout_steps,
        device=device,
    )
    # --- 数据源: 优先真实数据 (.vtu 高精度 unisolver / .vts), 否则回退在线 datapipe ---
    # hydra.job.chdir=True 会把 cwd 切到 run.dir, 故相对 train_dir 须相对原始 cwd 解析,
    # 否则去 run.dir/train_dir 找 (不存在) 而误回退在线 datapipe。
    train_dir = cfg.data.get("train_dir", "train_set")
    if not os.path.isabs(train_dir):
        train_dir = os.path.join(hydra.utils.get_original_cwd(), train_dir)
    vtu_files = glob.glob(os.path.join(train_dir, "*.vtu")) if os.path.isdir(train_dir) else []
    vts_files = glob.glob(os.path.join(train_dir, "*.vts")) if os.path.isdir(train_dir) else []
    # vtu 与 vts 真实数据集共用同一组物理/网格参数 (接口一致, 见各自 __init__)
    real_kwargs = dict(
        batch_size=cfg.data.batch_size, device=device,
        phi=cfg.physics.phi, Dm=cfg.physics.Dm, permeability=cfg.physics.permeability,
        viscosity=cfg.physics.viscosity, g=cfg.physics.g, rho_f=cfg.physics.rho_f,
        drho=cfg.physics.drho, W=cfg.physics.W, H=cfg.physics.H,
        dt_macro=cfg.physics.dt_macro, flow_sign=cfg.physics.get("flow_sign", 1.0),
    )
    if len(vtu_files) >= 2:
        val_frac = float(cfg.data.get("val_frac", 0.2))
        n_val_blocks = int(cfg.data.get("n_val_blocks", 8))
        val_gap = int(cfg.data.get("val_gap", 2))
        file_stride = int(cfg.data.get("file_stride", 1))
        log.info(f"using VTU dataset from {train_dir} ({len(vtu_files)} files, file_stride={file_stride}), "
                 f"val_frac={val_frac}, n_val_blocks={n_val_blocks}, val_gap={val_gap}")
        datapipe = VtuElderDataset(train_dir, split="train", val_frac=val_frac,
                                   n_val_blocks=n_val_blocks, val_gap=val_gap,
                                   file_stride=file_stride, **real_kwargs)
        # val 拆成 n_val_blocks 个小块均匀散布 (覆盖早/中/晚多时段), _share 复用已加载数据;
        # 每块两侧 val_gap 个快照 + 跨集合配对一律丢弃 (零泄漏, 每个 snapshot 只属一个集合)。
        val_datapipe = VtuElderDataset(train_dir, split="val", val_frac=val_frac,
                                       n_val_blocks=n_val_blocks, val_gap=val_gap,
                                       file_stride=file_stride, _share=datapipe, **real_kwargs)
        log.info(f"  train pairs={datapipe.n_pairs}, val pairs={val_datapipe.n_pairs} "
                 f"@ {datapipe.dt_macro / 86400:.0f}-day/step")
    elif len(vts_files) >= 2:
        log.info(f"using VTS dataset from {train_dir} ({len(vts_files)} files)")
        datapipe = VtsElderDataset(train_dir, **real_kwargs)   # 真实求解器 point 数据
        val_datapipe = datapipe                                 # 同上: __iter__ 每次起独立生成器, 只读不改
    else:
        if os.path.isdir(train_dir):
            log.info(f"{train_dir}/ 中 .vtu/.vts 不足 2 个, 回退在线 datapipe")
        else:
            log.info(f"无 {train_dir}/, 回退在线 datapipe")
        datapipe = ElderProblem2D(**dp_kwargs)      # 在线参考解生成器
        val_datapipe = ElderProblem2D(**dp_kwargs)  # 验证数据生成器 (独立轨迹)

    p_hydro = datapipe.p_hydro                      # 淡水静水压参考场 rho_f*g*z (Pa)
    p_scale = datapipe.p_scale                      # 水头归一化尺度 (经验 max|h| 或显式值)

    # --- model ---------------------------------------------------------------
    model = FNO(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        decoder_layers=cfg.model.decoder_layers,
        decoder_layer_size=cfg.model.decoder_layer_size,
        dimension=cfg.model.dimension,
        latent_channels=cfg.model.latent_channels,
        num_fno_layers=cfg.model.num_fno_layers,
        num_fno_modes=_resolve_fno_modes(
            OmegaConf.to_container(cfg.model, resolve=True)["num_fno_modes"],
            datapipe, cfg.model.padding,
        ),
        padding=cfg.model.padding,
    ).to(device)

    # 残差预测: 网络输出解释为增量 Δ, 重建 c_{n+1}=c_n+Δc / h_{n+1}=h_n+Δh。
    use_residual = bool(cfg.training.get("residual", False))
    if use_residual:
        zero_init = bool(cfg.training.get("zero_init", False))
        if zero_init:
            _zero_init_last_linear(model.decoder_net)   # 起步=不变基线 (但可能因梯度弱而卡住)
            log.info("residual prediction ON + decoder 零初始化 (起步=不变基线)")
        else:
            log.info("residual prediction ON (无零初始化, 标准残差结构)")

    if cfg.training.get("compile", False):
        forward_model = torch.compile(            # 编译前向以加速 (CUDA, 长训练值得)
            model, mode=cfg.training.get("compile_mode", "default")
        )
        log.info(f"torch.compile enabled (mode={cfg.training.compile_mode})")
    else:
        forward_model = model                      # 不编译, 直接用原模型

    # --- physics residual evaluator ------------------------------------------
    use_physics = cfg.training.physics_weight > 0  # 是否启用 PDE 残差损失
    physics_backend = str(cfg.training.get("physics_backend", "own_fd")).lower()
    residual_mask = None
    elder_phy = None
    if use_physics:
        if physics_backend == "unisolver_fv":
            # 与 Elder::FormFunction 对齐的 FV 残差 (见 elder_residual_fv.py)
            elder_phy = ElderPhysics(
                phi=float(cfg.physics.phi),
                perm=float(cfg.physics.permeability),
                visc=float(cfg.physics.viscosity),
                Dm=float(cfg.physics.Dm),
                rho_f=float(cfg.physics.rho_f),
                drho=float(cfg.physics.drho),
                g=float(cfg.physics.g),
                W=float(cfg.physics.W),
                H=float(cfg.physics.H),
            )
            log.info("physics_backend=unisolver_fv (match Elder FormFunction)")
        else:
            physics_backend = "own_fd"
            residual_mask = _build_residual_mask(
                datapipe,
                mask_top_rows=cfg.training.get("mask_top_rows", 2),
                device=device,
            )
            log.info("physics_backend=own_fd (legacy central FD + mask)")

    optimizer = Adam(model.parameters(), lr=cfg.training.start_lr)   # Adam 优化器
    scheduler = ExponentialLR(optimizer, gamma=cfg.training.gamma)   # 每 epoch 衰减 lr

    ckpt_args = {
        "path": "./checkpoints",                   # checkpoint 目录 (相对 hydra run dir)
        "models": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    start_epoch = load_checkpoint(device=device, **ckpt_args)   # 恢复 (返回已训练到的 epoch; 无则 0)
    out_dir = os.getcwd()                          # hydra 已把 cwd 切到 run dir (输出根目录)

    steps_per_epoch = cfg.data.steps_per_epoch     # 每 epoch 的 batch 数
    val_iters = max(1, cfg.validation.sample_size // cfg.data.batch_size)   # 验证迭代数
    physics_weight = cfg.training.physics_weight   # PDE 项权重 (0 = 关闭)
    p_data_weight = float(cfg.training.get("p_data_weight", 1.0))   # h 数据项权重
    continuity_weight = float(cfg.training.get("continuity_weight", 1.0))   # 连续性/transport 权重

    # 跨 epoch 的 loss 历史, 供验证图绘制 loss 曲线 (log y)。仅续训时从磁盘加载 (全新 run 清空)。
    # 截断到 start_epoch, 避免 checkpoint (每 save_every 存) 与每 epoch 存的 loss 历史错位导致重复点。
    train_history, val_history = (_load_loss_history(out_dir) if start_epoch > 0 else ([], []))
    train_history = [h for h in train_history if h[0] <= start_epoch]
    val_history = [h for h in val_history if h[0] <= start_epoch]

    if start_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming from epoch {start_epoch + 1}.")

    for epoch in range(max(1, start_epoch + 1), cfg.training.max_epochs + 1):   # 续训则从 start_epoch+1 开始
        model.train()                              # 训练模式 (dropout/BN 等)
        epoch_loss_sum = 0.0                       # 累加本 epoch 的 loss_data (用于 loss 曲线)
        epoch_loss_n = 0
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=steps_per_epoch, epoch_alert_freq=5
        ) as logger:
            for batch, _ in zip(datapipe, range(steps_per_epoch)):
                c0 = batch["c0"]                   # 上一步浓度
                p0 = batch["p0"]                   # 上一步压力 (Pa)
                c1 = batch["c1"]                   # 目标浓度
                p1 = batch["p1"]                   # 目标压力 (Pa)
                dt = batch["dt"]                   # macro 步长 (s), 用于 PDE 残差

                h0 = (p0 - p_hydro) / p_scale      # 归一化输入水头
                h1 = (p1 - p_hydro) / p_scale      # 归一化目标水头
                invar = torch.cat([c0, h0], dim=1)            # [B, 2, Ny+2, Nx+2]

                with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                    raw = forward_model(invar)                # [B,2,...] 直接预测值 (或残差模式下的增量 Δ)
                    if use_residual:                          # 残差预测: c_{n+1}=c_n+Δc, h_{n+1}=h_n+Δh
                        pred_c = c0 + raw[:, 0:1]
                        pred_h = h0 + raw[:, 1:2]
                    else:                                     # 直接预测全场
                        pred_c, pred_h = raw[:, 0:1], raw[:, 1:2]
                    loss_data = F.mse_loss(pred_c, c1) + p_data_weight * F.mse_loss(pred_h, h1)   # 数据 loss

                    loss_pde_c = torch.zeros((), device=device)   # PDE 残差默认 0
                    loss_pde_p = torch.zeros((), device=device)
                    if use_physics:
                        # PDE 残差在 fp32 下算 (关掉 autocast) 以保证有限差分精度。
                        with torch.autocast(device_type=autocast_dev, enabled=False):
                            pred_c32 = pred_c.float()
                            pred_p32 = pred_h.float() * p_scale + p_hydro
                            c0_32 = c0.float()
                            p0_32 = p0.float()
                            # dt may be 0-dim tensor or float
                            dt_val = dt.float() if torch.is_tensor(dt) else float(dt)
                            if physics_backend == "unisolver_fv":
                                # L2 residual fields matching Elder::FormFunction
                                Fp, Fc = form_function_elder(
                                    pred_c32, pred_p32, c0_32, p0_32, dt_val, elder_phy
                                )
                                # mean-square loss (scale-invariant to batch); log L2 norms
                                loss_pde_p = (Fp * Fp).mean()
                                loss_pde_c = (Fc * Fc).mean()
                                loss_pde = loss_pde_p + continuity_weight * loss_pde_c
                            else:
                                loss_pde_c, loss_pde_p = _residuals_own_fd(
                                    pred_c32, pred_p32, c0_32, dt_val,
                                    datapipe, residual_mask,
                                )
                                loss_pde = loss_pde_c + continuity_weight * loss_pde_p
                    else:
                        loss_pde = torch.zeros((), device=device)

                    loss = loss_data + physics_weight * loss_pde   # 总损失

                optimizer.zero_grad(set_to_none=True)   # 清梯度 (set_to_none 省显存)
                scaler.scale(loss).backward()           # AMP 缩放后反传
                scaler.step(optimizer)                  # 更新参数 (含梯度反缩放)
                scaler.update()                         # 更新缩放因子
                scheduler.step()                        # 每 step 衰减 lr (ExponentialLR)

                logger.log_minibatch(                   # 记录各 loss 到日志
                    {
                        "loss_data": loss_data.detach(),
                        "loss_pde": (loss_pde_c + loss_pde_p).detach(),
                        "loss_pde_c": loss_pde_c.detach(),
                        "loss_pde_p": loss_pde_p.detach(),
                    }
                )
                epoch_loss_sum += float(loss_data.detach())   # 累加用于 epoch 平均
                epoch_loss_n += 1
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            train_history.append((epoch, epoch_loss_sum / max(epoch_loss_n, 1)))   # 记录本 epoch 平均 loss_data

        if epoch % cfg.training.val_every == 0:     # 每 val_every epoch 验证一次
            with LaunchLogger("valid", epoch=epoch) as logger:
                val_loss = validation_step(         # 验证 + 出 val 图 + 更新 val_history
                    forward_model, val_datapipe, p_hydro, p_scale,
                    val_iters, epoch, out_dir, device, use_amp,
                    val_history=val_history, train_history=train_history,
                    residual=use_residual,
                )
                logger.log_epoch({"Validation error": val_loss})

        # 大文件 checkpoint 每 save_every epoch 存一次 (+ 最后一 epoch); loss 历史每 epoch 都存。
        if epoch % int(cfg.training.get("save_every", 10)) == 0 or epoch == cfg.training.max_epochs:
            save_checkpoint(**ckpt_args, epoch=epoch)
        _save_loss_history(out_dir, train_history, val_history)   # loss 历史每 epoch 存 (KB 级)

    log.success("Training completed *yay*")


if __name__ == "__main__":
    main()