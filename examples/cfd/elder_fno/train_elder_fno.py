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


def _enable_tf32() -> None:
    """Allow TF32 for matmul/conv on Ampere+ GPUs (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


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
    rho_f, drho = dp.rho_f, dp.drho
    phi, Dm = dp.phi, dp.Dm
    kom, gz = dp.k_over_mu, dp.gz
    dx, dy = dp.dx, dp.dy

    # Physical scales so the (SI) residuals are O(1): the transport residual
    # is scaled by the accumulation rate phi*rho_f/dt, the continuity residual
    # by the buoyancy-driven mass-flux rate rho_f*q_ref/H (q_ref = k*drho*g/mu).
    scale_c = phi * rho_f / dt
    q_ref = kom * drho * abs(gz)
    scale_p = rho_f * q_ref / dp.H

    # Density and Darcy velocity from predicted (c, p) on the full grid.
    rho = rho_f + drho * pred_c
    p_x, p_z = _grads(pred_p, dx, dy)
    qx = -kom * p_x
    qz = -kom * (p_z - rho[..., 1:-1, 1:-1] * gz)
    # Lift velocities to the full grid (replicate walls) for the divergence.
    qx_full = F.pad(qx, (1, 1, 1, 1), mode="replicate")
    qz_full = F.pad(qz, (1, 1, 1, 1), mode="replicate")
    rhoq_x = rho * qx_full
    rhoq_z = rho * qz_full

    # Continuity residual:  d(phi rho)/dt + div(rho q)
    #   with  d(phi rho)/dt = phi*drho*dc/dt  (rho = rho_f + drho*c, phi const).
    pc = pred_c[..., 1:-1, 1:-1]
    c0i = c0[..., 1:-1, 1:-1]
    storage = phi * drho * (pc - c0i) / dt
    R_p = storage + _div(rhoq_x, rhoq_z, dx, dy)

    # Transport residual:
    #   d(phi rho c)/dt + div(rho q c) - div(rho phi Dm grad c)
    # with  d(phi rho c)/dt = phi (rho_f + 2 drho c) (c_pred - c0)/dt.
    time_term = phi * (rho_f + 2.0 * drho * pc) * (pc - c0i) / dt
    adv = _div(rhoq_x * pred_c, rhoq_z * pred_c, dx, dy)
    # Diffusion: div(rho phi Dm grad c) via face coefficients, walls = 0.
    kd = rho * phi * Dm
    cp = pred_c
    cc = cp[..., 1:-1, 1:-1]
    ke = 0.5 * (kd[..., 1:-1, 1:-1] + kd[..., 1:-1, 2:])
    kw = 0.5 * (kd[..., 1:-1, 1:-1] + kd[..., 1:-1, :-2])
    kn = 0.5 * (kd[..., 1:-1, 1:-1] + kd[..., 2:, 1:-1])
    ks = 0.5 * (kd[..., 1:-1, 1:-1] + kd[..., :-2, 1:-1])
    diff = (
        (ke * (cp[..., 1:-1, 2:] - cc) - kw * (cc - cp[..., 1:-1, :-2])) / dx**2
        + (kn * (cp[..., 2:, 1:-1] - cc) - ks * (cc - cp[..., :-2, 1:-1])) / dy**2
    )
    R_c = time_term + adv - diff

    m = mask.view(1, 1, *mask.shape)
    n = m.sum().clamp(min=1.0)
    loss_c = (R_c.abs() * m).sum() / (n * scale_c)
    loss_p = (R_p.abs() * m).sum() / (n * scale_p)
    return loss_c, loss_p


def validation_step(model, datapipe, p_hydro, p_scale, num_iters, epoch, out_dir, device, use_amp):
    """Evaluate MSE on fresh trajectory samples and save a comparison plot."""
    model.eval()
    total_c, total_h, count = 0.0, 0.0, 0
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    last = None
    with torch.no_grad():
        for batch, _ in zip(datapipe, range(num_iters)):
            c0 = batch["c0"]
            h0 = (batch["p0"] - p_hydro) / p_scale
            c1 = batch["c1"]
            h1 = (batch["p1"] - p_hydro) / p_scale
            
            # 提取物理时间信息
            t0 = batch.get("t0", torch.zeros(c0.shape[0])) 
            dt_days = batch.get("dt", 0) / (24 * 3600.0)
            
            invar = torch.cat([c0, h0], dim=1)
            with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                pred = model(invar)
            pred_c, pred_h = pred[:, 0:1], pred[:, 1:2]
            total_c += F.mse_loss(pred_c, c1).item()
            total_h += F.mse_loss(pred_h, h1).item()
            count += 1
            
            # 智能选图: 寻找这个 batch 里推演物理时间最长的样本 (最能展示指进)
            max_idx = torch.argmax(t0).item()
            t1_days = t0[max_idx].item() + dt_days
            
            last = (c1[max_idx, 0], pred_c[max_idx, 0], h1[max_idx, 0], pred_h[max_idx, 0], t1_days)

    model.train()
    mean_c = total_c / max(count, 1)
    mean_h = total_h / max(count, 1)

    if last is not None:
        c_true, c_pred, h_true, h_pred = (t.detach().cpu().numpy() for t in last[:4])
        t1_days = last[4]
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        
        # 添加带有 Epoch 和 物理时间的全局大标题
        fig.suptitle(f"Elder FNO Validation - Epoch {epoch} | Physical Time: {t1_days:.1f} Days", 
                     fontsize=18, fontweight='bold')
                     
        titles = [
            ("True c", "Pred c", "|c error|"),
            ("True h", "Pred h", "|h error|"),
        ]
        fields = [
            (c_true, c_pred, np.abs(c_pred - c_true)),
            (h_true, h_pred, np.abs(h_pred - h_true)),
        ]
        for row in range(2):
            for col in range(3):
                f = fields[row][col]
                vmax = 1.0 if row == 0 and col < 2 else (None)
                im = ax[row, col].imshow(f, origin="upper", vmin=0.0, vmax=vmax)
                ax[row, col].set_title(titles[row][col])
                plt.colorbar(im, ax=ax[row, col])
        
        # 调整布局，为大标题留出空间
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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

    log = PythonLogger(name="elder_fno")
    log.file_logging()
    LaunchLogger.initialize()

    use_amp = bool(cfg.training.get("use_amp", False)) and device.type == "cuda"
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
    datapipe = ElderProblem2D(**dp_kwargs)
    val_datapipe = ElderProblem2D(**dp_kwargs)

    p_hydro = datapipe.p_hydro
    p_scale = datapipe.p_scale

    # --- model ---------------------------------------------------------------
    model = FNO(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        decoder_layers=cfg.model.decoder_layers,
        decoder_layer_size=cfg.model.decoder_layer_size,
        dimension=cfg.model.dimension,
        latent_channels=cfg.model.latent_channels,
        num_fno_layers=cfg.model.num_fno_layers,
        num_fno_modes=OmegaConf.to_container(cfg.model.num_fno_modes, resolve=True),
        padding=cfg.model.padding,
    ).to(device)

    if cfg.training.get("compile", False):
        forward_model = torch.compile(
            model, mode=cfg.training.get("compile_mode", "default")
        )
        log.info(f"torch.compile enabled (mode={cfg.training.compile_mode})")
    else:
        forward_model = model

    # --- physics residual evaluator ------------------------------------------
    use_physics = cfg.training.physics_weight > 0
    residual_mask = None
    if use_physics:
        residual_mask = _build_residual_mask(
            datapipe,
            mask_top_rows=cfg.training.get("mask_top_rows", 2),
            device=device,
        )

    optimizer = Adam(model.parameters(), lr=cfg.training.start_lr)
    scheduler = ExponentialLR(optimizer, gamma=cfg.training.gamma)

    ckpt_args = {
        "path": "./checkpoints",
        "models": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    start_epoch = load_checkpoint(device=device, **ckpt_args)
    out_dir = os.getcwd()  # hydra changes cwd to the run dir

    steps_per_epoch = cfg.data.steps_per_epoch
    val_iters = max(1, cfg.validation.sample_size // cfg.data.batch_size)
    physics_weight = cfg.training.physics_weight
    p_data_weight = float(cfg.training.get("p_data_weight", 1.0))
    continuity_weight = float(cfg.training.get("continuity_weight", 1.0))

    if start_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming from epoch {start_epoch + 1}.")

    for epoch in range(max(1, start_epoch + 1), cfg.training.max_epochs + 1):
        model.train()
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=steps_per_epoch, epoch_alert_freq=5
        ) as logger:
            for batch, _ in zip(datapipe, range(steps_per_epoch)):
                c0 = batch["c0"]
                p0 = batch["p0"]
                c1 = batch["c1"]
                p1 = batch["p1"]
                dt = batch["dt"]

                h0 = (p0 - p_hydro) / p_scale
                h1 = (p1 - p_hydro) / p_scale
                invar = torch.cat([c0, h0], dim=1)            # [B, 2, Ny+2, Nx+2]

                with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                    pred = forward_model(invar)                # [B, 2, Ny+2, Nx+2]
                    pred_c, pred_h = pred[:, 0:1], pred[:, 1:2]
                    loss_data = F.mse_loss(pred_c, c1) + p_data_weight * F.mse_loss(pred_h, h1)

                    loss_pde_c = torch.zeros((), device=device)
                    loss_pde_p = torch.zeros((), device=device)
                    if use_physics:
                        # PDE residual in fp32 (autocast disabled) for FD precision.
                        with torch.autocast(device_type=autocast_dev, enabled=False):
                            pred_c32 = pred_c.float()
                            pred_p32 = pred_h.float() * p_scale + p_hydro
                            loss_pde_c, loss_pde_p = _residuals_own_fd(
                                pred_c32, pred_p32, c0.float(), dt,
                                datapipe, residual_mask,
                            )
                        loss_pde = loss_pde_c + continuity_weight * loss_pde_p
                    else:
                        loss_pde = torch.zeros((), device=device)

                    loss = loss_data + physics_weight * loss_pde

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                logger.log_minibatch(
                    {
                        "loss_data": loss_data.detach(),
                        "loss_pde": (loss_pde_c + loss_pde_p).detach(),
                        "loss_pde_c": loss_pde_c.detach(),
                        "loss_pde_p": loss_pde_p.detach(),
                    }
                )
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if epoch % cfg.training.val_every == 0:
            with LaunchLogger("valid", epoch=epoch) as logger:
                val_loss = validation_step(
                    forward_model, val_datapipe, p_hydro, p_scale,
                    val_iters, epoch, out_dir, device, use_amp,
                )
                logger.log_epoch({"Validation error": val_loss})

        save_checkpoint(**ckpt_args, epoch=epoch)

    log.success("Training completed *yay*")

def generate_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Collect ``batch_size`` samples by advancing trajectories in rounds.

        Each round advances all ``n_trajectories`` in parallel (one batched
        flow solve) and yields that many samples; rounds repeat until the
        batch is full, then the concatenation is trimmed to ``batch_size``.
        """
        c0s, p0s, c1s, p1s, t0s = [], [], [], [], []
        n = self.batch_size
        have = 0
        while have < n:
            # 记录推进前的物理时间 (将秒转换为天)
            t0 = self._traj_step.clone() * self.dt_macro / (24 * 3600.0)
            c0, p0, c1, p1 = self._advance_all()
            c0s.append(c0)
            p0s.append(p0)
            c1s.append(c1)
            p1s.append(p1)
            t0s.append(t0)
            have += c0.shape[0]
        return (
            torch.cat(c0s, dim=0)[:n],
            torch.cat(p0s, dim=0)[:n],
            torch.cat(c1s, dim=0)[:n],
            torch.cat(p1s, dim=0)[:n],
            torch.cat(t0s, dim=0)[:n],
        )

def __iter__(self) -> Dict[str, Tensor]:
        """Yield batches of ``{c0, p0, c1, p1, t0, dt}`` infinitely.

        Tensor shapes are ``[batch, 1, Ny+2, Nx+2]`` (walls included); ``dt``
        is the macro time step (Python float) used to form the time derivative
        in the PDE residual.
        """
        while True:
            c0, p0, c1, p1, t0 = self.generate_batch()
            yield {
                "c0": c0,
                "p0": p0,
                "c1": c1,
                "p1": p1,
                "t0": t0,
                "dt": self.dt_macro,
            }


if __name__ == "__main__":
    main()
