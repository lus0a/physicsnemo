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

"""Physics-informed FNO (PINO) training for the 2-D convection-diffusion equation.

The model learns the single-step operator ``T_n, u, v -> T_{n+1}`` over a macro
time step ``dt``. Training combines:

* a **data loss** (MSE between prediction and the reference solution), and
* a **PDE-residual loss** built from the full CD residual
  ``T_t + u*T_x + v*T_y - D*(T_xx + T_yy)`` evaluated on the prediction.

The spatial operator is computed by :class:`PhysicsInformer` with finite
differences; the time derivative ``T_t = (T_{n+1} - T_n)/dt`` is added manually
(PhysicsInformer only handles spatial derivatives). All data is generated
on the fly by :class:`ConvectionDiffusion2D`.

GPU readiness knobs (in ``config.yaml``):
* ``training.use_amp``      - mixed precision (autocast + GradScaler).
* ``training.tf32``         - enable TF32 matmul/conv on Ampere+.
* ``seed``                  - reproducibility (per-rank seeding under DDP).
"""

from __future__ import annotations

import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from cd_pde import AdvectionDiffusion
from datapipe import ConvectionDiffusion2D


def _enable_tf32() -> None:
    """Allow TF32 for matmul/conv on Ampere+ GPUs (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    # Hints the matmul precision on CUDA; harmless on CPU.
    torch.set_float32_matmul_precision("high")


def validation_step(model, datapipe, num_iters, epoch, out_dir, device, use_amp):
    """Evaluate MSE on fresh random samples and save a comparison plot.

    ``model`` here is the forward callable (possibly a ``torch.compile`` wrapper).
    """
    model.eval()
    total = 0.0
    count = 0
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for batch, _ in zip(datapipe, range(num_iters)):
            T0 = batch["T0"]
            T1 = batch["T1"]
            u = batch["u"]
            v = batch["v"]
            invar = torch.cat([T0, u, v], dim=1)
            with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                pred = model(invar)
            total += F.mse_loss(pred, T1).item()
            count += 1
            last = (T1[0, 0], pred[0, 0])

    model.train()
    mean_loss = total / max(count, 1)

    # save a comparison plot for the last seen sample
    true, pred = (t.detach().cpu().numpy() for t in last)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = true.min(), true.max()
    for a, field, title in zip(
        ax, (true, pred, np.abs(pred - true)), ("True T1", "Pred T1", "|Error|")
    ):
        im = a.imshow(field, vmin=vmin, vmax=vmax if title != "|Error|" else None)
        a.set_title(title)
        plt.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"val_{epoch:04d}.png"))
    plt.close(fig)
    return mean_loss


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Multi-GPU readiness: initialize the distributed manager once. On a
    # single process this is a no-op; on a multi-GPU node it sets up DDP.
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Reproducibility (per-rank under DDP so ranks generate different data).
    seed = int(cfg.get("seed", 0))
    if seed:
        torch.manual_seed(seed + dist.rank)
        np.random.seed(seed + dist.rank)

    if cfg.training.get("tf32", True):
        _enable_tf32()

    log = PythonLogger(name="cd_fno")
    log.file_logging()
    LaunchLogger.initialize()

    use_amp = bool(cfg.training.get("use_amp", False)) and device.type == "cuda"
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- data ----------------------------------------------------------------
    datapipe = ConvectionDiffusion2D(
        resolution=cfg.data.resolution,
        batch_size=cfg.data.batch_size,
        D=cfg.physics.D,
        dt_macro=cfg.physics.dt_macro,
        nr_ic_modes=cfg.data.nr_ic_modes,
        nr_vel_modes=cfg.data.nr_vel_modes,
        u_max=cfg.data.u_max,
        substeps=cfg.data.substeps,
        length=cfg.data.length,
        device=device,
    )
    # separate stream for validation
    val_datapipe = ConvectionDiffusion2D(
        resolution=cfg.data.resolution,
        batch_size=cfg.data.batch_size,
        D=cfg.physics.D,
        dt_macro=cfg.physics.dt_macro,
        nr_ic_modes=cfg.data.nr_ic_modes,
        nr_vel_modes=cfg.data.nr_vel_modes,
        u_max=cfg.data.u_max,
        substeps=cfg.data.substeps,
        length=cfg.data.length,
        device=device,
    )

    # --- model ---------------------------------------------------------------
    model = FNO(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        decoder_layers=cfg.model.decoder_layers,
        decoder_layer_size=cfg.model.decoder_layer_size,
        dimension=cfg.model.dimension,
        latent_channels=cfg.model.latent_channels,
        num_fno_layers=cfg.model.num_fno_layers,
        num_fno_modes=cfg.model.num_fno_modes,
        padding=cfg.model.padding,
    ).to(device)

    # Optional torch.compile of the FNO forward only. The original `model` is
    # retained for checkpointing (preserves the .mdlus format); `forward_model`
    # is the compiled callable used for every forward pass. PhysicsInformer runs
    # eagerly (its internal Graph is not compile-friendly). First iterations
    # pay a one-time compile cost, so this pays off on longer GPU runs.
    if cfg.training.get("compile", False):
        forward_model = torch.compile(
            model, mode=cfg.training.get("compile_mode", "default")
        )
        log.info(f"torch.compile enabled (mode={cfg.training.compile_mode})")
    else:
        forward_model = model

    # --- physics residual evaluator (spatial operator only) ------------------
    use_physics = cfg.training.physics_weight > 0
    if use_physics:
        pde = AdvectionDiffusion(D=cfg.physics.D)
        phy_informer = PhysicsInformer(
            required_outputs=["adv_diff"],
            equations=pde,
            grad_method="finite_difference",
            device=device,
            fd_dx=cfg.data.length / cfg.data.resolution,
        )
    else:
        phy_informer = None

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
                T0 = batch["T0"]
                T1 = batch["T1"]
                u = batch["u"]
                v = batch["v"]
                dt = batch["dt"]

                invar = torch.cat([T0, u, v], dim=1)  # [B, 3, N, N]

                # Model forward under optional AMP. The PDE residual is kept in
                # fp32 (autocast disabled) for finite-difference precision.
                with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                    pred = forward_model(invar)  # [B, 1, N, N]
                    loss_data = F.mse_loss(pred, T1)

                    if use_physics:
                        with torch.autocast(
                            device_type=autocast_dev, enabled=False
                        ):
                            pred32 = pred.float()
                            spatial = phy_informer.forward(
                                {"T": pred32, "u": u, "v": v}
                            )["adv_diff"]
                            T_t = (pred32 - T0) / dt
                            residual = T_t + spatial
                            # L1 on interior points only (skip FD-stencil edge).
                            loss_pde = residual[:, :, 2:-2, 2:-2].abs().mean()
                        loss = loss_data + physics_weight * loss_pde
                    else:
                        loss_pde = torch.zeros((), device=device)
                        loss = loss_data

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                logger.log_minibatch(
                    {
                        "loss_data": loss_data.detach(),
                        "loss_pde": loss_pde.detach(),
                    }
                )
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # validation
        if epoch % cfg.training.val_every == 0:
            with LaunchLogger("valid", epoch=epoch) as logger:
                val_loss = validation_step(
                    forward_model, val_datapipe, val_iters, epoch, out_dir, device, use_amp
                )
                logger.log_epoch({"Validation error": val_loss})

        save_checkpoint(**ckpt_args, epoch=epoch)

    log.success("Training completed *yay*")


if __name__ == "__main__":
    main()
