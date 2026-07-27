# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared FNO/U-FNO load + single-step predict (used by fno_step.py and fno_server.py)."""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint

from ufno import build_model
from vtu_dataset import VtuElderDataset
from train_elder_fno import _resolve_fno_modes, _resolve_in_channels, build_invar

NORM_CACHE = "fno_norm.npz"

# Request/response magic for TCP protocol (C++ hybrid)
MAGIC_REQ = b"FNOQ"
MAGIC_OK = b"FNOA"
MAGIC_ERR = b"FNOE"


def _load_or_build_norm(cfg, device):
    if os.path.exists(NORM_CACHE):
        z = np.load(NORM_CACHE)
        p_hydro = torch.from_numpy(z["p_hydro"]).to(device)
        p_scale = float(z["p_scale"])
        modes = [int(m) for m in z["modes"]]
        return p_hydro, p_scale, modes, int(z["NY"]), int(z["NX"])

    phy = cfg.physics
    dp = VtuElderDataset(
        cfg.data.train_dir,
        1,
        device,
        phy.phi,
        phy.Dm,
        phy.permeability,
        phy.viscosity,
        phy.g,
        phy.rho_f,
        phy.drho,
        phy.W,
        phy.H,
        phy.dt_macro,
        file_stride=int(cfg.data.get("file_stride", 1)),
    )
    modes = _resolve_fno_modes(
        OmegaConf.to_container(cfg.model, resolve=True)["num_fno_modes"],
        dp,
        cfg.model.padding,
    )
    np.savez(
        NORM_CACHE,
        p_hydro=dp.p_hydro.cpu().numpy(),
        p_scale=np.float32(dp.p_scale),
        modes=np.asarray(modes, dtype=np.int64),
        NY=np.int64(dp.Ny_tot),
        NX=np.int64(dp.Nx_tot),
    )
    print(
        f"fno_infer: built {NORM_CACHE} (p_scale={float(dp.p_scale):.1f}, "
        f"modes={modes}, {dp.Ny_tot}x{dp.Nx_tot})",
        flush=True,
    )
    return dp.p_hydro, float(dp.p_scale), modes, int(dp.Ny_tot), int(dp.Nx_tot)


def _load_model_weights(model, checkpoint: str, device) -> str:
    import glob as _glob

    _mdlus = sorted(_glob.glob(os.path.join(checkpoint, "*.mdlus")))
    if _mdlus:

        def _epoch_of(_f):
            try:
                return int(os.path.basename(_f).rsplit(".", 2)[-2])
            except ValueError:
                return 0

        _mdlus.sort(key=_epoch_of)
        path = _mdlus[-1]
        model.load(path)
        return path
    load_checkpoint(path=checkpoint, models=model, device=device)
    return checkpoint


@dataclass
class FnoEngine:
    """Loaded model + norm; one process, many predict() calls."""

    model: torch.nn.Module
    device: torch.device
    p_hydro: torch.Tensor
    p_scale: float
    NY: int
    NX: int
    residual: bool
    dt_aware: bool
    dt_ref: float
    arch: str
    ckpt_path: str

    @property
    def n_cells(self) -> int:
        return self.NY * self.NX

    @classmethod
    def load(
        cls,
        config: str = "config.yaml",
        checkpoint: str = "outputs_elder_ufno/checkpoints",
        device: Optional[str] = None,
    ) -> "FnoEngine":
        warnings.filterwarnings(
            "ignore",
            message="Could not initialize using ENV, SLURM or OPENMPI methods",
        )
        cfg = OmegaConf.load(config)
        if device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        DistributedManager.initialize()
        p_hydro, p_scale, modes, NY, NX = _load_or_build_norm(cfg, dev)
        residual = bool(cfg.training.get("residual", False))
        mdl = cfg.model
        model = build_model(
            mdl, num_fno_modes=modes, in_channels=_resolve_in_channels(mdl)
        ).to(dev)
        ckpt_path = _load_model_weights(model, checkpoint, dev)
        model.eval()
        arch = str(mdl.get("arch", "fno"))
        print(
            f"fno_infer: ready arch={arch} residual={residual} "
            f"grid={NY}x{NX} device={dev} ckpt={ckpt_path}",
            flush=True,
        )
        return cls(
            model=model,
            device=dev,
            p_hydro=p_hydro,
            p_scale=p_scale,
            NY=NY,
            NX=NX,
            residual=residual,
            dt_aware=bool(cfg.model.get("dt_channel", False)),
            dt_ref=float(cfg.model.get("dt_ref_s", 2.592e6)),
            arch=arch,
            ckpt_path=ckpt_path,
        )

    def predict(
        self, c_flat: np.ndarray, P_flat: np.ndarray, dt_sec: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """c,P: float32 length NY*NX row-major; returns c_pred, P_pred same layout."""
        n = self.n_cells
        if c_flat.size != n or P_flat.size != n:
            raise ValueError(
                f"bad field size c={c_flat.size} P={P_flat.size} expect {n}"
            )
        if self.dt_aware and dt_sec is None:
            raise ValueError("dt_channel=true requires dt_sec")
        dt_val = float(dt_sec if dt_sec is not None else 0.0)
        if self.dt_aware and 0.0 < dt_val < 1.0:
            print(
                f"fno_infer: WARNING dt={dt_val} looks like rec_dt=1/Δt, not Δt seconds",
                flush=True,
            )

        c_n = torch.from_numpy(np.asarray(c_flat, dtype=np.float32).reshape(self.NY, self.NX)[None, None]).to(
            self.device
        )
        P_n = torch.from_numpy(np.asarray(P_flat, dtype=np.float32).reshape(self.NY, self.NX)[None, None]).to(
            self.device
        )
        h_n = (P_n - self.p_hydro) / self.p_scale
        invar = build_invar(c_n, h_n, dt_val, self.dt_aware, self.dt_ref)
        with torch.no_grad():
            raw_out = self.model(invar)
        if self.residual:
            c_pred = c_n + raw_out[:, 0:1]
            h_pred = h_n + raw_out[:, 1:2]
        else:
            c_pred = raw_out[:, 0:1]
            h_pred = raw_out[:, 1:2]
        P_pred = h_pred * self.p_scale + self.p_hydro
        c_arr = c_pred[0, 0].detach().cpu().numpy().astype(np.float32).ravel()
        P_arr = P_pred[0, 0].detach().cpu().numpy().astype(np.float32).ravel()
        return c_arr, P_arr

    def predict_files(self, inpath: str, outpath: str, dt_sec: float) -> None:
        raw = np.fromfile(inpath, dtype=np.float32)
        n = self.n_cells
        if raw.size < 2 * n:
            raise ValueError(f"input too short ({raw.size} < {2 * n})")
        c_arr, P_arr = self.predict(raw[:n], raw[n : 2 * n], dt_sec)
        with open(outpath, "wb") as f:
            f.write(c_arr.tobytes())
            f.write(P_arr.tobytes())
        d_tilde = (dt_sec / self.dt_ref) if (self.dt_aware and self.dt_ref > 0) else float("nan")
        print(
            f"fno_infer: dt={dt_sec:.6g}s={dt_sec/86400:.4g}d d_tilde={d_tilde:.4g} "
            f"c[{c_arr.min():.3f},{c_arr.max():.3f}] P[{P_arr.min():.3e},{P_arr.max():.3e}]",
            flush=True,
        )
