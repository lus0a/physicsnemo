# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""VTS-based Elder dataset — 用真实求解器导出的 .vts 快照训练。

与 :class:`datapipe.ElderProblem2D` **同接口** (暴露相同的物理/网格属性, yield
相同的 batch dict ``{c0, p0, c1, p1, t0, dt}``)，因此训练/验证/PDE 残差代码无需改动,
只需在 ``train_elder_fno.py`` 里二选一。

约定 (与用户确认):
* 目录 ``train_dir`` 下按文件名排序的 ``.vts`` = 单条轨迹的连续快照;
* 相邻两个文件 = 一个单步对 ``(c_n,p_n) -> (c_{n+1},p_{n+1})``, 时间差 = ``dt_macro``;
* VTS PointData 数组名 ``c`` / ``p``; ``p`` 为真实压力 (Pa); 网格含壁面节点
  (shape ``(Ny+2)x(Nx+2)``)。
"""

from __future__ import annotations

import glob
import os
from typing import Dict

import numpy as np
import pyvista as pv
import torch

Tensor = torch.Tensor


class VtsElderDataset:
    """Load sorted ``.vts`` snapshots and yield consecutive-step pairs.

    Parameters
    ----------
    train_dir : str
        含按时间排序的 ``.vts`` 快照的目录 (>=2 个文件)。
    batch_size : int
        每个 batch 的单步对数量。
    device : torch.device
        数据放置的设备。
    phi, Dm, permeability, viscosity, g, rho_f, drho, W, H : float
        物理参数 (SI), 与 ``ElderProblem2D`` 一致, 供 PDE 残差使用。
    dt_macro : float
        相邻快照的时间间隔 [s] (= 网络学到的单步时长)。
    flow_sign : float
        浮力方向符号 (+1 dense 下沉)。
    """

    def __init__(
        self,
        train_dir: str,
        batch_size: int,
        device,
        phi: float,
        Dm: float,
        permeability: float,
        viscosity: float,
        g: float,
        rho_f: float,
        drho: float,
        W: float,
        H: float,
        dt_macro: float,
        flow_sign: float = 1.0,
    ):
        self.device = torch.device(device)
        # --- 物理参数 (暴露给 _residuals_own_fd 等) ---
        self.phi = float(phi)
        self.Dm = float(Dm)
        self.permeability = float(permeability)
        self.viscosity = float(viscosity)
        self.g = float(g)
        self.rho_f = float(rho_f)
        self.drho = float(drho)
        self.W = float(W)
        self.H = float(H)
        self.dt_macro = float(dt_macro)
        self.k_over_mu = self.permeability / self.viscosity
        self.gz = float(flow_sign) * self.g
        self.batch_size = int(batch_size)

        files = sorted(glob.glob(os.path.join(train_dir, "*.vts")))
        if len(files) < 2:
            raise ValueError(
                f"VtsElderDataset 需要至少 2 个 .vts 文件 (才能配成单步对), "
                f"在 {train_dir!r} 只找到 {len(files)} 个"
            )
        self.n_files = len(files)

        # --- 用第一个文件确定网格尺寸与方向 (row 0 = 顶部, 与 datapipe 一致) ---
        g0 = pv.read(files[0])
        nx_tot, ny_tot, _ = g0.dimensions            # VTK dimensions 序: (x, y, z)
        self.Nx_tot, self.Ny_tot = int(nx_tot), int(ny_tot)
        self.Nx, self.Ny = self.Nx_tot - 2, self.Ny_tot - 2   # 内部 cell 数
        pts0 = np.asarray(g0.points, dtype=np.float64).reshape(self.Ny_tot, self.Nx_tot, 3)
        x_cols = pts0[0, :, 0]                        # 每列的 x 坐标 (行 0)
        z_rows = pts0[:, 0, 2]                        # 每行的 z 坐标 (列 0)
        # 方向: 含 c≈1 源段的那一行 = 顶部; 若不是 row 0 则翻转。
        c0 = np.asarray(g0.point_data["c"], dtype=np.float64).reshape(self.Ny_tot, self.Nx_tot)
        flip = self._source_row(c0) != 0
        if flip:                                      # 翻转行序使顶部 -> row 0
            z_rows = z_rows[::-1]

        # 网格间距 (从坐标算, 回退到 W/H)
        self.dx = float(np.mean(np.abs(np.diff(x_cols)))) if self.Nx_tot > 1 else self.W / max(1, self.Nx)
        depth = np.abs(z_rows - z_rows[0])           # 每行相对顶部的深度 (顶部=0, 向下增)
        self.dy = float(np.mean(np.abs(np.diff(depth)))) if self.Ny_tot > 1 else self.H / max(1, self.Ny)

        # --- 静水压参考 p_hydro = rho_f * g * depth (每行, 广播到全网格) ---
        p_hydro = (self.rho_f * self.g * depth).astype(np.float32)   # (Ny_tot,)
        p_hydro = np.broadcast_to(p_hydro[:, None], (self.Ny_tot, self.Nx_tot)).copy()

        # --- 载入全部快照 [N, 2, Ny_tot, Nx_tot] (通道 0=c, 1=p), 并按方向对齐 ---
        data = np.empty((self.n_files, 2, self.Ny_tot, self.Nx_tot), dtype=np.float32)
        h_amp = 0.0                                   # 跟踪 max|h| 用于 p_scale
        for k, f in enumerate(files):
            gk = pv.read(f)
            c = np.asarray(gk.point_data["c"], dtype=np.float32).reshape(self.Ny_tot, self.Nx_tot)
            p = np.asarray(gk.point_data["p"], dtype=np.float32).reshape(self.Ny_tot, self.Nx_tot)
            if flip:
                c = c[::-1, :].copy()
                p = p[::-1, :].copy()
            data[k, 0] = c
            data[k, 1] = p
            h_amp = max(h_amp, float(np.max(np.abs(p - p_hydro))))
        self.data = torch.from_numpy(data).to(self.device)
        self.p_hydro = torch.from_numpy(p_hydro).to(self.device)
        floor = 0.01 * (self.drho * self.g * self.H)  # 与 datapipe 一致的下限
        self.p_scale = float(max(h_amp, floor))

        self.n_pairs = self.n_files - 1               # 单步对数
        self._dt_days = self.dt_macro / (24.0 * 3600.0)

    @staticmethod
    def _source_row(c: np.ndarray) -> int:
        """含 c≈1 源段的行号 (= 顶部行)。用 IC (首帧) 的源段判定方向。"""
        rowsum = (c > 0.5).sum(axis=1)                # 每行 c≈1 的 cell 数
        return int(np.argmax(rowsum))

    def __iter__(self):
        """无限 yield batch ``{c0, p0, c1, p1, t0, dt}``; 每轮打乱配对顺序。"""
        n = self.batch_size
        while True:
            order = np.random.permutation(self.n_pairs)
            for i in range(0, self.n_pairs, n):
                idxs = order[i:i + n]                 # 本 batch 的配对索引 (pair j = 快照 j -> j+1)
                j = torch.as_tensor(idxs, device=self.device, dtype=torch.long)
                yield {
                    "c0": self.data[j, 0:1],          # [B,1,Ny_tot,Nx_tot]
                    "p0": self.data[j, 1:2],
                    "c1": self.data[j + 1, 0:1],
                    "p1": self.data[j + 1, 1:2],
                    "t0": j.float() * self._dt_days,  # 该对的物理起始时间 (天) [B]
                    "dt": self.dt_macro,
                }
