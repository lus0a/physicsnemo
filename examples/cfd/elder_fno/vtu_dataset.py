# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VTU-based Elder dataset — 用 unisolver 高精度 .vtu 快照训练。

与 :class:`datapipe.ElderProblem2D` / :class:`vts_dataset.VtsElderDataset` **同接口**
(暴露相同的物理/网格属性, yield 相同的 batch dict ``{c0, p0, c1, p1, t0, dt}``),
因此训练/验证/PDE 残差代码无需改动, 只需在 ``train_elder_fno.py`` 里按数据源三选一。

与 ``vts_dataset`` 的区别仅在"怎么读单个文件":
* ``.vtu`` 是 **cell-centered 非结构网格** (unisolver 并行分块输出), 变量在 CellData;
* 需按 cell 中心坐标排序才能还原成规则 ``[Ny_tot, Nx_tot] = [64, 256]`` 图;
* 网格是均匀 256×64, 间距 2.34375 m, 域 600×150 m, 与 config 物理参数一致。

约定 (与 datapipe 的"全网格 = 内部 + 1 圈边界"对齐):
* 256×64 的**最外圈 cell 即边界** (顶行承载 c≈1 源段、底行 c=0、侧壁无流), 故
  ``Nx_tot=256, Ny_tot=64`` 直接喂给 FNO, 不再额外补壁环;
* 内部 (PDE 残差评估区) = 剥掉最外圈 → ``Nx=254, Ny=62``, 与现有
  ``_residuals_own_fd`` 的 ``field[..., 1:-1, 1:-1]`` 切片约定天然一致。
* 相邻两个文件 = 一个单步对 ``(c_n,p_n) -> (c_{n+1},p_{n+1})``, 时间差 = ``dt_macro``;
* CellData 名 ``c`` / ``P``; ``P`` 为真实压力 (Pa); ``h`` 由 ``P - p_hydro`` 重算
  (已验证等于 VTU 自带 ``h`` 字段, 但重算避免依赖其约定)。
"""

from __future__ import annotations

import glob
import os
from typing import Dict

import numpy as np
import pyvista as pv
import torch

Tensor = torch.Tensor


class VtuElderDataset:
    """Load sorted ``.vtu`` cell-centered snapshots and yield consecutive-step pairs.

    Parameters
    ----------
    train_dir : str
        含按时间数值排序的 ``.vtu`` 快照的目录 (>=2 个文件)。
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
    split : str
        "all" = 用全部配对 (默认, eval_rollout 用); "train"/"val" = 中间窗口留出验证
        (val 占中间 ``val_frac`` 比例快照, 丢弃跨边界泄漏配对, 见 :meth:`_select_pairs`)。
    val_frac : float
        val 快照占快照总数的比例, 拆成多个小块散布, 仅 ``split != "all"`` 时生效。
    n_val_blocks : int
        val 拆成多少个等宽小块, 均匀散布在时间轴上 (覆盖多个时段, 避免单段集中)。
    val_gap : int
        每个 val 块两侧各丢弃多少快照作缓冲 (降低边界“相邻快照相似”的残余泄漏)。
    file_stride : int
        配对跨步数 (原始文件间隔 = 10 天): pair k = 快照 k -> 快照 k+file_stride。=1 => 10 天/步,
        =3 => 30 天/步。保留全部快照 (不抽稀), 故任意步长都不严重缩水。dt_macro 须 = file_stride × 864000。
    _share : VtuElderDataset | None
        非空时复用其已加载的 data/p_hydro/网格 (避免二次读盘), 仅本实例配对选择不同。
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
        split: str = "all",
        val_frac: float = 0.2,
        n_val_blocks: int = 8,
        val_gap: int = 2,
        file_stride: int = 1,
        _share: "VtuElderDataset | None" = None,
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
        self.split = str(split)
        self.val_frac = float(val_frac)
        self.n_val_blocks = int(n_val_blocks)
        self.val_gap = int(val_gap)
        self.file_stride = max(1, int(file_stride))

        if _share is not None:
            # 复用已加载实例的数据/网格 (避免二次读盘); 仅本实例配对选择不同。
            for _a in ("data", "p_hydro", "p_scale", "dx", "dy",
                       "Nx_tot", "Ny_tot", "Nx", "Ny", "n_files",
                       "file_interval", "_times_days", "dt_macro"):
                setattr(self, _a, getattr(_share, _a))
            self.pair_indices = self._select_pairs()
            self.n_pairs = int(len(self.pair_indices))
            return

        # --- 文件按时间数值排序 (绝不能字典序), 全部保留; file_stride 控制配对跨步 (不抽稀, 不丢数据) ---
        files = sorted(
            glob.glob(os.path.join(train_dir, "*.vtu")),
            key=_extract_time,
        )                                            # 全部文件; 配对用 (k -> k+stride)
        if len(files) < self.file_stride + 1:
            raise ValueError(
                f"VtuElderDataset 需要至少 file_stride+1 个 .vtu 文件, "
                f"在 {train_dir!r} 只找到 {len(files)} 个"
            )
        self.n_files = len(files)
        # 文件原始间隔 (秒) 与各快照物理时间 (天); 用于 t0 标注 + 一致性检查。
        times_s = np.asarray([_extract_time(f) for f in files], dtype=np.float64)
        self.file_interval = float(np.median(np.diff(times_s)))   # 原始文件间隔 (~864000=10天)
        self._times_days = torch.from_numpy((times_s / 86400.0).astype(np.float32)).to(self.device)
        # dt_macro 自动从 file_stride × 原始间隔推导 (file_stride 是唯一步长控制量, 改它即可;
        # config 里的 dt_macro 在 VTU 路径被忽略, 仅在线 datapipe/VTS 路径生效)。
        self.dt_macro = self.file_stride * self.file_interval

        # --- 用第一个文件确定 cell 中心排序索引 (mesh 全程相同, 复用) ---
        g0 = pv.read(files[0])
        cc0 = np.asarray(g0.cell_centers().points)             # [Ncell, 3] cell 中心
        nx_uniq = np.unique(np.round(cc0[:, 0], 4)).size
        nz_uniq = np.unique(np.round(cc0[:, 2], 4)).size
        # 反推均匀间距 dx (cell 中心间距); W/H 作回退。
        dx_guess = float(np.median(np.diff(np.sort(np.unique(cc0[:, 0]))))) if nx_uniq > 1 else self.W / 4.0
        self.dx = dx_guess
        self.dy = dx_guess                                     # 方形网格 dx=dy
        self.Nx_tot = nx_uniq                                  # 列数 (含边界 cell)
        self.Ny_tot = nz_uniq                                  # 行数 (含边界 cell)
        self.Nx = self.Nx_tot - 2                              # 内部列 (剥掉左右壁 cell)
        self.Ny = self.Ny_tot - 2                              # 内部行 (剥掉上下壁 cell)
        ncell_expect = self.Nx_tot * self.Ny_tot
        if g0.n_cells != ncell_expect:
            raise ValueError(
                f"cell 数 {g0.n_cells} != {self.Nx_tot}x{self.Ny_tot}={ncell_expect}, "
                f"网格非均匀 {self.Nx_tot}x{self.Ny_tot}, 需检查数据"
            )
        # cell 中心 -> (行, 列) 整数索引: 中心在 (i+0.5)*dx, 故 /dx - 0.5。
        ix = np.round(cc0[:, 0] / self.dx - 0.5).astype(int)   # 列号 0..Nx_tot-1
        iz = np.round(cc0[:, 2] / self.dy - 0.5).astype(int)   # 行号 0..Ny_tot-1
        if len({(r, c) for r, c in zip(iz.tolist(), ix.tolist())}) != g0.n_cells:
            raise ValueError("cell 中心坐标无法唯一映射到网格, 间距假设不对")
        self._order = np.lexsort((ix, iz))                     # sorted-by-(row,col) 的 cell 下标

        # --- 每行的 cell 中心深度 z (取排序后每行首列的 z), 算静水压 ---
        cc_sorted = cc0[self._order].reshape(self.Ny_tot, self.Nx_tot, 3)
        z_rows = cc_sorted[:, 0, 2]                            # 每行深度 z [m] (向下增)
        p_hydro = (self.rho_f * self.g * z_rows).astype(np.float32)      # (Ny_tot,)
        p_hydro = np.broadcast_to(p_hydro[:, None], (self.Ny_tot, self.Nx_tot)).copy()

        # --- 载入全部快照 [N, 2, Ny_tot, Nx_tot] (通道 0=c, 1=P), 按 _order 还原成规则网格 ---
        data = np.empty((self.n_files, 2, self.Ny_tot, self.Nx_tot), dtype=np.float32)
        h_amp = 0.0                                            # 跟踪 max|h| 用于 p_scale
        for k, f in enumerate(files):
            gk = pv.read(f)
            c = gk.cell_data["c"][self._order].reshape(self.Ny_tot, self.Nx_tot)
            P = gk.cell_data["P"][self._order].reshape(self.Ny_tot, self.Nx_tot)
            data[k, 0] = c
            data[k, 1] = P
            h_amp = max(h_amp, float(np.max(np.abs(P - p_hydro))))
        self.data = torch.from_numpy(data).to(self.device)
        self.p_hydro = torch.from_numpy(p_hydro).to(self.device)
        floor = 0.01 * (self.drho * self.g * self.H)          # 与 datapipe 一致的下限
        self.p_scale = float(max(h_amp, floor))

        self.pair_indices = self._select_pairs()               # 按 split 选配对 (丢弃边界泄漏配对)
        self.n_pairs = int(len(self.pair_indices))             # 单步对数

    def _select_pairs(self) -> np.ndarray:
        """按 ``self.split`` 选配对下标 (pair k = 快照 k -> k+file_stride)。

        ``split="all"`` 返回全部配对; ``"train"``/``"val"`` 把 ``val_frac`` 比例的快照拆成
        ``n_val_blocks`` 个等宽小块, **均匀散布**在时间轴上 (覆盖早/中/晚多个时段, 避免单段集中),
        块中心等距分布于内部 (首末快照始终留 train)。每个 val 块两侧各 ``val_gap`` 个快照作缓冲
        被丢弃 (降低边界"相邻快照相似"的残余泄漏)。配对按"两端快照同属一个集合"归入 train/val,
        跨集合或落在 gap 的配对一律丢弃, 保证每个 snapshot 只属于一个集合。
        """
        n = self.n_files
        s = self.file_stride
        all_pairs = np.arange(n - s, dtype=np.int64)           # pair k = (s_k -> s_{k+s})
        if self.split == "all" or self.val_frac <= 0.0:
            return all_pairs
        nb = max(1, int(self.n_val_blocks))                    # val 块数
        total = max(2 * nb, int(round(self.val_frac * n)))     # val 快照总数 (每块至少 2)
        w = max(2, total // nb)                                # 每块宽度
        val_set, gap_set = set(), set()
        for i in range(nb):
            # 块中心等距分布于内部 (i+1)/(nb+1) 处, 避开首末。
            c = int(round((i + 1) * n / (nb + 1)))
            lo = c - w // 2
            hi = lo + w - 1
            lo = max(0, lo); hi = min(n - 1, hi)
            for s_ in range(lo, hi + 1):
                val_set.add(s_)
            # 块两侧各 val_gap 个快照作缓冲 (既不 train 也不 val)。
            for g in range(1, int(self.val_gap) + 1):
                for s_ in (lo - g, hi + g):
                    if 0 <= s_ < n:
                        gap_set.add(s_)
        if self.split == "val":
            sel = [k for k in all_pairs if k in val_set and (k + s) in val_set]
        else:                                                   # "train"
            sel = [k for k in all_pairs
                   if (k not in val_set and (k + s) not in val_set
                       and k not in gap_set and (k + s) not in gap_set)]
        return np.asarray(sel, dtype=np.int64)

    def __iter__(self) -> Dict[str, Tensor]:
        """无限 yield batch ``{c0, p0, c1, p1, t0, dt}``; 每轮打乱配对顺序。"""
        n = self.batch_size
        pairs = self.pair_indices
        s = self.file_stride
        while True:
            order = np.random.permutation(len(pairs))
            for i in range(0, len(pairs), n):
                j = torch.as_tensor(pairs[order[i:i + n]], device=self.device, dtype=torch.long)
                yield {
                    "c0": self.data[j, 0:1],                   # [B,1,Ny_tot,Nx_tot] pair j = 快照 j
                    "p0": self.data[j, 1:2],
                    "c1": self.data[j + s, 0:1],               # 快照 j+s (目标; s=1 退化为相邻)
                    "p1": self.data[j + s, 1:2],
                    "t0": self._times_days[j],                 # 该对起始快照的物理时间 (天) [B]
                    "dt": self.dt_macro,
                }


def _extract_time(path: str) -> float:
    """从 ``progame_name-<秒>.vtu`` 抽出时间数值, 供数值排序。"""
    name = os.path.basename(path)
    stem = name.rsplit(".", 1)[0]                              # 去 .vtu
    if "-" in stem:
        stem = stem.rsplit("-", 1)[-1]                        # 取最后一段数字
    try:
        return float(stem)
    except ValueError:
        return 0.0
