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

"""Symbolic PDE definition for the variable-density Elder problem (c, p).

The Elder problem (non-Boussinesq) couples a conservative concentration
transport equation to a variable-density Darcy flow equation (with fluid-mass
storage). The FNO learns the joint single-step operator
``(c_n, p_n) -> (c_{n+1}, p_{n+1})``. The two residuals used in the physics
loss are::

    transport :  d(phi rho c)/dt + div(rho q c) - div(rho phi Dm grad c)
    flow      :  d(phi rho)/dt + div(rho q),   with  q = -(k/mu)(grad p - rho g)

with ``rho = rho_f + drho*c`` and ``d(phi rho c)/dt = phi (rho_f + 2 drho c) c_t``.

This module defines the *spatial* part of these operators symbolically so they
could be evaluated by :class:`physicsnemo.sym.eq.phy_informer.PhysicsInformer`
(an optional residual backend). The default training path uses the ``own_fd``
backend in ``train_elder_fno.py`` instead, which computes the residuals with
hand-written non-periodic finite differences (correct all the way to the walls).

Note: ``PhysicsInformer``'s finite-difference gradient uses periodic
``torch.roll`` stencils, so its residual is only valid away from the Elder
domain walls. Prefer the ``own_fd`` backend for wall-accurate residuals. This
file is therefore kept as a symbolic reference and is not imported by the
default training script.
"""

from __future__ import annotations

from sympy import Function, Number, Symbol

from physicsnemo.sym.eq.pde import PDE


class ElderFlowTransport(PDE):
    """2-D variable-density Elder transport + flow spatial operators.

    Parameters
    ----------
    c, p : str
        Names of the concentration and pressure variables that will be supplied
        to ``PhysicsInformer.forward`` (velocity is derived from ``p`` and ``c``
        via Darcy's law, so it is not an independent input here).
    phi, Dm, permeability, viscosity, g, rho_f, drho : float
        Physical parameters (SI); see ``datapipe.ElderProblem2D``.
    """

    def __init__(
        self,
        c: str = "c",
        p: str = "p",
        phi: float = 0.1,
        Dm: float = 3.565e-6,
        permeability: float = 4.845e-13,
        viscosity: float = 1.0e-3,
        g: float = 9.81,
        rho_f: float = 1000.0,
        drho: float = 200.0,
    ):
        self.dim = 2                               # 2D 问题
        x, z = Symbol("x"), Symbol("z")            # 空间坐标 (z 向下为正)
        c_var = Function(c)(x, z)                  # 浓度场 c(x,z)
        p_var = Function(p)(x, z)                  # 压力场 p(x,z)

        phi_n = Number(phi)                        # 孔隙度 (常数)
        Dm_n = Number(Dm)                          # 分子扩散系数
        kom = Number(permeability / viscosity)     # k/mu
        g_n = Number(g)                            # 重力加速度
        rho_f_n = Number(rho_f)                    # 淡水密度
        drho_n = Number(drho)                      # 密度差

        # 变密度 rho(c) = rho_f + drho * c。
        rho = rho_f_n + drho_n * c_var
        # Darcy 速度 (z 向下, 重力 +z): q = -(k/mu)(grad p - rho g)。
        qx = -kom * p_var.diff(x)                  # x 分量: -(k/mu) dp/dx
        qz = -kom * (p_var.diff(z) - rho * g_n)    # z 分量: -(k/mu)(dp/dz - rho*g)

        # 守恒累积系数: d(phi rho c)/dt = phi (rho_f + 2 drho c) c_t
        # (时间导数 c_t 在训练脚本里加上以构成完整输运残差)。
        accum = phi_n * (rho_f_n + 2.0 * drho_n * c_var)

        self.equations = {
            # 输运残差的空间部分 (对流 - 扩散)。
            "transport_spatial": (
                (rho * qx * c_var).diff(x)         # 对流通量散度 div(rho q c) 的 x 部分
                + (rho * qz * c_var).diff(z)       # z 部分
                - ((rho * phi_n * Dm_n * c_var.diff(x)).diff(x)   # 扩散散度 div(rho phi Dm grad c) x 部分
                   + (rho * phi_n * Dm_n * c_var.diff(z)).diff(z))   # z 部分
            ),
            # 累积系数 (乘到 c_t 上)。
            "transport_accum": accum,
            # 流/连续性残差: d(phi rho)/dt + div(rho q), 其中
            # d(phi rho)/dt = phi*drho*c_t (rho = rho_f + drho*c, phi 常数)。
            "flow_storage": phi_n * drho_n,        # 时间项系数 (乘 c_t)
            "continuity_spatial": (rho * qx).diff(x) + (rho * qz).diff(z),   # div(rho q) 空间部分
        }
