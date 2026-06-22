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

"""Symbolic PDE definition for the 2-D convection-diffusion equation.

This defines only the *spatial* operator

    adv_diff = u * T_x + v * T_y - D * (T_xx + T_yy)

so it can be evaluated by :class:`physicsnemo.sym.eq.phy_informer.PhysicsInformer`,
which only computes spatial (x, y, z) derivatives.  The time derivative
``T_t = (T_{n+1} - T_n) / dt`` is added manually in the training script to
form the full residual ``T_t + adv_diff``.
"""

from __future__ import annotations

from sympy import Function, Number, Symbol

from physicsnemo.sym.eq.pde import PDE


class AdvectionDiffusion(PDE):
    """2-D convection-diffusion spatial operator.

    Parameters
    ----------
    T, u, v : str
        Names of the scalar field, x-velocity and y-velocity variables that
        will be supplied to ``PhysicsInformer.forward``.
    D : float
        (Constant) diffusivity.
    """

    def __init__(self, T: str = "T", u: str = "u", v: str = "v", D: float = 1e-3):
        self.dim = 2
        x, y = Symbol("x"), Symbol("y")
        T_var = Function(T)(x, y)
        u_var = Function(u)(x, y)
        v_var = Function(v)(x, y)
        D_var = Number(D) if isinstance(D, (int, float)) else D

        self.equations = {
            "adv_diff": (
                u_var * T_var.diff(x)
                + v_var * T_var.diff(y)
                - D_var * (T_var.diff(x, 2) + T_var.diff(y, 2))
            ),
        }
