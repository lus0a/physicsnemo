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
        self.dim = 2
        x, z = Symbol("x"), Symbol("z")
        c_var = Function(c)(x, z)
        p_var = Function(p)(x, z)

        phi_n = Number(phi)
        Dm_n = Number(Dm)
        kom = Number(permeability / viscosity)
        g_n = Number(g)
        rho_f_n = Number(rho_f)
        drho_n = Number(drho)

        # Variable density rho(c) = rho_f + drho * c.
        rho = rho_f_n + drho_n * c_var
        # Darcy velocity (z downward, gravity +z): q = -(k/mu)(grad p - rho g).
        qx = -kom * p_var.diff(x)
        qz = -kom * (p_var.diff(z) - rho * g_n)

        # Conservative accumulation coefficient: d(phi rho c)/dt
        # = phi (rho_f + 2 drho c) c_t  (the time derivative c_t is added in
        # the training script to form the full transport residual).
        accum = phi_n * (rho_f_n + 2.0 * drho_n * c_var)

        self.equations = {
            # Spatial part of the transport residual (advection - diffusion).
            "transport_spatial": (
                (rho * qx * c_var).diff(x)
                + (rho * qz * c_var).diff(z)
                - ((rho * phi_n * Dm_n * c_var.diff(x)).diff(x)
                   + (rho * phi_n * Dm_n * c_var.diff(z)).diff(z))
            ),
            # Accumulation coefficient multiplying c_t.
            "transport_accum": accum,
            # Flow / continuity residual: d(phi rho)/dt + div(rho q), with
            # d(phi rho)/dt = phi*drho*c_t (rho = rho_f + drho*c, phi const).
            "flow_storage": phi_n * drho_n,
            "continuity_spatial": (rho * qx).diff(x) + (rho * qz).diff(z),
        }
