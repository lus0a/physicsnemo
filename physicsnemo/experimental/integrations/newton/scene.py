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

"""Headless construction of Newton example scenes for PhysicsNeMo workflows."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from typing import Any

import torch
import warp as wp


def load_example_scene(
    example_cls: type,
    *,
    device: str | torch.device | None = None,
    substeps: int | None = None,
    verbose: bool | None = None,
    args: argparse.Namespace | None = None,
    arg_overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Construct a ``newton.examples`` ``Example`` headless, with no graph capture.

    Newton example classes take ``(viewer, args)`` and capture their own
    forward/backward CUDA graph in ``__init__``. We pass the shipped no-op
    ``newton.viewer.ViewerNull`` and skip that capture (we drive our own
    rollouts), so the scene becomes a reusable source of ``model``/``solver``.

    Parameters
    ----------
    example_cls : type
        Newton ``Example`` class, such as one from
        ``newton.examples.diffsim.example_diffsim_ball``.
    device : str or torch.device, optional
        Device on which to build the model. An active distributed run uses its
        rank-local device when omitted.
    substeps : int, optional
        Override for the scene's ``sim_substeps`` and corresponding ``sim_dt``.
    verbose : bool, optional
        Override for the example's ``args.verbose``. When omitted, a
        caller-supplied ``args.verbose`` is preserved; otherwise the example
        runs quietly.
    args : argparse.Namespace, optional
        Pre-built example argument namespace.
    arg_overrides : Mapping[str, Any], optional
        Values to set on the argument namespace before construction.

    Returns
    -------
    Any
        Constructed headless Newton example scene.
    """

    from physicsnemo.experimental.integrations.newton.dependencies import require_newton
    from physicsnemo.experimental.integrations.newton.distributed import resolve_device

    if substeps is not None and substeps <= 0:
        # Validate before the (potentially expensive) scene construction.
        raise ValueError("substeps must be positive")
    ViewerNull = require_newton().viewer.ViewerNull
    device = resolve_device(device)

    class HeadlessExample(example_cls):
        """Example subclass that disables scene-owned graph capture."""

        def capture(self) -> None:
            """Leave graph execution to the PhysicsNeMo rollout driver."""
            self.graph = None

    if args is None:
        create_parser = getattr(example_cls, "create_parser", None)
        args = (
            create_parser().parse_args([])
            if callable(create_parser)
            else argparse.Namespace()
        )
    else:
        # Deep-copy so the caller's namespace (including any nested mutable
        # values) is fully isolated from our mutations below and from in-place
        # changes a Newton Example might make during construction.
        args = copy.deepcopy(args)
    if verbose is not None:
        args.verbose = verbose
    elif not hasattr(args, "verbose"):
        args.verbose = False
    for name, value in dict(arg_overrides or {}).items():
        setattr(args, name, value)

    with wp.ScopedDevice(str(device)):
        scene = HeadlessExample(ViewerNull(num_frames=0), args)
    scene.graph = None

    if substeps is not None:
        frame_dt = getattr(scene, "frame_dt", None)
        if frame_dt is None:
            sim_dt = getattr(scene, "sim_dt", None)
            sim_substeps = getattr(scene, "sim_substeps", None)
            if sim_dt is None or sim_substeps is None:
                raise AttributeError(
                    "cannot override substeps: scene exposes neither frame_dt nor "
                    "both sim_dt and sim_substeps"
                )
            frame_dt = float(sim_dt) * int(sim_substeps)
        scene.sim_substeps = int(substeps)
        scene.sim_dt = float(frame_dt) / int(substeps)
    return scene
