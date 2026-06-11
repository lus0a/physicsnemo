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

"""Helpers for running the Newton integration under PhysicsNeMo's distributed stack.

The integration does not initialize a process group itself. Every model takes an
explicit ``device`` and most rollout/training loops remain plain per-process
loops. The NeRD high-level trainers are the exception: when the caller initializes
PhysicsNeMo under ``torchrun``, they use the active process group for sharded
teacher harvesting, global normalization statistics, and DDP.

The public helpers make the common multi-rank pattern ergonomic and correct.
Internal helpers centralize the PhysicsNeMo distributed policy used by the NeRD
trainers. They read ``physicsnemo.distributed.DistributedManager``
(PhysicsNeMo's standard distributed entry point) without requiring it: when no
manager is initialized they behave like a single process.

Typical use, after ``DistributedManager.initialize()``:

.. code-block:: python

    device = resolve_device(args.torch_device)
    scene = load_example_scene(Example, device=str(device))
    surrogate = BPTTSurrogate(..., device=device)
    if is_main_process():
        (out_dir / "report.md").write_text(report)
"""

from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.distributed import DistributedManager


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a requested device against PhysicsNeMo's distributed runtime.

    During a distributed run, the active rank-local device takes precedence so
    each process builds Newton and its model on the correct accelerator. An
    explicitly requested ``device`` is otherwise honored. When ``device`` is
    ``None`` and a manager is initialized (even as a single process with no
    process group), its reported device is used so the result matches
    ``DistributedManager.device`` rather than silently diverging from it;
    absent any manager this falls back to ``torch.get_default_device``. Pass
    the returned device to both Newton and the learned model.
    """
    if DistributedManager.is_initialized() and DistributedManager().distributed:
        resolved = DistributedManager().device
    elif device is not None:
        resolved = torch.device(device)
    elif DistributedManager.is_initialized():
        resolved = DistributedManager().device
    else:
        resolved = torch.get_default_device()

    # PyTorch accepts the shorthand ``cuda`` but tensors allocated there report
    # the concrete device ``cuda:<current>``. Canonicalize it once so Newton and
    # learned-step device validation cannot reject two equivalent devices.
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def is_main_process() -> bool:
    """``True`` on rank 0, or whenever no ``DistributedManager`` is initialized.

    Guard one-off side effects (writing a report or figure, printing a summary)
    with this so they happen once instead of once per rank."""
    return not DistributedManager.is_initialized() or DistributedManager().rank == 0


def _rank_world_size() -> tuple[int, int]:
    """Active PhysicsNeMo rank and world size, or serial defaults."""
    if not DistributedManager.is_initialized():
        return 0, 1
    manager = DistributedManager()
    return manager.rank, manager.world_size


def _all_reduce(
    *tensors: torch.Tensor,
    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
) -> None:
    """All-reduce tensors through PhysicsNeMo's active default process group."""
    if not DistributedManager.is_initialized():
        return
    manager = DistributedManager()
    if manager.world_size == 1:
        return
    group = manager.group()
    for tensor in tensors:
        torch.distributed.all_reduce(tensor, op=op, group=group)


def _distribute_model(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """Wrap ``model`` using PhysicsNeMo's active DDP policy when distributed."""
    if not DistributedManager.is_initialized():
        return model
    manager = DistributedManager()
    if not manager.distributed:
        return model
    device_ids = [manager.local_rank] if manager.device.index is not None else None
    output_device = manager.local_rank if manager.device.index is not None else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=manager.broadcast_buffers,
        find_unused_parameters=manager.find_unused_parameters,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
