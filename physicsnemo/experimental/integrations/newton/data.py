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

"""Zero-copy data exchange between Newton/Warp arrays and Torch tensors."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import warp as wp


def field_to_torch(
    value: Any,
    *,
    clone: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a Torch view of a Newton/Warp, NumPy, or Torch value.

    Warp arrays use ``wp.to_torch`` so GPU-resident Newton fields stay zero-copy
    when no dtype/device conversion or clone is requested. Pass ``clone=True``
    only when a stable snapshot is required (for example a gradient buffer that
    the Warp tape will overwrite).

    For Warp inputs the returned view inherits the source array's
    ``requires_grad`` (and ``grad``), so a view of a differentiable Newton field
    is itself autograd-tracked. Callers who need an inert data tensor should call
    ``.detach()`` on the result; ``clone=True`` alone does not strip
    ``requires_grad`` because ``clone`` stays in the autograd graph.
    """

    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, wp.array):
        tensor = wp.to_torch(value)
    else:
        tensor = torch.as_tensor(value)
    source = tensor
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    # A dtype/device conversion that actually ran returns a fresh, non-aliasing
    # tensor; cloning it again would be a redundant second allocation. ``.to()``
    # returns ``self`` when no conversion was needed, so only skip the clone when
    # the result no longer aliases the input.
    if clone and tensor is source:
        return tensor.clone()
    return tensor


def _assign_value(target: Any, value: Any) -> None:
    """Write ``value`` into a Newton/Warp, Torch, or NumPy ``target`` in place.

    Assignment mutates simulation state in place and is not part of Torch's
    autograd graph; Torch targets are therefore updated under ``no_grad`` so no
    gradients are recorded.
    """

    if isinstance(target, wp.array):
        if isinstance(value, torch.Tensor):
            # Write through the array's zero-copy Torch view so a GPU-resident
            # tensor stays on device. wp.array.assign() routes its source through
            # a host NumPy array, which cannot represent a CUDA tensor.
            with torch_warp_stream(target.device), torch.no_grad():
                field_to_torch(target).copy_(value.detach())
            return
        target.assign(value)
        return
    if isinstance(target, torch.Tensor):
        # ``target`` may be a Torch view onto a Warp buffer (for example a
        # BodyView sub-component slice of body_q/body_qd), so order this write on
        # the same stream as prior Warp work, exactly like the wp.array branch.
        with torch_warp_stream(target.device), torch.no_grad():
            target.copy_(
                field_to_torch(value, dtype=target.dtype, device=target.device)
            )
        return
    if isinstance(target, np.ndarray):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        target[...] = np.asarray(value)
        return
    raise TypeError(f"Cannot assign into target of type {type(target).__name__}.")


@contextmanager
def torch_warp_stream(device: Any) -> Iterator[None]:
    """Run Warp work on the current Torch stream for ``device``.

    Warp/Torch views share memory but a view alone does not order work submitted
    on different CUDA streams. Use this context around mixed-framework work;
    :class:`NewtonEnv` does so automatically for reset, collision, solver, and
    learned-step operations.
    """

    warp_device = wp.get_device(str(device))
    if not warp_device.is_cuda:
        yield
        return
    torch_device = torch.device(str(warp_device))
    stream = wp.stream_from_torch(torch.cuda.current_stream(torch_device))
    with wp.ScopedStream(stream):
        yield


def _copy_newton_object(target: Any, source: Any) -> None:
    """Copy a Newton state or control object without silently skipping fields."""

    assign = getattr(target, "assign", None)
    if callable(assign):
        assign(source)
        return
    for name, value in vars(source).items():
        if name.startswith("_") or value is None:
            continue
        if not hasattr(target, name):
            raise AttributeError(f"target has no field {name!r}")
        current = getattr(target, name)
        if current is None:
            raise ValueError(f"target field {name!r} is None while source is populated")
        if isinstance(value, (wp.array, torch.Tensor, np.ndarray)):
            _assign_value(current, value)
        elif hasattr(value, "__dict__"):
            _copy_newton_object(current, value)
        else:
            raise TypeError(
                f"cannot copy Newton field {name!r} of type {type(value).__name__}"
            )
