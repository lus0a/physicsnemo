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

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Callable, Sequence, cast
from warnings import warn

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.tensor import DTensor, distribute_module
from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
)
from torch.distributed.tensor.placement_types import (
    Placement,
    Replicate,
    Shard,
)

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel._shard_redistribute import (
    ShardRedistribute,
)
from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    _infer_shard_tensor_spec_from_local_chunks,
    _stride_from_contiguous_shape_C_style,
)

aten = torch.ops.aten


# ======================================================================

# ============================================================================
# Layer 1 -- Semi-private conversions (no autograd, no spec inference)
# ============================================================================


def _shard_tensor_to_dtensor(st: "ShardTensor") -> DTensor:
    r"""Convert a ShardTensor to a plain DTensor (no autograd).

    Creates a DTensor sharing the same ``_local_tensor`` and ``_spec``.
    Use for dispatch or inside backward when building a DTensor gradient.
    """
    dtensor = torch.Tensor._make_wrapper_subclass(
        DTensor,
        st._spec.tensor_meta.shape,
        strides=st._spec.tensor_meta.stride,
        dtype=st.dtype,
        device=st.device,
        layout=st.layout,
        requires_grad=st.requires_grad,
    )
    dtensor._local_tensor = st._local_tensor
    dtensor._spec = st._spec
    return dtensor


def _dtensor_to_shard_tensor(dtensor: DTensor, spec: ShardTensorSpec) -> "ShardTensor":
    r"""Promote a DTensor to a ShardTensor (no autograd).

    Callers must supply a resolved ``spec``.  Use inside backward (with spec
    from ctx) or after resolving a spec via :func:`_resolve_spec_for_dtensor`.
    """
    if isinstance(dtensor, ShardTensor):
        # Shortcut if we're already a ShardTensor:
        return dtensor
    st = ShardTensor.__new__(
        ShardTensor,
        local_tensor=dtensor._local_tensor,
        spec=spec,
        requires_grad=dtensor.requires_grad,
    )
    return st


# ============================================================================
# Layer 2 -- Autograd Functions (use Layer 1 inside fwd / bwd)
# ============================================================================


class _DTensorToShardTensor(torch.autograd.Function):
    r"""Differentiable promotion: DTensor -> ShardTensor.

    This is to always connect the graphs for the backward pass
    when we have to use a fallback option.

    Forward: :func:`_dtensor_to_shard_tensor`.
    Backward: :func:`_shard_tensor_to_dtensor`.
    """

    @staticmethod
    def forward(ctx, dtensor: DTensor, spec: ShardTensorSpec) -> "ShardTensor":
        return _dtensor_to_shard_tensor(dtensor, spec)

    @staticmethod
    def backward(ctx, grad_output: "ShardTensor"):
        return _shard_tensor_to_dtensor(grad_output), None


class _ShardTensorToDTensor(torch.autograd.Function):
    r"""Differentiable conversion: ShardTensor -> DTensor.

    This is to always connect the graphs for the backward pass
    when we have to use a fallback option.

    Forward: :func:`_shard_tensor_to_dtensor` (caches spec).
    Backward: :func:`_dtensor_to_shard_tensor` (reuses cached spec).
    """

    @staticmethod
    def forward(ctx, st: "ShardTensor") -> DTensor:
        ctx.shard_tensor_spec = st._spec
        return _shard_tensor_to_dtensor(st)

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        return (_dtensor_to_shard_tensor(grad_output, ctx.shard_tensor_spec),)


# ============================================================================
# Layer 3 -- Smart single-tensor converters (auto-diff when grad_fn present)
# ============================================================================


def _resolve_spec_for_dtensor(
    dtensor: DTensor, input_args: tuple = ()
) -> ShardTensorSpec:
    r"""Resolve a ShardTensorSpec for *dtensor*.

    Tries to reuse a spec from a ShardTensor in *input_args* whose
    ``tensor_meta`` and ``placements`` match.  Falls back to chunk-based
    inference (no communication).
    """
    for arg in input_args:
        if (
            isinstance(arg, ShardTensor)
            and dtensor._spec.tensor_meta == arg._spec.tensor_meta
            and dtensor._spec.placements == arg._spec.placements
        ):
            return arg._spec
    return _infer_shard_tensor_spec_from_local_chunks(
        dtensor._local_tensor,
        dtensor._spec.mesh,
        dtensor._spec.placements,
        sharding_shapes="chunk",
        global_shape=dtensor.shape,
    )


# This is a thread-safe reentry guard.
# Goal is to prevent recursion into the fall back conversion paths.
# Here's the scenario we're preventing:
# 1. A ShardTensor needs to use the DTensor path in a torch_function level call.
#    This will enter torch_function for ShardTensor and trigger the fallback path.
# 2. Because that builds the autograd graph, the conversion from ShardTensor to DTensor
#    must be differentiable.
# 3. The conversion path itself will call _ShardTensorToDTensor, which will enter
#    torch_function for ShardTensor.
# 4. There is no overload for converting ShardTensor to DTensor, so it will
#    enter the fallback conversion path
# 5. Infinite recursion / profit.
_conversion_guard = threading.local()


def _conversion_active() -> bool:
    r"""Return whether ShardTensor<->DTensor conversion is currently active."""
    return getattr(_conversion_guard, "depth", 0) > 0


@contextmanager
def _conversion_scope():
    r"""Re-entrant conversion guard for cast-down/cast-up paths."""
    previous_depth = getattr(_conversion_guard, "depth", 0)
    _conversion_guard.depth = previous_depth + 1
    try:
        yield
    finally:
        if previous_depth == 0:
            delattr(_conversion_guard, "depth")
        else:
            _conversion_guard.depth = previous_depth


def _convert_st_to_dt(st: "ShardTensor") -> DTensor:
    r"""ShardTensor -> DTensor; differentiable when *st* is non-leaf."""
    with _conversion_scope():
        if st.requires_grad and st.grad_fn is not None:
            return _ShardTensorToDTensor.apply(st)
        return _shard_tensor_to_dtensor(st)


def _convert_dt_to_st(dtensor: DTensor, input_args: tuple = ()) -> "ShardTensor":
    r"""DTensor -> ShardTensor; differentiable when *dtensor* is non-leaf.

    Resolves spec, then uses Layer 2 or Layer 1 depending on whether the
    DTensor carries a ``grad_fn``.
    """
    if isinstance(dtensor, ShardTensor):
        return dtensor
    with _conversion_scope():
        spec = _resolve_spec_for_dtensor(dtensor, input_args)
        if dtensor.grad_fn is not None:
            return _DTensorToShardTensor.apply(dtensor, spec)
        res = _dtensor_to_shard_tensor(dtensor, spec)
        return res


def _dispatch_fallback_via_dtensor(
    func: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None = None,
) -> object:
    r"""Execute an ATen op through DTensor fallback and promote results back."""
    with _conversion_scope():
        converted_args = tuple(_convert_args_to_dtensor(arg) for arg in args)
        converted_kwargs = {
            k: _convert_args_to_dtensor(v) for k, v in (kwargs or {}).items()
        }
    dispatch_res = DTensor._op_dispatcher.dispatch(
        func, converted_args, converted_kwargs
    )
    with _conversion_scope():
        return _convert_results_to_shard_tensor(dispatch_res, args)


def _torch_function_fallback_via_dtensor(
    func: Callable,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None = None,
) -> object:
    r"""Execute a ``__torch_function__`` fallback through DTensor safely.

    The fallback call itself is wrapped in ``DisableTorchFunctionSubclass`` to
    avoid re-entering tensor-subclass ``__torch_function__`` while still
    allowing autograd to record DTensor ops.
    """

    with _conversion_scope():
        # Here, we take args and kwargs and push all ShardTensors to DTensors.
        # Other args are left as is.
        converted_args = tuple(_convert_args_to_dtensor(arg) for arg in args)
        converted_kwargs = {
            k: _convert_args_to_dtensor(v) for k, v in (kwargs or {}).items()
        }
    with torch._C.DisableTorchFunctionSubclass():
        result = func(*converted_args, **converted_kwargs)
    with _conversion_scope():
        # The output results promote any DTensor results back to ShardTensors
        converted_result = _convert_results_to_shard_tensor(result, args)
    return converted_result


# ============================================================================
# Layer 4 -- Recurse utilities (walk args / kwargs / results)
# ============================================================================


def _convert_args_to_dtensor(arg: object) -> object:
    r"""Recursively replace ShardTensors with DTensors in a single arg.

    Walks mappings, tuples, and lists. Each ShardTensor is converted via
    :func:`_convert_st_to_dt`.
    """
    match arg:
        case ShardTensor():
            return _convert_st_to_dt(arg)
        case DTensor():
            # DTensor can be iterable; exit early deliberatly
            return arg
        case Mapping():
            return type(arg)({k: _convert_args_to_dtensor(v) for k, v in arg.items()})
        case tuple():
            return tuple(_convert_args_to_dtensor(a) for a in arg)
        case list():
            return [_convert_args_to_dtensor(a) for a in arg]
        case _:
            return arg


def _convert_results_to_shard_tensor(result: object, input_args: tuple) -> object:
    r"""Recursively replace DTensors with ShardTensors in an op result.

    Walks single tensor, mappings, and iterables (excluding str/bytes).
    Each DTensor is converted via :func:`_convert_dt_to_st`.
    """
    if isinstance(result, DTensor):
        res = _convert_dt_to_st(result, input_args)
        return res
    if isinstance(result, Mapping):
        return type(result)(
            {
                k: _convert_dt_to_st(v, input_args) if isinstance(v, DTensor) else v
                for k, v in result.items()
            }
        )
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        return type(result)(
            _convert_dt_to_st(d, input_args) if isinstance(d, DTensor) else d
            for d in result
        )
    return result


class _ToTorchTensor(torch.autograd.Function):
    r"""Autograd function to convert a ShardTensor to a regular PyTorch tensor.

    This class handles the conversion from ShardTensor to ``torch.Tensor`` in both
    forward and backward passes, maintaining proper gradient flow. Slices the
    ShardTensor to the local component only on the current rank.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: "ShardTensor",
        grad_placements: Sequence[Placement] | None = None,
    ) -> torch.Tensor:
        r"""Convert ShardTensor to torch.Tensor in forward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context for saving tensors/variables for backward.
        input : ShardTensor
            ShardTensor to convert.
        grad_placements : Sequence[Placement], optional
            Sequence of placements to use for gradients.

        Returns
        -------
        torch.Tensor
            Local tensor representation of the ShardTensor.
        """
        ctx.shard_tensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor

        # JUST LIKE DTENSOR:
        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this ShardTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple["ShardTensor", None]:
        r"""Convert gradient torch.Tensor back to ShardTensor in backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors/variables from forward.
        grad_output : torch.Tensor
            Gradient tensor to convert back to ShardTensor.

        Returns
        -------
        Tuple[ShardTensor, None]
            Tuple containing the ShardTensor gradient and None for
            grad_placements gradient (not differentiable).
        """
        shard_tensor_spec = ctx.shard_tensor_spec
        mesh = shard_tensor_spec.mesh
        if ctx.grad_placements is not None:
            if ctx.grad_placements != shard_tensor_spec.placements:
                grad_placements = ctx.grad_placements
                grad_sharding_shapes = "infer"
            else:
                # If the placements are the same as the input placements,
                # we reuse the sharding sizes from the input placements.
                grad_placements = ctx.grad_placements
                grad_sharding_shapes = shard_tensor_spec._sharding_shapes
        else:
            grad_placements = shard_tensor_spec.placements
            grad_sharding_shapes = shard_tensor_spec._sharding_shapes
        if grad_sharding_shapes is None:
            grad_sharding_shapes = "infer"
        # Generate a spec based on grad outputs and the expected placements:
        grad_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            grad_output, mesh, grad_placements, grad_sharding_shapes
        )

        return (
            ShardTensor(
                grad_output, grad_tensor_spec, requires_grad=grad_output.requires_grad
            ),
            None,
        )


class _FromTorchTensor(torch.autograd.Function):
    r"""Autograd function for converting a torch.Tensor to a ShardTensor.

    This class handles the forward and backward passes for converting between
    ``torch.Tensor`` and ShardTensor types, maintaining gradient information.

    Global shape information is inferred using collective communication on
    the specified device mesh.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        local_input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "chunk",
    ) -> "ShardTensor":
        r"""Convert a local torch.Tensor to a ShardTensor in forward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context for saving tensors/variables for backward.
        local_input : torch.Tensor
            Local tensor to convert to ShardTensor.
        device_mesh : DeviceMesh
            Device mesh specifying process groups.
        placements : Tuple[Placement, ...]
            Tuple of placement rules for sharding.
        sharding_shapes : Union[str, Dict[int, List[Tuple[int, ...]]]], default="chunk"
            Controls how shard tensor spec is generated:

            - ``"chunk"``: Use ``torch.chunk`` shapes to infer shapes from
              global shape (no communication).
            - ``"infer"``: Use collective communication to infer shapes from
              mesh neighbors.
            - Manual dict mapping mesh dim to list of shard shapes: Use
              provided shapes. Must pass on each rank.

        Returns
        -------
        ShardTensor
            ShardTensor constructed from the local input tensor.
        """
        ctx.previous_placement = placements
        ctx.previous_mesh = device_mesh

        # This function is simpler than the corresponding DTensor implementation on the surface
        # because under the hood, we have some logic here to infer the sharding shapes.
        shard_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            local_input, device_mesh, placements, sharding_shapes
        )

        shard_tensor = ShardTensor(
            local_input,
            shard_tensor_spec,
            requires_grad=local_input.requires_grad,
        )

        return shard_tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "ShardTensor",
    ) -> tuple[torch.Tensor, None, None, None]:
        r"""Convert gradient ShardTensor back to torch.Tensor in backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors/variables from forward.
        grad_output : ShardTensor
            Gradient ShardTensor to convert back to torch.Tensor.

        Returns
        -------
        Tuple[torch.Tensor, None, None, None]
            Tuple containing the local tensor gradient, and None for
            device_mesh, placements, and sharding_shapes gradients
            (not differentiable).

        Raises
        ------
        RuntimeError
            If gradient tensor has different placement than original and
            the original placement contains partial placements.
        """
        previous_placement = ctx.previous_placement
        if grad_output.placements != previous_placement:
            # Automatically redistribute to the previous placement as long as it's not a partial.
            if not any(p.is_partial() for p in previous_placement):
                grad_output = grad_output.redistribute(
                    grad_output._spec.mesh, previous_placement
                )
            else:
                raise RuntimeError(
                    "Resharding gradients with partial placements not implemented"
                )

        return grad_output.to_local(), None, None, None


class ShardTensor(DTensor):
    r"""A distributed tensor class with support for uneven data sharding.

    Similar to PyTorch's native ``DTensor`` but with more flexibility for
    uneven data sharding. Leverages a very similar API to ``DTensor``
    (identical where possible) but deliberately tweaks routines to avoid
    implicit assumptions about tensor sharding.

    The key differences from ``DTensor`` are:

    - Supports uneven sharding where different ranks can have different
      local tensor sizes
    - Tracks and propagates shard size information across operations
    - Handles redistribution of unevenly sharded tensors
    - Provides custom collective operations optimized for uneven sharding

    Like ``DTensor``, operations are dispatched through PyTorch's dispatcher
    system. Most operations work by:

    1. Converting inputs to local tensors
    2. Performing the operation locally
    3. Constructing a new ShardTensor with appropriate sharding spec
    4. Handling any needed communication between ranks

    The class provides methods for:

    - Converting to/from local tensors
    - Redistributing between different sharding schemes
    - Performing collective operations like all_gather and reduce_scatter
    - Basic tensor operations that maintain sharding information

    Attributes
    ----------
    _local_tensor : torch.Tensor
        The local tensor data on this rank.
    _spec : ShardTensorSpec
        The specification defining sharding scheme and metadata.
    """

    _local_tensor: torch.Tensor
    _spec: ShardTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # For torch.ops.aten operators (low-level dispatch)
    _dispatch_registry: dict[torch._ops.OpOverload, Callable] = {}
    # Fallback by op name (e.g. "aten.neg.default") when the OpOverload
    # passed to __torch_dispatch__ is not the same object as the one used to register.
    _dispatch_registry_by_name: dict[str, Callable] = {}

    # For Python-level functions (torch.mean, tensor.mean, etc.)
    _function_registry: dict[Callable, Callable] = {}

    # For custom functions registered with PyTorch,
    # it is sometimes necessary to match by name.
    # For instance, if you declare an op with
    #
    # @torch.library.custom_op(
    #    "module::function_name", mutates_args=()
    # )
    # def function_external_to_torch(
    #
    # Then, you likely want to register the handler with
    #
    # ShardTensor.register_named_function_handler("module.function_name.default", handler)
    _named_function_registry: dict[str, Callable] = {}

    # Upon construction of any ShardTensor objects, this will be set to true.
    # Wrappers are triggered dynamically, so the wrapping will be pass-through
    # exclusively until true.
    _enable_shard_patches: bool = False

    @classmethod
    def patches_enabled(cls) -> bool:
        r"""Check whether patches are enabled for this class.

        Returns
        -------
        bool
            ``True`` if shard patches are enabled, ``False`` otherwise.
            Default is ``False`` until a ShardTensor is constructed.
        """
        return cls._enable_shard_patches

    @classmethod
    def register_dispatch_handler(
        cls, op: torch._ops.OpOverload, handler: Callable
    ) -> None:
        r"""Register a handler for a specific PyTorch operator in the dispatch system.

        Parameters
        ----------
        op : torch._ops.OpOverload
            The PyTorch operator to register a handler for.
        handler : Callable
            The handler function to call when the operator is invoked.
        """
        cls._dispatch_registry[op] = handler
        cls._dispatch_registry_by_name[str(op)] = handler

    @classmethod
    def register_function_handler(cls, func: Callable, handler: Callable) -> None:
        r"""Register a handler for a Python-level function or method.

        Parameters
        ----------
        func : Callable
            The Python function to register a handler for.
        handler : Callable
            The handler function to call when the function is invoked.
        """
        cls._function_registry[func] = handler

    @classmethod
    def register_named_function_handler(cls, func_name: str, handler: Callable) -> None:
        r"""Register a named function registered via ``torch.library.custom_op``.

        Parameters
        ----------
        func_name : str
            The string name of the custom op (e.g., ``"module.function_name.default"``).
        handler : Callable
            The handler function to call when the function is invoked.
        """
        cls._named_function_registry[func_name] = handler

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: ShardTensorSpec,
        *,
        requires_grad: bool,
    ) -> "ShardTensor":
        r"""Construct a new ShardTensor from a local tensor and specification.

        Note that unlike ``DTensor``, ShardTensor will automatically collect
        the shard size information from all participating devices. This enables
        uneven and dynamic sharding.

        Parameters
        ----------
        local_tensor : torch.Tensor
            Local tensor to use as the data.
        spec : ShardTensorSpec
            ShardTensorSpec defining the sharding scheme.
        requires_grad : bool
            Whether the tensor requires gradients.

        Returns
        -------
        ShardTensor
            A new ShardTensor instance.

        Note
        ----
        This implementation is heavily derived from ``torch.distributed.tensor.DTensor``.
        """
        if local_tensor.requires_grad and not requires_grad:
            warn(
                "To construct a new ShardTensor from torch.Tensor, "
                "it's recommended to use local_tensor.detach() and "
                "make requires_grad consistent."
            )

        if spec.tensor_meta is None:
            raise ValueError("TensorMeta should not be None!")

        # Check the sharding information is known:
        ret = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )

        ret._spec = spec
        ret._local_tensor = local_tensor

        cls._enable_shard_patches = True

        return ret

    def __repr__(self) -> str:
        return (
            "ShardTensor("
            f"local_tensor={repr(self._local_tensor)}, "
            f"device_mesh={repr(self._spec.mesh)}, "
            f"placements={repr(self._spec.placements)}"
            ")"
        )

    def __str__(self) -> str:
        # Avoid Tensor/DTensor string formatting paths that can re-enter dispatch.
        return self.__repr__()

    def __format__(self, format_spec: str) -> str:
        # Format as plain Python string to bypass tensor formatting internals.
        return format(str(self), format_spec)

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "ShardTensor":
        r"""Convert a DTensor to a ShardTensor.

        Differentiable when *dtensor* is non-leaf (has a ``grad_fn``).
        Spec is inferred from the DTensor (chunk-based, no communication).

        Parameters
        ----------
        dtensor : DTensor
            DTensor to convert.

        Returns
        -------
        ShardTensor
            Equivalent ShardTensor with the same local tensor and inferred spec.
        """
        return _convert_dt_to_st(dtensor)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if _conversion_active():
            # When converting shard tensor to dtensor, or dtensor to shard tensor,
            # we just skip this function entirely.
            return super().__torch_function__(func, types, args, kwargs)
        if func in cls._function_registry and cls._enable_shard_patches:
            return cls._function_registry[func](func, types, args, kwargs)
        if str(func) in cls._named_function_registry and cls._enable_shard_patches:
            return cls._named_function_registry[str(func)](func, types, args, kwargs)
        res = _torch_function_fallback_via_dtensor(func, args, kwargs)
        return res

    @classmethod
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> "ShardTensor" | Iterable["ShardTensor"] | object:
        # Use a handler, if we have one:
        handler = cls._dispatch_registry.get(func)
        if handler is None:
            handler = cls._dispatch_registry_by_name.get(str(func))
        if handler is not None:
            return handler(*args, **kwargs)
        # Otherwise, try the dtensor route:
        return _dispatch_fallback_via_dtensor(func, args, kwargs)

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "infer",
    ) -> "ShardTensor":
        r"""Generate a new ShardTensor from local torch tensors.

        Uses device mesh and placements to infer global tensor properties.
        No restriction is made on forcing tensors to have equal shapes locally.
        Instead, the requirement is that tensor shapes could be concatenated
        into a single tensor according to the placements.

        Parameters
        ----------
        local_tensor : torch.Tensor
            Local chunk of tensor. All participating tensors must be of the
            same rank and concatenatable across the mesh dimensions.
        device_mesh : Optional[DeviceMesh], optional
            Target device mesh. If not specified, will use the current mesh.
        placements : Optional[Sequence[Placement]], optional
            Target placements. Must have same number of elements as
            ``device_mesh.ndim``.
        sharding_shapes : Union[str, Dict[int, List[Tuple[int, ...]]]], default="infer"
            Controls how shard tensor spec is generated:

            - ``"chunk"``: Use ``torch.chunk`` shapes to infer shapes from
              global shape (no communication).
            - ``"infer"``: Use collective communication to infer shapes from
              mesh neighbors.
            - Manual dict mapping mesh dim to list of shard shapes: Use
              provided shapes. Must pass on each rank.

        Returns
        -------
        ShardTensor
            A new ShardTensor instance.
        """

        # This implementation follows the pytorch DTensor Implementation Closely.
        device_mesh = device_mesh or _mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type

        # convert the local tensor to desired device base on device mesh's device_type
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]
        else:
            placements = list(placements)
            for idx, placement in enumerate(placements):
                # normalize shard dim to be positive
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    if placement.dim < 0:
                        placements[idx] = Shard(placement.dim + local_tensor.ndim)

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor,
            device_mesh,
            tuple(placements),
            sharding_shapes,
        )

    def offsets(self, mesh_dim: int | None = None) -> list[int] | int:
        r"""Get offsets of shards along a mesh dimension.

        Parameters
        ----------
        mesh_dim : Optional[int], optional
            Mesh dimension to get offsets for. If ``None``, returns all offsets.

        Returns
        -------
        Union[List[int], int]
            List of offsets for shards along all dimensions, or single offset
            if ``mesh_dim`` is specified.
        """
        return self._spec.offsets(mesh_dim)

    def redistribute(
        self,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
    ) -> "ShardTensor":
        r"""Redistribute tensor across device mesh with new placement scheme.

        Like ``DTensor.redistribute`` but uses custom layer for shard
        redistribution that supports uneven sharding.

        Parameters
        ----------
        device_mesh : Optional[DeviceMesh], optional
            Target device mesh. Uses current mesh if ``None``.
        placements : Optional[Sequence[Placement]], optional
            Target placement scheme. Required.
        async_op : bool, default=False
            Whether to run asynchronously.

        Returns
        -------
        ShardTensor
            Redistributed ShardTensor with new placement scheme.

        Raises
        ------
        RuntimeError
            If placements is not specified or contains invalid placements
            (e.g., ``Partial`` placements or negative shard dimensions).
        """

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to Partial, redistributing to Partial is for internal use only!"
                )
            elif isinstance(placement, Shard) and placement.dim < 0:
                # normalize shard dim to be positive
                placements[i] = Shard(placement.dim + self.ndim)
        placements = tuple(placements)

        return ShardRedistribute.apply(self, device_mesh, placements, async_op)

    def to_local(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        r"""Get local tensor from this ShardTensor.

        Parameters
        ----------
        grad_placements : Optional[Sequence[Placement]], optional
            Future layout of gradients. If provided, gradients will be
            constructed with this placement scheme during backward pass.

        Returns
        -------
        torch.Tensor
            Local tensor. Shape may vary between ranks for sharded tensors.
        """

        if not torch.is_grad_enabled():
            return self._local_tensor

        if grad_placements is not None:
            grad_placements = tuple(grad_placements)

        return _ToTorchTensor.apply(self, grad_placements)

    def full_tensor(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        r"""Gather the full tensor from all ranks.

        Redistributes to ``Replicate`` placement on all mesh dimensions and
        returns the local tensor.

        Parameters
        ----------
        grad_placements : Optional[Sequence[Placement]], optional
            Future layout of gradients. If provided, gradients will be
            constructed with this placement scheme during backward pass.

        Returns
        -------
        torch.Tensor
            The full gathered tensor, identical on all ranks.
        """

        redist_res = self.redistribute(
            placements=[Replicate()] * self.device_mesh.ndim, async_op=False
        )
        if grad_placements is not None:
            grad_placements = tuple(grad_placements)
        return _ToTorchTensor.apply(redist_res, grad_placements)

    def backward(self, *args, **kwargs):
        r"""Perform backward pass for ShardTensor.

        Handles the redistribution of the tensor to resolve any partial
        placements before calling backward on the local tensor.

        Parameters
        ----------
        *args
            Positional arguments passed to ``torch.Tensor.backward``.
        **kwargs
            Keyword arguments passed to ``torch.Tensor.backward``.
        """

        # Before calling backward, we need to resolve any partial placements.
        new_placements = []
        needs_redistribute = False
        for placement in self._spec.placements:
            if placement.is_partial():
                new_placements.append(Replicate())
                needs_redistribute = True
            else:
                new_placements.append(placement)

        if needs_redistribute:
            self = self.redistribute(placements=new_placements)

        return self.to_local().backward(*args, **kwargs)


class FSDPOutputTensorAdapter(nn.Module):
    """Wrap a module and convert ShardTensor outputs to torch.Tensor."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return out.to_local() if isinstance(out, ShardTensor) else out


def wrap_for_fsdp(module: nn.Module) -> nn.Module:
    """Return a module wrapper that exposes tensor outputs for FSDP hooks."""
    return FSDPOutputTensorAdapter(module)


def distribute_over_domain_for_fsdp(
    module: nn.Module,
    device_mesh: DeviceMesh,
    partition_fn: (Callable[[str, nn.Module, DeviceMesh], None] | None) = None,
) -> nn.Module:
    """Distribute a module over a domain mesh and adapt outputs for FSDP."""
    distributed_module = distribute_module(
        module,
        device_mesh=device_mesh,
        partition_fn=partition_fn,
    )
    return wrap_for_fsdp(distributed_module)


def scatter_tensor(
    tensor: torch.Tensor,
    global_src: int,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
    global_shape: torch.Size | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
) -> "ShardTensor":
    r"""Distribute a tensor from source rank across devices on the mesh.

    Takes a tensor that exists on a single source rank and distributes it
    across a device mesh according to the specified placement scheme. For
    multi-dimensional meshes, it performs a flattened scatter operation
    before constructing the sharded tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to distribute. Must exist on source rank; can be ``None``
        on other ranks.
    global_src : int
        Global rank ID of the source process.
    mesh : DeviceMesh
        Device mesh defining the process topology.
    placements : Tuple[Placement, ...]
        Tuple of placement specifications defining how to distribute the tensor.
    global_shape : Optional[torch.Size], optional
        Global shape of the tensor. If ``None``, will be broadcast from source.
    dtype : Optional[torch.dtype], optional
        Data type of the tensor. If ``None``, will be broadcast from source.
    requires_grad : bool, default=False
        Whether the resulting ShardTensor requires gradients.

    Returns
    -------
    ShardTensor
        The distributed tensor with specified placements.

    Raises
    ------
    ValueError
        If ``global_src`` is not an integer or not in the mesh.
    """
    dm = DistributedManager()

    if not isinstance(global_src, int):
        raise ValueError("Global source must be an integer rank")
    if global_src not in mesh.mesh:
        raise ValueError("Please specify a tensor source in this mesh")

    is_src = dm.rank == global_src

    # For multi-dimensional meshes, we use a flattened process group
    mesh_group = dm.get_mesh_group(mesh)

    # Broadcast tensor metadata from source
    if global_shape is None or dtype is None:
        if dm.rank == global_src:
            meta = [TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)]
        else:
            meta = [None]

        dist.broadcast_object_list(meta, src=global_src, group=mesh_group)

        local_meta = meta[0]
    else:
        stride = _stride_from_contiguous_shape_C_style(global_shape)
        local_meta = TensorMeta(global_shape, stride, dtype)

    # This needs to be optimized, but I want to get the whole pipeline optimized first.
    # This only gets done when scatter_tensor is called and it should be relatively small
    # in full applications.

    # What isn't optimized?  Broadcasting the full tensor when placement is likely
    # Shard on at least one mesh dimension.  It would be more efficient to iteratively
    # scatter along Shard dimensions.  BUT, the focus is on performance of full applications
    # and this is a once-per-iteration cost.

    # Broadcast the tensor to all ranks
    if tensor is None and not is_src:
        # Tensor is allowed to be none if not on the root rank
        tensor = torch.empty(local_meta.shape, dtype=local_meta.dtype, device=dm.device)

    dist.broadcast(tensor, src=global_src, group=mesh_group)

    # Create a fully-replicated spec:
    spec = ShardTensorSpec(
        mesh=mesh,
        placements=[Replicate() for _ in range(mesh.ndim)],
        tensor_meta=local_meta,
        _sharding_shapes={},
    )

    # Make a "fully-replicated" tensor on all ranks:
    st = ShardTensor.__new__(
        ShardTensor,
        local_tensor=tensor,
        spec=spec,
        requires_grad=requires_grad,
    )

    # Redistribute the tensor to the desired placements:
    st = st.redistribute(mesh, placements, async_op=False)
    # This is an unoptimal step but is functional:
    if requires_grad:
        st = st.detach()
        st.requires_grad = True
    return st
