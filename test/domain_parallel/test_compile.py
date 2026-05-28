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

r"""Tests for ShardTensor integration with ``torch.compile`` / AOTAutograd.

The focus is on the runtime tangent-coercion hook
``ShardTensor.__coerce_same_metadata_as_tangent__``, which AOTAutograd
invokes during the compiled backward when the runtime tangent's spec
doesn't match the recorded one. The tests cover uneven sharding, which
DTensor does not have to handle and which earlier coerce implementations
silently dropped (defaulting back to even chunking).
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel._shard_tensor_spec import ShardTensorSpec
from test.domain_parallel.test_redistribute import shard_tensor_factory


def _replicate_placements(mesh):
    return [Replicate()] * mesh.ndim


def run_coerce_replicate_to_uneven_shard(mesh):
    # Round-trip: uneven Shard -> Replicate -> coerce back to recorded uneven Shard.
    st_uneven = shard_tensor_factory(mesh, uneven=True)
    recorded_spec = st_uneven._spec
    expected_local_shape = tuple(st_uneven._local_tensor.shape)
    expected_full = st_uneven.full_tensor().clone()

    st_replicated = st_uneven.redistribute(placements=_replicate_placements(mesh))

    coerced = st_replicated.__coerce_same_metadata_as_tangent__((recorded_spec, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded_spec.placements
    assert tuple(coerced._local_tensor.shape) == expected_local_shape
    assert torch.allclose(coerced.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_replicate_to_uneven_shard_1d(distributed_mesh):
    run_coerce_replicate_to_uneven_shard(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_replicate_to_uneven_shard_2d(distributed_mesh_2d):
    run_coerce_replicate_to_uneven_shard(distributed_mesh_2d)


def run_coerce_same_placements_unknown_shapes(mesh):
    # Recorded spec carries the same placements but no _sharding_shapes; the
    # hook must accept it without erroring and preserve local data.
    st = shard_tensor_factory(mesh, uneven=True)
    expected_local_shape = tuple(st._local_tensor.shape)
    expected_full = st.full_tensor().clone()

    modified_spec = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=st._spec.placements,
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )

    coerced = st.__coerce_same_metadata_as_tangent__((modified_spec, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == st._spec.placements
    assert coerced._spec._sharding_shapes is None
    assert tuple(coerced._local_tensor.shape) == expected_local_shape
    assert torch.allclose(coerced.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_same_placements_unknown_shapes_1d(distributed_mesh):
    run_coerce_same_placements_unknown_shapes(distributed_mesh)


def run_coerce_expected_type_returns_none(mesh):
    # Mismatched expected_type must short-circuit to None (DTensor convention).
    st = shard_tensor_factory(mesh, uneven=True)
    out = st.__coerce_same_metadata_as_tangent__(
        (st._spec, False), expected_type=torch.Tensor
    )
    assert out is None


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_expected_type_returns_none_1d(distributed_mesh):
    run_coerce_expected_type_returns_none(distributed_mesh)


def _sum_squares(x):
    return (x**2).sum()


def run_compile_backward_uneven_shard(mesh):
    # Smoke test: compile + backward over an uneven ShardTensor must not raise
    # AOTAutograd's "guessed metadata incorrectly" tangent error. Gradient values
    # are validated by the direct __coerce_same_metadata_as_tangent__ tests.
    x = shard_tensor_factory(mesh, uneven=True).detach().requires_grad_(True)

    torch._dynamo.reset()
    compiled = torch.compile(_sum_squares, fullgraph=True, backend="aot_eager")

    loss = compiled(x)
    loss.backward()


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_backward_uneven_shard_1d(distributed_mesh):
    run_compile_backward_uneven_shard(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_backward_uneven_shard_2d(distributed_mesh_2d):
    run_compile_backward_uneven_shard(distributed_mesh_2d)
