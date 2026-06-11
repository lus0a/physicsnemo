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

"""Tests for reusable bounded design spaces and constraints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physicsnemo.experimental.integrations.newton import (
    DesignRegularizer,
    DesignSpace,
    DesignVariable,
    SimilarityConstraint,
    SmoothnessConstraint,
    select_diverse_designs,
    select_verified_design,
)


def _design_space() -> DesignSpace:
    return DesignSpace(
        (
            DesignVariable("length_0", 1.0, 3.0, tags=("length",)),
            DesignVariable("length_1", 1.0, 3.0, tags=("length",)),
            DesignVariable("thickness", 1.0e-3, 1.0e-1, scale="log"),
            DesignVariable("part_count", 2.0, 5.0, kind="integer"),
        )
    )


def test_design_space_round_trip_sampling_and_schema() -> None:
    design_space = _design_space()
    normalized = design_space.sample_sobol(8, seed=12)
    physical = design_space.decode(normalized)

    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == (8, 4)
    np.testing.assert_allclose(
        design_space.encode(physical), normalized, rtol=1.0e-5, atol=1.0e-6
    )
    assert design_space.indices(tag="length") == (0, 1)
    assert design_space.index("thickness") == 2
    assert len(design_space.fingerprint) == 64
    restored = DesignSpace.from_config(design_space.to_config())
    assert restored == design_space
    assert restored.fingerprint == design_space.fingerprint
    assert (
        design_space.fingerprint
        != DesignSpace((DesignVariable("other", 0.0, 1.0),)).fingerprint
    )

    realized = design_space.decode(
        np.asarray([[0.0, 0.5, 1.0, 0.51]], dtype=np.float32),
        realize_discrete=True,
    )
    np.testing.assert_allclose(realized[0], [1.0, 2.0, 0.1, 4.0])
    named = design_space.decode_named(normalized[:2], realize_discrete=True)
    assert set(named) == set(design_space.names)
    assert named["length_0"].shape == (2,)


def test_design_space_torch_decode_preserves_gradients() -> None:
    design_space = _design_space()
    normalized = torch.full((2, 4), 0.4, requires_grad=True)
    physical = design_space.decode(normalized)
    assert isinstance(physical, torch.Tensor)

    physical.sum().backward()
    assert normalized.grad is not None
    assert torch.isfinite(normalized.grad).all()
    torch.testing.assert_close(
        design_space.encode(physical.detach()), normalized.detach()
    )


def test_design_space_validates_schema_and_shapes() -> None:
    with pytest.raises(ValueError, match="unique"):
        DesignSpace(
            (
                DesignVariable("x", 0.0, 1.0),
                DesignVariable("x", 1.0, 2.0),
            )
        )
    with pytest.raises(ValueError, match="integer bounds"):
        DesignVariable("count", 1.5, 4.0, kind="integer")
    with pytest.raises(ValueError, match="dimension 4"):
        _design_space().decode(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(KeyError, match="unknown"):
        _design_space().index("missing")


def test_similarity_and_smoothness_constraints_are_composable() -> None:
    design_space = _design_space()
    regularizer = DesignRegularizer(
        design_space,
        (
            SimilarityConstraint(
                (("length_0", "length_1"),),
                weight=2.0,
            ),
            SmoothnessConstraint(
                ("length_0", "length_1", "thickness"),
                weight=0.5,
                order=2,
            ),
        ),
    )
    equal = torch.tensor([[0.4, 0.4, 0.4, 0.4]])
    varied = torch.tensor([[0.1, 0.9, 0.2, 0.4]])

    assert regularizer(equal).shape == (1,)
    assert regularizer(varied).item() > regularizer(equal).item()
    assert SimilarityConstraint(
        (("length_0", "length_1"),), coordinate_space="physical"
    ).penalty(equal, design_space).item() == pytest.approx(0.0)


def test_select_diverse_designs_prefers_separation_then_fills() -> None:
    designs = np.asarray(
        (
            (0.00, 0.00),
            (0.01, 0.00),
            (0.90, 0.90),
            (0.45, 0.45),
        ),
        dtype=np.float32,
    )
    scores = np.asarray((0.0, 0.1, 0.2, 0.3), dtype=np.float32)

    np.testing.assert_array_equal(
        select_diverse_designs(
            designs,
            scores,
            count=3,
            min_distance=0.2,
        ),
        (0, 2, 3),
    )
    np.testing.assert_array_equal(
        select_diverse_designs(
            designs[:2],
            scores[:2],
            count=2,
            min_distance=1.0,
        ),
        (0, 1),
    )


def test_select_diverse_designs_preserves_group_coverage() -> None:
    designs = np.asarray(
        (
            (0.00, 0.00),
            (0.05, 0.00),
            (0.50, 0.50),
            (0.55, 0.50),
            (0.90, 0.90),
            (0.95, 0.90),
        ),
        dtype=np.float32,
    )
    scores = np.asarray((0.0, 0.1, 0.2, 0.3, 4.0, 5.0), dtype=np.float32)
    groups = np.asarray((0, 0, 1, 1, 2, 2))

    selected = select_diverse_designs(
        designs,
        scores,
        count=4,
        min_distance=0.1,
        group_ids=groups,
        min_per_group=1,
    )

    assert set(groups[selected]) == {0, 1, 2}
    np.testing.assert_array_equal(selected, (0, 2, 4, 1))


def test_select_diverse_designs_preserves_required_anchors() -> None:
    designs = np.eye(4, dtype=np.float32)
    scores = np.asarray((0.0, 0.1, 8.0, 9.0), dtype=np.float32)

    selected = select_diverse_designs(
        designs,
        scores,
        count=3,
        required_indices=(3, 2),
    )

    np.testing.assert_array_equal(selected, (3, 2, 0))
    with pytest.raises(ValueError, match="accommodate"):
        select_diverse_designs(
            designs,
            scores,
            count=1,
            required_indices=(2, 3),
        )


def test_select_verified_design_retains_or_accepts_against_incumbent() -> None:
    retained = select_verified_design(
        np.asarray((0.30, 0.18, 0.20)),
        ("proposal", "proposal", "incumbent"),
        min_improvement=0.03,
    )
    assert retained.index == 2
    assert retained.proposal_index == 1
    assert retained.incumbent_index == 2
    assert not retained.accepted

    accepted = select_verified_design(
        np.asarray((0.30, 0.15, 0.20)),
        ("proposal", "proposal", "incumbent"),
        min_improvement=0.03,
    )
    assert accepted.index == 1
    assert accepted.accepted


def test_select_verified_design_treats_positive_infinity_as_ineligible() -> None:
    retained = select_verified_design(
        np.asarray([np.inf, 0.4]),
        ("proposal", "incumbent"),
    )
    assert retained.index == 1
    assert not retained.accepted

    accepted = select_verified_design(
        np.asarray([0.3, np.inf]),
        ("proposal", "incumbent"),
    )
    assert accepted.index == 0
    assert accepted.accepted

    with pytest.raises(ValueError, match="at least one finite"):
        select_verified_design(
            np.asarray([np.inf, np.inf]),
            ("proposal", "incumbent"),
        )
    with pytest.raises(ValueError, match="proposal or incumbent"):
        select_verified_design(
            np.asarray([np.inf, np.inf, 0.0]),
            ("proposal", "incumbent", "diagnostic"),
        )
