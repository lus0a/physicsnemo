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

import tempfile
from pathlib import Path

import pytest
import torch

from physicsnemo.core import Module
from physicsnemo.experimental.models.nerd import NeRDEntityTransformer
from physicsnemo.nn import TimmSelfAttention

FEATURE_DIM, PREDICTION_DIM = 19, 12
ENTITY_COUNT, CONTEXT_FRAMES = 7, 5


def _model(device):
    return NeRDEntityTransformer(
        feature_dim=FEATURE_DIM,
        prediction_dim=PREDICTION_DIM,
        num_entities=ENTITY_COUNT,
        context_frames=CONTEXT_FRAMES,
        hidden_size=32,
        entity_depth=2,
        temporal_depth=2,
        num_heads=4,
        head_hidden=32,
    ).to(device)


def _inputs(device, frames=CONTEXT_FRAMES):
    return torch.randn(2, frames, ENTITY_COUNT, FEATURE_DIM, device=device)


def test_nerd_entity_forward_shape(device):
    model = _model(device).eval()
    inputs = _inputs(device)
    with torch.no_grad():
        prediction = model(inputs)
    assert prediction.shape == (2, CONTEXT_FRAMES, ENTITY_COUNT, PREDICTION_DIM)


def test_nerd_entity_uses_physicsnemo_attention():
    model = _model("cpu")
    assert isinstance(model.entity_blocks[0].attn, TimmSelfAttention)
    assert isinstance(model.temporal_blocks[0].attn, TimmSelfAttention)


def test_nerd_entity_temporal_attention_is_causal(device):
    model = _model(device).eval()
    inputs = _inputs(device)
    changed = inputs.clone()
    changed[:, -1] += 100.0
    with torch.no_grad():
        before = model(inputs)
        after = model(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1.0e-6)
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_nerd_entity_accepts_short_history(device):
    model = _model(device).eval()
    inputs = _inputs(device, frames=2)
    with torch.no_grad():
        prediction = model(inputs)
    assert prediction.shape == (2, 2, ENTITY_COUNT, PREDICTION_DIM)


def test_nerd_entity_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        NeRDEntityTransformer(
            feature_dim=13,
            prediction_dim=12,
            num_entities=2,
            hidden_size=30,
            num_heads=4,
        )


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (
            torch.randn(2, CONTEXT_FRAMES, FEATURE_DIM),
            "features must have shape",
        ),
        (
            torch.randn(2, CONTEXT_FRAMES, ENTITY_COUNT, FEATURE_DIM + 1),
            "expected feature_dim",
        ),
    ],
)
def test_nerd_entity_rejects_bad_input_shapes(features, message):
    with pytest.raises(ValueError, match=message):
        _model("cpu")(features)


def test_nerd_entity_from_checkpoint(device):
    model = _model(device).eval()
    inputs = _inputs(device)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nerd_entity.mdlus"
        model.save(str(path))
        loaded = Module.from_checkpoint(str(path)).to(device).eval()
    assert isinstance(loaded, NeRDEntityTransformer)
    with torch.no_grad():
        assert torch.allclose(model(inputs), loaded(inputs), atol=1.0e-6)
