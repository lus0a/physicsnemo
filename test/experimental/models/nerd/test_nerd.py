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
from physicsnemo.experimental.models.nerd import NeRDTransformer
from physicsnemo.nn import TimmSelfAttention
from test import common

INPUT_DIM, PREDICTION_DIM, CONTEXT_FRAMES = 6, 4, 10


def _model(device):
    return NeRDTransformer(
        input_dim=INPUT_DIM,
        prediction_dim=PREDICTION_DIM,
        context_frames=CONTEXT_FRAMES,
        n_layer=2,
        n_head=4,
        n_embd=32,
    ).to(device)


def _tokens(device, batch=2, time_count=CONTEXT_FRAMES):
    return torch.randn(batch, time_count, INPUT_DIM).to(device)


def test_nerd_constructor(device):
    model = _model(device)
    assert model.input_dim == INPUT_DIM
    assert model.prediction_dim == PREDICTION_DIM
    assert model.context_frames == CONTEXT_FRAMES


def test_nerd_reference_cartpole_parameter_count():
    model = NeRDTransformer(
        input_dim=6,
        prediction_dim=4,
        context_frames=10,
        block_size=32,
        n_layer=6,
        n_head=12,
        n_embd=192,
        head_hidden=64,
        bias=False,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_713_668


def test_nerd_uses_physicsnemo_attention():
    model = _model("cpu")
    assert isinstance(model.blocks[0].attn, TimmSelfAttention)


def test_nerd_forward_shape(device):
    model = _model(device).eval()
    tokens = _tokens(device)
    with torch.no_grad():
        out = model(tokens)
    assert out.shape == (2, CONTEXT_FRAMES, PREDICTION_DIM)


def test_nerd_attention_is_causal(device):
    model = _model(device).eval()
    tokens = _tokens(device)
    changed = tokens.clone()
    changed[:, -1] += 100.0
    with torch.no_grad():
        before = model(tokens)
        after = model(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1.0e-6)
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_nerd_accepts_short_sequence(device):
    # A window shorter than context_frames must still run (warm-up at deployment).
    model = _model(device).eval()
    with torch.no_grad():
        out = model(_tokens(device, time_count=3))
    assert out.shape == (2, 3, PREDICTION_DIM)


def test_nerd_rejects_sequence_longer_than_block_size(device):
    model = NeRDTransformer(
        input_dim=INPUT_DIM, prediction_dim=PREDICTION_DIM, block_size=16
    ).to(device)
    with pytest.raises(ValueError, match="block_size"):
        model(_tokens(device, time_count=32))


def test_nerd_block_size_must_cover_context():
    with pytest.raises(ValueError, match="block_size"):
        NeRDTransformer(
            input_dim=INPUT_DIM,
            prediction_dim=PREDICTION_DIM,
            block_size=4,
            context_frames=10,
        )


def test_nerd_head_must_divide_embedding():
    with pytest.raises(ValueError, match="divisible"):
        NeRDTransformer(
            input_dim=INPUT_DIM, prediction_dim=PREDICTION_DIM, n_head=5, n_embd=32
        )


def test_nerd_checkpoint(device):
    torch.manual_seed(0)
    model_1 = _model(device)
    torch.manual_seed(1)
    model_2 = _model(device)
    assert common.validate_checkpoint(model_1, model_2, (_tokens(device),))


def test_nerd_from_checkpoint(device):
    torch.manual_seed(0)
    original = _model(device).eval()
    tokens = _tokens(device)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nerd.mdlus"
        original.save(str(path))
        loaded = Module.from_checkpoint(str(path)).to(device).eval()
    assert isinstance(loaded, NeRDTransformer)
    assert loaded.prediction_dim == original.prediction_dim
    with torch.no_grad():
        assert torch.allclose(original(tokens), loaded(tokens), atol=1e-6)
