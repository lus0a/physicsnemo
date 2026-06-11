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

"""Distributed regression tests for the Newton NeRD trainer batch sharding."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.integrations.newton import (
    NeRDTrainingConfig,
)
from physicsnemo.experimental.integrations.newton.nerd import (
    JointLayout,
    NeRDBodyStateCodec,
    NeRDDataset,
    NeRDJointStateCodec,
    NeRDModelSpec,
    train_nerd,
)
from physicsnemo.experimental.models.nerd import NeRDTransformer


def _small_nerd_model(spec: NeRDModelSpec) -> NeRDTransformer:
    """Build a small Transformer for the distributed trainer regression."""
    return NeRDTransformer(
        input_dim=spec.input_dim,
        prediction_dim=spec.prediction_dim,
        context_frames=spec.context_frames,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=8,
        head_hidden=8,
    )


def _run_uneven_trainer_batch_sharding(rank: int, world_size: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    DistributedManager._shared_state = {}
    DistributedManager.initialize()
    device = DistributedManager().device

    # Always uneven for world_size > 1, without fixing the test to a rank count.
    global_batch_size = 2 * world_size - 1
    expected_local_batch = 2 if rank < world_size - 1 else 1

    layout = JointLayout(
        world_count=2,
        dof_q=2,
        dof_qd=2,
        continuous_q_mask=torch.tensor([False, True]),
        base_translation_mask=torch.tensor([True, False]),
        root_is_free=False,
        up_axis_index=2,
        quaternion_q_starts=(),
    )
    time = torch.arange(7, dtype=torch.float32).view(1, 7, 1)
    q = torch.cat((0.01 * time, 0.02 * time), dim=-1).repeat(2, 1, 1)
    qd = torch.cat((0.03 * time, 0.04 * time), dim=-1).repeat(2, 1, 1)
    transition_condition = torch.full((2, 6, 1), 0.1 * (rank + 1))
    trained = train_nerd(
        NeRDDataset(
            states=torch.cat((q, qd), dim=-1),
            inputs=transition_condition,
            codec=NeRDJointStateCodec(layout=layout, robot_centric=False),
        ),
        NeRDTrainingConfig(
            context_frames=3,
            epochs=1,
            steps_per_epoch=2,
            batch_size=global_batch_size,
        ),
        dynamics_model=_small_nerd_model,
        device=device,
        log=lambda _message: None,
    )

    states = torch.zeros(2, 7, 2, 13)
    states[..., 6] = 1.0
    states[..., 0] = 0.01 * time
    states[..., 7] = 0.02 * time
    inputs = torch.full((2, 6, 1), 0.1 * (rank + 1))
    body_codec = NeRDBodyStateCodec(
        SimpleNamespace(
            world_count=1,
            body_count=2,
            body_world=torch.tensor([0, 0]),
        )
    )
    entity = train_nerd(
        NeRDDataset(states=states, inputs=inputs, codec=body_codec),
        NeRDTrainingConfig(
            context_frames=3,
            epochs=1,
            steps_per_epoch=2,
            batch_size=global_batch_size,
        ),
        dynamics_model="NeRDEntityTransformer",
        model_kwargs={
            "hidden_size": 8,
            "entity_depth": 1,
            "temporal_depth": 1,
            "num_heads": 2,
            "head_hidden": 8,
            "head_layers": 1,
        },
        device=device,
        log=lambda _message: None,
    )

    for metadata in (trained.metadata, entity.metadata):
        assert metadata["distributed_world_size"] == world_size
        assert metadata["global_batch_size"] == global_batch_size
        assert metadata["local_batch_size"] == expected_local_batch
        assert metadata["min_local_batch_size"] == 1
        assert metadata["max_local_batch_size"] == 2

    DistributedManager.cleanup()


@pytest.mark.multigpu_dynamic
def test_nerd_distributed_uses_active_world_size(monkeypatch) -> None:
    world_size = torch.accelerator.device_count()
    assert world_size > 1, "Not enough GPUs available for distributed NeRD test"
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12385")
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.spawn(
        _run_uneven_trainer_batch_sharding,
        args=(world_size,),
        nprocs=world_size,
        join=True,
        daemon=True,
    )
