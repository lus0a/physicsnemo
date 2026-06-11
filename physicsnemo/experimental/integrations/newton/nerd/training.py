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

"""Distributed NeRD training and the collect-then-train ``fit_nerd`` entry point."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from physicsnemo.experimental.integrations.newton.distributed import (
    _all_reduce,
    _distribute_model,
    _rank_world_size,
    resolve_device,
)
from physicsnemo.experimental.integrations.newton.nerd.deploy import TrainedNeRDModel
from physicsnemo.experimental.integrations.newton.nerd.model_builders import (
    _resolve_nerd_model,
)
from physicsnemo.experimental.integrations.newton.nerd.problem import (
    NeRDDataset,
    NeRDProblem,
    collect_nerd_trajectories,
)
from physicsnemo.experimental.integrations.newton.nerd.runtime import (
    NeRDNormalizers,
    _append_inputs,
    _global_abs_max,
    _global_mean_std,
    _model_class_name,
    _model_prediction,
    _nerd_log,
)
from physicsnemo.experimental.integrations.newton.nerd.spec import (
    NeRDModelSpec,
    NeRDTrainingConfig,
)


def train_nerd(
    trajectories: NeRDDataset,
    config: NeRDTrainingConfig | None = None,
    *,
    dynamics_model: str | torch.nn.Module | Callable[[NeRDModelSpec], torch.nn.Module],
    model_kwargs: dict[str, Any] | None = None,
    loss_weights: torch.Tensor | np.ndarray | None = None,
    device: str | torch.device | None = None,
    seed: int = 0,
    log: Callable[[str], None] | None = _nerd_log,
) -> TrainedNeRDModel:
    """Train one deployment-aligned NeRD model from any supported state codec.

    ``dynamics_model`` explicitly selects a supported PhysicsNeMo
    ``physicsnemo.core.ModelRegistry`` name, a ready module, or a callable
    receiving :class:`NeRDModelSpec`. Choose ``"NeRDTransformer"`` for vector
    state or ``"NeRDEntityTransformer"`` for entity-token state. The other
    directly adapted registry model is ``"FullyConnected"``; use a builder for
    any other architecture. When ``device`` is omitted, training follows the
    trajectory data or active distributed rank.

    Note that ``"FullyConnected"`` applies its linear layers along the feature
    dimension only and cannot mix information across the causal history, so it is
    effectively a single-frame (Markov) dynamics model: training still supplies a
    gradient from every frame in the window, but at inference only the final
    frame of the ``context_frames`` window determines the prediction. Choose a
    transformer model when history must be used.
    """
    config = config or NeRDTrainingConfig()
    torch_device = resolve_device(
        device if device is not None else trajectories.states.device
    )
    rank, world_size = _rank_world_size()
    if config.batch_size < world_size:
        raise ValueError(
            f"global batch_size ({config.batch_size}) must be at least the "
            f"distributed world size ({world_size})"
        )
    base_batch, remainder = divmod(config.batch_size, world_size)
    local_batch = base_batch + int(rank < remainder)
    max_local_batch = base_batch + int(remainder > 0)
    torch.manual_seed(seed)

    states = trajectories.states.to(torch_device, dtype=torch.float32)
    inputs = trajectories.inputs.to(torch_device, dtype=torch.float32)
    if states.shape[0] == 0:
        raise ValueError("each distributed rank must receive at least one trajectory")
    trajectory_count = torch.tensor(
        states.shape[0], dtype=torch.int64, device=torch_device
    )
    _all_reduce(trajectory_count)
    transition_count = states.shape[1] - 1
    if transition_count < config.context_frames:
        raise ValueError(
            f"need at least context_frames ({config.context_frames}) transitions, "
            f"got {transition_count}"
        )

    current, next_state = states[:, :-1], states[:, 1:]
    encoded = trajectories.codec.encode_state(current)
    model_inputs = _append_inputs(encoded, inputs)
    targets = trajectories.codec.state_to_delta(current, next_state)
    input_mean, input_std = _global_mean_std(model_inputs, config.normalization_floor)
    target_mean, target_std = _global_mean_std(targets, config.normalization_floor)
    normalizers = NeRDNormalizers(input_mean, input_std, target_mean, target_std)
    inputs_norm = (model_inputs - input_mean) / input_std
    targets_norm = (targets - target_mean) / target_std

    active = _global_abs_max(targets) > config.active_delta_threshold
    if loss_weights is None:
        weights = torch.ones_like(active, dtype=torch.float32)
    else:
        weights = torch.as_tensor(
            loss_weights, device=torch_device, dtype=torch.float32
        )
        if weights.shape != targets.shape[2:]:
            raise ValueError(
                f"loss_weights must have shape {tuple(targets.shape[2:])}, "
                f"got {tuple(weights.shape)}"
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
            raise ValueError("loss_weights must be finite and non-negative")
    weights = weights * active
    if not bool((weights > 0.0).any()):
        raise ValueError(
            "no active NeRD target channels remain after applying loss_weights"
        )
    weighted_count = weights.sum().clamp_min(1.0)

    spec = NeRDModelSpec(
        input_dim=int(model_inputs.shape[-1]),
        prediction_dim=int(targets.shape[-1]),
        context_frames=config.context_frames,
        input_shape=tuple(model_inputs.shape[2:]),
        prediction_shape=tuple(targets.shape[2:]),
    )
    model = _resolve_nerd_model(
        dynamics_model, spec, device=torch_device, model_kwargs=model_kwargs
    )
    model.eval()
    with torch.inference_mode():
        for length in sorted({1, config.context_frames}):
            _model_prediction(
                model, inputs_norm[:1, :length], trajectories.codec.prediction_shape
            )
    train_model = _distribute_model(model)
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=config.lr_start,
        weight_decay=config.weight_decay,
    )

    h = config.context_frames
    start_count = transition_count - h + 1
    generator = torch.Generator(device=torch_device).manual_seed(seed + rank)
    offsets = torch.arange(h, device=torch_device)
    global_weighted_count = config.batch_size * h * weighted_count
    curve: list[float] = []
    if rank == 0 and world_size > 1 and log is not None:
        log(
            f"NeRD DDP: {world_size} ranks, global batch {config.batch_size}, "
            f"local batches {base_batch}-{max_local_batch}"
        )

    for epoch in range(config.epochs):
        train_model.train()
        ratio = epoch / max(config.epochs - 1, 1)
        lr = config.lr_start * (1.0 - ratio) + config.lr_end * ratio
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_square_error = torch.zeros((), device=torch_device)
        for _ in range(config.steps_per_epoch):
            trajectories_idx = torch.randint(
                states.shape[0],
                (local_batch,),
                generator=generator,
                device=torch_device,
            )
            starts = torch.randint(
                start_count,
                (local_batch,),
                generator=generator,
                device=torch_device,
            )
            frames = starts[:, None] + offsets[None, :]
            batch_idx = trajectories_idx[:, None].expand(-1, h)
            input_window = inputs_norm[batch_idx, frames]
            target_window = targets_norm[batch_idx, frames]
            # Causal prediction at position k sees exactly the deployed prefix
            # of length k + 1. Loss on every position therefore trains startup
            # histories from one frame through the complete context window.
            prediction = train_model(input_window)
            square_error = ((prediction - target_window).square() * weights).sum()
            loss = square_error * world_size / global_weighted_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_square_error += square_error.detach()
        _all_reduce(epoch_square_error)
        mean_loss = float(
            epoch_square_error / (config.steps_per_epoch * global_weighted_count)
        )
        curve.append(mean_loss)
        if rank == 0 and log is not None:
            log(
                f"nerd epoch {epoch + 1}/{config.epochs}: train_delta_mse={mean_loss:.6f}"
            )

    model.eval()
    metadata = {
        **trajectories.metadata,
        "trajectory_count": int(trajectory_count),
        "train_curve": curve,
        "final_train_delta_mse": curve[-1] if curve else float("nan"),
        "distributed_world_size": world_size,
        "global_batch_size": config.batch_size,
        "local_batch_size": local_batch,
        "min_local_batch_size": base_batch,
        "max_local_batch_size": max_local_batch,
        "optimizer_updates": config.epochs * config.steps_per_epoch,
        "state_codec": trajectories.codec.name,
        "model_class": _model_class_name(model),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "active_delta_channels": int(active.sum()),
        "total_delta_channels": int(active.numel()),
    }
    return TrainedNeRDModel(
        model=model,
        normalizers=normalizers.to(torch_device),
        codec=trajectories.codec,
        config=config,
        external_input_shape=trajectories.external_input_shape,
        active_delta_mask=active.to(torch_device),
        metadata=metadata,
        frame_dt=trajectories.frame_dt,
    )


def fit_nerd(
    problem: NeRDProblem,
    *,
    num_trajectories: int,
    steps: int,
    config: NeRDTrainingConfig | None = None,
    dynamics_model: str | torch.nn.Module | Callable[[NeRDModelSpec], torch.nn.Module],
    model_kwargs: dict[str, Any] | None = None,
    loss_weights: torch.Tensor | np.ndarray | None = None,
    device: str | torch.device | None = None,
    seed: int = 0,
    max_abs_state: float | None = None,
    trajectory_filter: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
    log: Callable[[str], None] | None = _nerd_log,
) -> TrainedNeRDModel:
    """Collect teacher trajectories and train a deployable NeRD model.

    This is the recommended entry point for a new Newton problem. The
    architecture is always selected explicitly through ``dynamics_model``. Use
    ``train_nerd`` only when teacher trajectories already exist.
    """
    trajectories = collect_nerd_trajectories(
        problem,
        num_trajectories=num_trajectories,
        steps=steps,
        device=device,
        seed=seed,
        max_abs_state=max_abs_state,
        trajectory_filter=trajectory_filter,
        log=log,
    )
    return train_nerd(
        trajectories,
        config,
        dynamics_model=dynamics_model,
        model_kwargs=model_kwargs,
        loss_weights=loss_weights,
        device=device,
        seed=seed,
        log=log,
    )
