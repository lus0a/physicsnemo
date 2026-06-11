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

"""Cartpole scene and NeRD problem shared by training and rendering."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import numpy as np
import torch

from physicsnemo.experimental.integrations.newton import (
    NeRDControlInput,
    NeRDProblem,
    NeRDTrainingConfig,
    NewtonEnv,
    field_to_torch,
)
from physicsnemo.experimental.integrations.newton.nerd import NeRDJointStateCodec

CHECKPOINT_BASE_ERR = {100: 0.0002, 500: 0.0042, 1000: 0.0318}
CHECKPOINT_JOINT_ERR = {100: 0.0004, 500: 0.0149, 1000: 0.0761}

MODEL_LABELS = {
    "nerd-transformer": "NeRDTransformer scaled short run",
    "fully-connected": "PhysicsNeMo FullyConnected model-swap run",
}


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angular differences to ``[-pi, pi]``."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class CartpoleScene:
    """Replicated reference NeRD cartpole on Newton Featherstone."""

    def __init__(self, num_worlds: int, device: str) -> None:
        self.fps = 60
        self.dt = 1.0 / self.fps
        self.substeps = 5
        self.sim_dt = self.dt / self.substeps
        self.num_worlds = num_worlds

        cartpole = newton.ModelBuilder()
        cartpole.default_joint_cfg.armature = 0.01
        cartpole.default_joint_cfg.limit_ke = 1.0e4
        cartpole.default_joint_cfg.limit_kd = 1.0e1
        cartpole.add_urdf(
            str(Path(__file__).resolve().parent / "assets" / "cartpole_nerd.urdf"),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        builder = newton.ModelBuilder()
        builder.replicate(cartpole, num_worlds, spacing=(0.0, 2.0, 0.0))
        self.model = builder.finalize(device=device)
        self.solver = newton.solvers.SolverFeatherstone(
            self.model, update_mass_matrix_interval=self.substeps
        )
        self.contacts = None


def sample_initial_state(cart_band: float = 1.0, velocity: float = 1.0):
    """Return a sampler for randomized cartpole initial states."""

    def sample(
        rng: np.random.Generator, world_count: int, layout: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros((world_count, layout.dof_q), dtype=np.float32)
        q[:, 0] = rng.uniform(-cart_band, cart_band, world_count)
        q[:, 1] = rng.uniform(-np.pi, np.pi, world_count)
        qd = rng.uniform(-velocity, velocity, (world_count, layout.dof_qd)).astype(
            np.float32
        )
        return q, qd

    return sample


def sample_force(scale: float):
    """Return a sampler for random per-frame cart forces."""

    def sample(
        rng: np.random.Generator, world_count: int, _frame: int, layout: Any
    ) -> np.ndarray:
        force = np.zeros((world_count, layout.dof_qd), dtype=np.float32)
        force[:, 0] = rng.uniform(-scale, scale, world_count)
        return force

    return sample


def training_config(args: argparse.Namespace) -> NeRDTrainingConfig:
    """Build the shared NeRD training configuration."""
    return NeRDTrainingConfig(
        context_frames=args.context_frames,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )


def model_selection(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Return the selected PhysicsNeMo model and architecture arguments."""
    if args.model == "fully-connected":
        return "FullyConnected", {
            "layer_size": args.n_embd,
            "num_layers": args.n_layer,
            "activation_fn": "silu",
            "skip_connections": True,
        }
    return "NeRDTransformer", {
        "block_size": max(32, args.context_frames),
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
    }


def artifact_stem(args: argparse.Namespace) -> str:
    """Return the output stem for the selected model."""
    if args.model == "nerd-transformer":
        return "cartpole_nerd"
    return f"cartpole_{args.model.replace('-', '_')}_nerd"


def evaluation_initial_state(
    layout: Any, seed: int, cart_band: float, velocity: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample deterministic held-out passive-release states."""
    q, qd = sample_initial_state(cart_band, velocity)(
        np.random.default_rng(seed), layout.world_count, layout
    )
    return torch.from_numpy(q), torch.from_numpy(qd)


def make_problem(
    scene: CartpoleScene,
    *,
    cart_band: float = 1.0,
    init_velocity: float = 1.0,
    force_scale: float = 1500.0,
    fixed_init: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> NeRDProblem:
    """Build the Cartpole-specific reset and force policy."""
    codec = NeRDJointStateCodec(scene.model, robot_centric=False)
    control_input = NeRDControlInput(
        "joint_f",
        per_world_shape=(codec.layout.dof_qd,),
    )
    env = NewtonEnv.from_model(
        scene.model,
        solver=scene.solver,
        dt=scene.sim_dt,
        substeps=scene.substeps,
        collisions=False,
    )

    def reset(environment: NewtonEnv, rng: np.random.Generator) -> None:
        q, qd = (
            sample_initial_state(cart_band, init_velocity)(
                rng, codec.batch_size, codec.layout
            )
            if fixed_init is None
            else fixed_init
        )
        q_target = field_to_torch(environment.state.joint_q)
        qd_target = field_to_torch(environment.state.joint_qd)
        q_target.copy_(torch.as_tensor(q, device=q_target.device).reshape(-1))
        qd_target.copy_(torch.as_tensor(qd, device=qd_target.device).reshape(-1))

    def inputs(rng: np.random.Generator, batch_size: int, frame: int) -> np.ndarray:
        if fixed_init is not None:
            return np.zeros((batch_size, codec.layout.dof_qd), dtype=np.float32)
        return sample_force(force_scale)(rng, batch_size, frame, codec.layout)

    return NeRDProblem.from_env(
        env,
        state_codec=codec,
        randomize=reset,
        sample_inputs=inputs,
        apply_inputs=control_input.apply,
        name="Cartpole",
    )


@dataclass
class CartpoleEvaluation:
    """Passive-motion accuracy and runtime measurements."""

    horizons: tuple[int, ...]
    free_running_base_err: dict[int, float]
    free_running_joint_err: dict[int, float]
    teacher_forced_base_err: float
    teacher_forced_joint_err: float
    jerk_ratio: float
    max_cart: float
    max_pole_speed: float
    finite_fraction: float
    teacher_step_ms: float
    nerd_step_ms: float
    overlay: dict[str, np.ndarray]

    @property
    def speedup(self) -> float:
        """Return the analytical-to-learned step-time ratio."""
        return self.teacher_step_ms / max(self.nerd_step_ms, 1.0e-9)


def _sync(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def _cartpole_error(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    base = (prediction[..., 0] - truth[..., 0]).abs()
    joint = _wrap_angle(prediction[..., 1] - truth[..., 1]).abs()
    return base, joint


def _jerk(q: torch.Tensor) -> float:
    d1 = _wrap_angle(q[:, 1:, 1] - q[:, :-1, 1])
    d2 = d1[:, 1:] - d1[:, :-1]
    d3 = d2[:, 1:] - d2[:, :-1]
    return float(torch.sqrt(d3.square().mean()))


def evaluate(
    trained: Any,
    scene: CartpoleScene,
    *,
    init_q: torch.Tensor,
    init_qd: torch.Tensor,
    horizons: tuple[int, ...],
    device: str,
) -> CartpoleEvaluation:
    """Evaluate free-running and teacher-forced Cartpole accuracy."""
    torch_device = torch.device(device)
    problem = make_problem(scene, fixed_init=(init_q, init_qd))
    inputs = torch.zeros(
        problem.batch_size,
        max(horizons),
        problem.codec.layout.dof_qd,
        device=torch_device,
    )
    warmup_frames = min(3, max(horizons))

    problem.reset(np.random.default_rng(0))
    for frame in range(warmup_frames):
        problem.advance(inputs[:, frame], frame)
    _sync(torch_device)

    problem.reset(np.random.default_rng(0))
    states = [problem.codec.read(problem.get_state()).to(torch_device)]
    _sync(torch_device)
    start = time.perf_counter()
    for frame in range(max(horizons)):
        problem.advance(inputs[:, frame], frame)
        states.append(problem.codec.read(problem.get_state()).to(torch_device))
    _sync(torch_device)
    teacher_step_ms = (time.perf_counter() - start) * 1000.0 / max(horizons)
    teacher = torch.stack(states, dim=1)

    with torch.inference_mode():
        trained.rollout(teacher[:, 0], inputs[:, :warmup_frames], device=device)
    _sync(torch_device)
    start = time.perf_counter()
    prediction = trained.rollout(teacher[:, 0], inputs, device=device)
    _sync(torch_device)
    nerd_step_ms = (time.perf_counter() - start) * 1000.0 / max(horizons)

    layout = problem.codec.layout
    teacher_q = teacher[..., : layout.dof_q]
    prediction_q = prediction[..., : layout.dof_q]
    finite = torch.isfinite(prediction).flatten(start_dim=2).all(dim=2)

    free_base: dict[int, float] = {}
    free_joint: dict[int, float] = {}
    for horizon in horizons:
        base, joint = _cartpole_error(
            prediction_q[:, 1 : horizon + 1], teacher_q[:, 1 : horizon + 1]
        )
        free_base[horizon] = float(base.mean())
        free_joint[horizon] = float(joint.mean())

    history = trained.config.context_frames
    forced_base = []
    forced_joint = []
    for frame in range(history - 1, teacher.shape[1] - 1):
        next_state = trained.predict_next(
            teacher[:, frame - history + 1 : frame + 1],
            inputs[:, frame - history + 1 : frame + 1],
            device=device,
        )
        base, joint = _cartpole_error(
            next_state[..., : layout.dof_q], teacher_q[:, frame + 1]
        )
        forced_base.append(base.mean())
        forced_joint.append(joint.mean())

    per_world = _wrap_angle(prediction_q[..., 1] - teacher_q[..., 1]).abs().mean(dim=1)
    representative = int(torch.argsort(per_world)[len(per_world) // 2])
    return CartpoleEvaluation(
        horizons=horizons,
        free_running_base_err=free_base,
        free_running_joint_err=free_joint,
        teacher_forced_base_err=float(torch.stack(forced_base).mean()),
        teacher_forced_joint_err=float(torch.stack(forced_joint).mean()),
        jerk_ratio=_jerk(prediction_q) / max(_jerk(teacher_q), 1.0e-9),
        max_cart=float(prediction_q[..., 0].abs().max()),
        max_pole_speed=float(prediction[..., layout.dof_q + 1].abs().max()),
        finite_fraction=float(finite.all(dim=1).float().mean()),
        teacher_step_ms=teacher_step_ms,
        nerd_step_ms=nerd_step_ms,
        overlay={
            "teacher_q": teacher_q[representative].cpu().numpy(),
            "nerd_q": prediction_q[representative].detach().cpu().numpy(),
        },
    )
