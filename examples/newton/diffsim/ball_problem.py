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

"""Problem definition shared by the differentiable-ball example tools."""

from __future__ import annotations

import argparse
from typing import Any, Callable

import numpy as np
import torch
import warp as wp
from newton.examples.diffsim.example_diffsim_ball import Example as BallScene
from newton.examples.diffsim.example_diffsim_ball import loss_kernel

from physicsnemo.experimental.integrations.newton import (
    BPTTSurrogate,
    NewtonEnv,
    TeacherSample,
    collect_teacher_batch,
    differentiable_rollout,
    field_to_torch,
    load_example_scene,
    resolve_device,
)

LAUNCH_MEAN = np.array([0.0, 5.0, -5.0], dtype=np.float32)
LAUNCH_STD = np.array([1.0, 2.0, 2.0], dtype=np.float32)
LAUNCH_Y_CLIP = (1.0, 9.0)
LAUNCH_Z_CLIP = (-8.0, 2.0)

BPTT_LR = 0.08
NEWTON_REFINE_STEPS = 19
NEWTON_REFINE_LR = 0.05


def observe_ball(state: Any) -> torch.Tensor:
    """Return the ball position and velocity."""
    return torch.cat(
        (field_to_torch(state.particle_q)[0], field_to_torch(state.particle_qd)[0])
    )


def make_env(args: argparse.Namespace) -> NewtonEnv:
    """Build and wrap the differentiable Newton ball scene."""
    scene = load_example_scene(
        BallScene,
        device=str(resolve_device(args.newton_device)),
        substeps=args.substeps,
    )
    return NewtonEnv.from_scene(
        scene,
        observe=observe_ball,
        requires_grad=True,
        collide_on_reset=True,
        # The stock scene authors its differentiable plane contact once.
        collide_each_substep=False,
    )


def feasible_launch(rng: np.random.Generator) -> np.ndarray:
    """Draw one launch velocity from the feasible training distribution."""
    launch = LAUNCH_MEAN + rng.normal(0.0, LAUNCH_STD).astype(np.float32)
    launch[1] = np.clip(launch[1], *LAUNCH_Y_CLIP)
    launch[2] = np.clip(launch[2], *LAUNCH_Z_CLIP)
    return launch.astype(np.float32)


def feasible_sobol_launches(count: int, *, seed: int) -> np.ndarray:
    """Draw reproducible quasi-random launch candidates.

    The first candidate is always :data:`LAUNCH_MEAN`.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=int(seed))
    unit = engine.draw(count).clamp_(1.0e-6, 1.0 - 1.0e-6)
    normal = torch.erfinv(2.0 * unit - 1.0) * np.sqrt(2.0)
    starts = LAUNCH_MEAN + normal.numpy().astype(np.float32) * LAUNCH_STD
    starts[:, 1] = np.clip(starts[:, 1], *LAUNCH_Y_CLIP)
    starts[:, 2] = np.clip(starts[:, 2], *LAUNCH_Z_CLIP)
    starts[0] = LAUNCH_MEAN
    return starts.astype(np.float32)


def ball_loss_fn(env: NewtonEnv, target: np.ndarray) -> Callable[[list[Any]], Any]:
    """Return the final squared-distance objective for ``target``."""
    target_vec = wp.vec3(*np.asarray(target, np.float32).tolist())

    def loss_fn(states: list[Any]) -> Any:
        loss = wp.zeros(
            1, dtype=wp.float32, requires_grad=True, device=env.model.device
        )
        wp.launch(
            loss_kernel,
            dim=1,
            inputs=[states[-1].particle_q, target_vec, loss],
            device=env.model.device,
        )
        return loss

    return loss_fn


def newton_final_position(env: NewtonEnv, launch: np.ndarray, steps: int) -> np.ndarray:
    """Roll out ``launch`` and return the final Newton ball position."""
    env.reset(particle_qd=np.asarray(launch, np.float32).reshape(1, 3))
    final = field_to_torch(env.rollout(steps).final_state.particle_q)[0]
    return final.detach().cpu().numpy().astype(np.float32)


def reachable_target(
    env: NewtonEnv, rng: np.random.Generator, steps: int
) -> np.ndarray:
    """Sample a target reached by a feasible Newton launch."""
    return newton_final_position(env, feasible_launch(rng), steps)


def teacher_batch(
    env: NewtonEnv,
    count: int,
    args: argparse.Namespace,
    *,
    seed: int,
):
    """Collect target-conditioned differentiable Newton rollouts."""
    rng = np.random.default_rng(seed)

    def sample(_index: int) -> TeacherSample:
        launch = feasible_launch(rng)
        target = reachable_target(env, rng, args.steps)
        env.reset(particle_qd=launch.reshape(1, 3))
        rollout = differentiable_rollout(
            env,
            steps=args.steps,
            loss_fn=ball_loss_fn(env, target),
            field="particle_qd",
        )
        return TeacherSample(
            states=rollout.observations,
            parameters=launch,
            adjoints=rollout.adjoint[0],
            loss=rollout.loss,
            task_data={"target": target},
        )

    return collect_teacher_batch(count, sample)


def surrogate_inputs(params: torch.Tensor, batch: Any):
    """Build initial-state and target inputs for the surrogate."""
    target = batch.task_data["target"].to(params)
    start = batch.states[:, 0, :3].to(params)
    return torch.cat((start, params), dim=-1), target[:, None, :]


def surrogate_task_loss(
    prediction: torch.Tensor, _params: torch.Tensor, batch: Any
) -> torch.Tensor:
    """Return final squared-distance loss for the conditioned target."""
    target = batch.task_data["target"].to(prediction)
    return ((prediction[:, -1, :3] - target) ** 2).sum(-1)


def build_surrogate(env: NewtonEnv, args: argparse.Namespace):
    """Collect data and train the target-conditioned surrogate."""
    train = teacher_batch(env, args.samples, args, seed=args.seed)
    held_out = teacher_batch(env, args.val_samples, args, seed=args.seed + 100_000)
    surrogate = BPTTSurrogate(
        state_dim=6,
        param_dim=3,
        input_dim=3,
        to_inputs=surrogate_inputs,
        task_loss=surrogate_task_loss,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        device=resolve_device(args.torch_device),
    )
    fit = surrogate.fit(train, epochs=args.epochs)
    return surrogate, train, held_out, fit, surrogate.evaluate(held_out)


def single_target_batch(held_out: Any, target: np.ndarray):
    """Build a one-sample optimization batch for ``target``."""
    batch = held_out.head(1)
    batch.parameters = torch.as_tensor(LAUNCH_MEAN, dtype=torch.float32).reshape(1, 3)
    batch.task_data["target"] = torch.as_tensor(target, dtype=torch.float32).reshape(
        1, 3
    )
    return batch


def heldout_generalization(
    env: NewtonEnv,
    surrogate: BPTTSurrogate,
    held_out: Any,
    args: argparse.Namespace,
):
    """Optimize and validate launches for independently sampled targets."""
    rng = np.random.default_rng(args.seed + 500_000)
    cosines: list[float] = []
    misses: list[float] = []
    per_target: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    for index in range(args.heldout_targets):
        target = reachable_target(env, rng, args.steps)
        batch = single_target_batch(held_out, target)
        starts = feasible_sobol_launches(
            args.opt_samples, seed=args.seed + 600_000 + index
        )
        plan = surrogate.optimize_multistart(
            batch,
            starts=args.opt_samples,
            initial_params=starts,
            steps=args.opt_steps,
            lr=BPTT_LR,
            seed=args.seed + 600_000 + index,
        )
        best = np.asarray(plan["best_params"], np.float32).reshape(-1)
        miss = float(
            np.linalg.norm(newton_final_position(env, best, args.steps) - target)
        )
        cosine = target_gradient_cosine(
            env, surrogate, batch, LAUNCH_MEAN, target, args
        )
        cosines.append(cosine)
        misses.append(miss)
        plans.append(plan)
        per_target.append(
            {
                "index": index,
                "target": target.tolist(),
                "real_miss": miss,
                "gradient_cosine": cosine,
                "best_params": best.tolist(),
                "surrogate_initial_task_loss": float(plan["initial_task_loss"]),
                "surrogate_best_task_loss": float(plan["best_task_loss"]),
            }
        )

    return (
        {
            "heldout_targets": int(args.heldout_targets),
            "mean_real_miss": float(np.mean(misses)),
            "mean_cosine": float(np.mean(cosines)),
            "min_cosine": float(np.min(cosines)),
            "misses": misses,
            "cosines": cosines,
        },
        per_target,
        plans,
    )


def target_gradient_cosine(
    env: NewtonEnv,
    surrogate: BPTTSurrogate,
    batch: Any,
    launch: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
) -> float:
    """Compare surrogate and Newton gradients with respect to ``launch``."""
    env.reset(particle_qd=np.asarray(launch, np.float32).reshape(1, 3))
    rollout = differentiable_rollout(
        env,
        steps=args.steps,
        loss_fn=ball_loss_fn(env, target),
        field="particle_qd",
    )
    newton_gradient = rollout.adjoint[0].reshape(-1).float()

    one = batch.head(1).to(surrogate.device)
    params = (
        torch.as_tensor(launch, dtype=torch.float32)
        .reshape(1, 3)
        .to(surrogate.device)
        .requires_grad_(True)
    )
    prediction = surrogate.rollout(params, one)
    task = surrogate_task_loss(prediction, params, one).sum()
    surrogate_gradient = torch.autograd.grad(task, params)[0].reshape(-1).float().cpu()
    cosine = torch.nn.functional.cosine_similarity(
        surrogate_gradient, newton_gradient.cpu(), dim=-1, eps=1.0e-8
    )
    return float(cosine)


def newton_loss_for_target(
    env: NewtonEnv, target: np.ndarray, args: argparse.Namespace
) -> Callable[[np.ndarray], float]:
    """Return a forward-only Newton objective for ``target``."""

    def loss(params: np.ndarray) -> float:
        final = newton_final_position(env, np.asarray(params, np.float32), args.steps)
        return float(((final - target) ** 2).sum())

    return loss
