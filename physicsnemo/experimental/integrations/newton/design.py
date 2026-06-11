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

"""Surrogate-guided optimization with Newton as a black-box evaluator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.experimental.integrations.newton.distributed import resolve_device
from physicsnemo.utils.logging import PythonLogger

EvaluateDesigns = Callable[[np.ndarray], np.ndarray]


def _validate_score_prediction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "design surrogate model must map (batch, design_dim) -> "
            f"(batch, 1); expected {tuple(target.shape)}, "
            f"got {tuple(prediction.shape)}"
        )


@dataclass
class _DesignSurrogateMeta(ModelMetaData):
    auto_grad: bool = True


class DesignSurrogate(Module):
    """A normalized-design-to-score model with a small fit loop.

    Inputs are design vectors scaled to the unit cube ``[0, 1]``. The
    application maps those vectors to physical dimensions before running
    Newton.
    """

    def __init__(
        self, model: nn.Module, *, device: str | torch.device | None = None
    ) -> None:
        super().__init__(meta=_DesignSurrogateMeta())
        self.model = model
        self.register_buffer("target_mean", torch.zeros(1, 1))
        self.register_buffer("target_std", torch.ones(1, 1))
        self.register_buffer("_is_fitted", torch.tensor(False, dtype=torch.bool))
        self.to(resolve_device(device))

    @classmethod
    def mlp(
        cls,
        design_dim: int,
        *,
        hidden_dim: int = 128,
        depth: int = 4,
        device: str | torch.device | None = None,
    ) -> DesignSurrogate:
        """Build the default PhysicsNeMo fully connected score model.

        ``hidden_dim`` and ``depth`` map to
        ``physicsnemo.models.mlp.FullyConnected``'s ``layer_size`` and
        ``num_layers`` respectively.
        """
        from physicsnemo.models.mlp import FullyConnected

        if design_dim <= 0:
            raise ValueError("design_dim must be positive")
        return cls(
            FullyConnected(
                in_features=design_dim,
                out_features=1,
                layer_size=hidden_dim,
                num_layers=max(1, depth),
                activation_fn="silu",
            ),
            device=device,
        )

    def forward(self, normalized_designs: torch.Tensor) -> torch.Tensor:
        return self.model(normalized_designs) * self.target_std + self.target_mean

    def fit(
        self,
        normalized_designs: np.ndarray,
        scores: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 3.0e-3,
    ) -> float:
        """Fit evaluated normalized designs and return final score-space MSE.

        The returned MSE is computed in eval mode under the final weights, so it
        matches what :meth:`predict` would produce. Each call freshly constructs
        the optimizer (no Adam-moment carry-over) but leaves the model weights
        untouched, so repeated calls warm-start from the previous fit.
        """
        normalized_designs = _normalized_design_array(normalized_designs)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape != (normalized_designs.shape[0],):
            raise ValueError("scores must contain one value per normalized design")
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite")
        if epochs <= 0 or lr <= 0.0:
            raise ValueError("epochs and lr must be positive")

        x = torch.as_tensor(normalized_designs, device=self.device)
        y = torch.as_tensor(scores, device=self.device).reshape(-1, 1)
        mean = y.mean().reshape(1, 1)
        std = y.std(unbiased=False).clamp_min(1.0e-6).reshape(1, 1)
        target = (y - mean) / std
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1.0e-5
        )
        self._is_fitted.fill_(False)
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = self.model(x)
            try:
                _validate_score_prediction(prediction, target)
            except ValueError:
                self.eval()
                raise
            if epoch == 0:
                self.target_mean.copy_(mean)
                self.target_std.copy_(std)
            loss = nn.functional.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
        self.eval()
        # Report the error under the final, eval-mode weights that predict()
        # uses, rather than the train-mode loss from before the last step.
        with torch.no_grad():
            prediction = self.model(x)
            _validate_score_prediction(prediction, target)
            final_mse = nn.functional.mse_loss(prediction, target)
        self._is_fitted.fill_(True)
        return float(final_mse * std.square().squeeze())

    @torch.no_grad()
    def predict(self, normalized_designs: np.ndarray) -> np.ndarray:
        """Predict one score for each normalized design."""
        if not bool(self._is_fitted):
            raise RuntimeError("fit the design surrogate before prediction")
        self.eval()
        x = torch.as_tensor(
            _normalized_design_array(normalized_designs),
            dtype=torch.float32,
            device=self.device,
        )
        return self(x).reshape(-1).cpu().numpy()


@dataclass
class DesignResult:
    """Evaluated designs and the fitted surrogate returned by :func:`optimize_design`.

    The per-design objective is named ``scores`` because Newton is treated here
    as an opaque black-box evaluator. The differentiable per-candidate optimizers
    in this package instead minimize a *loss* (see :class:`GroupedDesignResult`);
    see :func:`optimize_design` for the score-vs-loss convention.
    """

    normalized_designs: np.ndarray
    scores: np.ndarray
    history: list[float]
    surrogate: DesignSurrogate

    @property
    def best_index(self) -> int:
        """Index of the lowest measured score."""
        return int(np.argmin(self.scores))

    @property
    def best_design(self) -> np.ndarray:
        """Lowest-scoring evaluated design in normalized ``[0, 1]`` coordinates."""
        return self.normalized_designs[self.best_index]

    @property
    def best_score(self) -> float:
        """Lowest measured score."""
        return float(self.scores[self.best_index])


def optimize_design(
    evaluate: EvaluateDesigns,
    *,
    design_dim: int,
    rounds: int = 6,
    batch_size: int = 8,
    initial_samples: int = 16,
    surrogate: DesignSurrogate | None = None,
    surrogate_epochs: int = 200,
    candidate_pool_size: int = 2048,
    seed: int = 0,
    device: str | torch.device | None = None,
    log: Callable[[str], None] | None = PythonLogger("design").info,
) -> DesignResult:
    """Optimize a unit-cube design using Sobol coverage and surrogate proposals.

    ``evaluate`` is the application-owned Newton oracle. It receives
    ``(batch, design_dim)`` normalized designs in ``[0, 1]`` and returns one
    scalar score to minimize per row. The application maps each vector to
    physical bounds before running Newton. The initial sample and every guided
    proposal are re-evaluated by Newton; the surrogate only decides which
    designs should receive that expensive evaluation.

    ``rounds`` is the number of outer surrogate-guided re-evaluation batches, a
    distinct quantity from the per-update gradient-iteration counts used by the
    gradient-descent optimizers in this package (e.g. ``optimization_steps``);
    each round here runs ``surrogate_epochs`` internal training epochs.

    The returned objective is called a *score* because Newton is treated here as
    an opaque black-box evaluator; the differentiable per-candidate optimizers in
    this package instead minimize a *loss*.

    ``device`` only takes effect when this function constructs the default
    surrogate. When a ``surrogate`` is supplied it is ignored, and the supplied
    surrogate's own device is used; set the device on the surrogate before
    passing it in.
    """
    if design_dim <= 0:
        raise ValueError("design_dim must be positive")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if batch_size <= 0 or initial_samples <= 0 or surrogate_epochs <= 0:
        raise ValueError(
            "batch_size, initial_samples, and surrogate_epochs must be positive"
        )
    if candidate_pool_size < batch_size:
        raise ValueError("candidate_pool_size must be at least batch_size")

    surrogate = surrogate or DesignSurrogate.mlp(design_dim, device=device)
    normalized_designs = (
        torch.quasirandom.SobolEngine(design_dim, scramble=True, seed=seed)
        .draw(initial_samples)
        .numpy()
    )
    scores = _evaluate_in_batches(evaluate, normalized_designs, batch_size)
    history = [float(scores.min())]
    surrogate.fit(normalized_designs, scores, epochs=surrogate_epochs)
    if log:
        log(
            f"initial sample: evaluated {len(normalized_designs)}, "
            f"best score {history[-1]:.4f}"
        )

    for round_index in range(rounds):
        candidates = (
            torch.quasirandom.SobolEngine(
                design_dim, scramble=True, seed=seed + round_index + 1
            )
            .draw(candidate_pool_size)
            .numpy()
        )
        order = np.argsort(surrogate.predict(candidates))
        explore = int(round(batch_size * 0.25))
        exploit_indices = order[: batch_size - explore]
        remaining = order[batch_size - explore :]
        explore_indices = (
            np.random.default_rng(seed + round_index + 1).choice(
                remaining, size=explore, replace=False
            )
            if explore
            else np.empty(0, dtype=np.int64)
        )
        proposed = candidates[np.concatenate((exploit_indices, explore_indices))]
        proposed_scores = _evaluate_in_batches(evaluate, proposed, batch_size)
        normalized_designs = np.concatenate((normalized_designs, proposed), axis=0)
        scores = np.concatenate((scores, proposed_scores), axis=0)
        surrogate.fit(normalized_designs, scores, epochs=surrogate_epochs)
        history.append(float(scores.min()))
        if log:
            log(
                f"round {round_index + 1}/{rounds}: evaluated {len(normalized_designs)}, "
                f"best score {history[-1]:.4f}"
            )

    return DesignResult(
        normalized_designs=normalized_designs,
        scores=scores,
        history=history,
        surrogate=surrogate,
    )


def _evaluate_in_batches(
    evaluate: EvaluateDesigns, normalized_designs: np.ndarray, batch_size: int
) -> np.ndarray:
    scores = []
    for start in range(0, len(normalized_designs), batch_size):
        batch = normalized_designs[start : start + batch_size]
        values = np.asarray(evaluate(batch), dtype=np.float32).reshape(-1)
        if values.shape != (len(batch),):
            raise ValueError(
                f"evaluate must return one score per design; got {values.shape} "
                f"for a batch of {len(batch)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("evaluate returned a non-finite score")
        scores.append(values)
    return np.concatenate(scores)


def _normalized_design_array(value: np.ndarray) -> np.ndarray:
    designs = np.asarray(value, dtype=np.float32)
    if designs.ndim != 2 or designs.shape[0] == 0 or designs.shape[1] == 0:
        raise ValueError("normalized_designs must have shape (samples, design_dim)")
    if not np.isfinite(designs).all():
        raise ValueError("normalized_designs must be finite")
    if (designs < 0.0).any() or (designs > 1.0).any():
        raise ValueError("normalized_designs must lie in [0, 1]")
    return designs
