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

"""Trained NeRD model, evaluation, checkpointing, and Newton step deployment."""

from __future__ import annotations

import math
import warnings
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from physicsnemo.experimental.integrations.newton.data import (
    _copy_newton_object,
    torch_warp_stream,
)
from physicsnemo.experimental.integrations.newton.distributed import resolve_device
from physicsnemo.experimental.integrations.newton.nerd.checkpoint import (
    _codec_descriptor,
    _codec_from_descriptor,
    _deployment_codec,
)
from physicsnemo.experimental.integrations.newton.nerd.codecs import (
    NeRDJointStateCodec,
    NeRDStateCodec,
)
from physicsnemo.experimental.integrations.newton.nerd.model_builders import (
    _checkpoint_model,
    _CompiledNeRDModel,
    _model_descriptor,
    _model_from_descriptor,
)
from physicsnemo.experimental.integrations.newton.nerd.problem import NeRDDataset
from physicsnemo.experimental.integrations.newton.nerd.runtime import (
    NeRDNormalizers,
    _append_inputs,
    _input_tensor,
    _input_width,
    _model_for_device,
    _predict_delta,
    _runtime_device,
    _state_rmse,
)
from physicsnemo.experimental.integrations.newton.nerd.spec import NeRDTrainingConfig

_CHECKPOINT_VERSION = 2


@dataclass
class TrainedNeRDModel:
    """A trained model plus the Newton codec and metadata needed for deployment."""

    model: torch.nn.Module
    normalizers: NeRDNormalizers
    codec: NeRDStateCodec
    config: NeRDTrainingConfig
    external_input_shape: tuple[int, ...]
    active_delta_mask: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    frame_dt: float | None = None

    def compile_for_inference(
        self,
        *,
        device: str | torch.device | None = None,
        mode: str = "reduce-overhead",
    ) -> TrainedNeRDModel:
        """Return a deployment copy accelerated with ``torch.compile``.

        Compilation is lazy, so the first rollout for each history length pays
        the compilation cost. Run one untimed warmup rollout before measuring
        latency. The returned object remains checkpoint-compatible: :meth:`save`
        serializes the original model rather than the compiled runtime wrapper.

        Parameters
        ----------
        device : str or torch.device, optional
            Inference device. Defaults to the model's current device.
        mode : str, optional
            Compilation mode passed to :func:`torch.compile`.

        Returns
        -------
        TrainedNeRDModel
            Deployment copy that uses a lazily compiled runtime model.
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        torch_device = resolve_device(
            device if device is not None else self.active_delta_mask.device
        )
        model = deepcopy(_checkpoint_model(self.model)).to(torch_device).eval()
        return TrainedNeRDModel(
            model=_CompiledNeRDModel(model, mode=mode),
            normalizers=self.normalizers.to(torch_device),
            codec=self.codec,
            config=self.config,
            external_input_shape=self.external_input_shape,
            active_delta_mask=self.active_delta_mask.to(torch_device),
            metadata={**self.metadata, "inference_compile_mode": mode},
            frame_dt=self.frame_dt,
        )

    @property
    def external_input_dim(self) -> int:
        """Final feature width of each global or per-entity application input."""
        return self.external_input_shape[-1]

    def save(self, path: str | Path) -> None:
        """Save a portable, versioned NeRD state-dict bundle.

        The built-in NeRD models, PhysicsNeMo ``FullyConnected``, and built-in
        codecs reconstruct automatically. Loading a bundle made with another
        model or custom codec requires passing the corresponding object to
        :meth:`load`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_model = _checkpoint_model(self.model)
        metadata = dict(self.metadata)
        metadata.pop("inference_compile_mode", None)
        model_descriptor = _model_descriptor(checkpoint_model)
        if model_descriptor is None:
            warnings.warn(
                "the NeRD model could not be self-described for this checkpoint; "
                "reloading it will require passing model=... to "
                f"{type(self).__name__}.load",
                stacklevel=2,
            )
        torch.save(
            {
                "format": "physicsnemo.experimental.newton.nerd",
                "version": _CHECKPOINT_VERSION,
                "model": model_descriptor,
                "model_state": checkpoint_model.state_dict(),
                "codec": _codec_descriptor(self.codec),
                "codec_signature": self.codec.compatibility_signature(),
                "normalizers": {
                    "input_mean": self.normalizers.input_mean,
                    "input_std": self.normalizers.input_std,
                    "target_mean": self.normalizers.target_mean,
                    "target_std": self.normalizers.target_std,
                },
                "config": asdict(self.config),
                "external_input_shape": self.external_input_shape,
                "active_delta_mask": self.active_delta_mask,
                "metadata": metadata,
                "frame_dt": self.frame_dt,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        model: torch.nn.Module | None = None,
        codec: NeRDStateCodec | None = None,
    ) -> TrainedNeRDModel:
        """Load a bundle saved by :meth:`save`.

        Pass ``model`` or ``codec`` when the saved bundle used a custom object
        that cannot be reconstructed from public PhysicsNeMo/built-in metadata.
        """
        device = resolve_device(device)
        bundle = torch.load(path, map_location=device, weights_only=True)
        if (
            not isinstance(bundle, dict)
            or bundle.get("format") != "physicsnemo.experimental.newton.nerd"
            or bundle.get("version") != _CHECKPOINT_VERSION
        ):
            raise ValueError(f"{path} is not a supported {cls.__name__} checkpoint")
        model = model or _model_from_descriptor(bundle["model"])
        if model is None:
            raise ValueError(
                "checkpoint used a custom Torch model; pass model=... to load it"
            )
        model.load_state_dict(bundle["model_state"])
        model = model.to(device)
        codec = codec or _codec_from_descriptor(bundle["codec"])
        if codec is None:
            raise ValueError(
                "checkpoint used a custom NeRD codec; pass codec=... to load it"
            )
        if codec.compatibility_signature() != tuple(bundle["codec_signature"]):
            raise ValueError("provided codec is incompatible with the saved NeRD codec")
        normalizers = bundle["normalizers"]
        # Filter to known init fields so a future config schema change cannot
        # break **-expansion on an older bundle.
        config_fields = {f.name for f in fields(NeRDTrainingConfig) if f.init}
        config_kwargs = {
            key: value
            for key, value in bundle["config"].items()
            if key in config_fields
        }
        return cls(
            model=model,
            normalizers=NeRDNormalizers(**normalizers),
            codec=codec,
            config=NeRDTrainingConfig(**config_kwargs),
            external_input_shape=tuple(bundle["external_input_shape"]),
            active_delta_mask=bundle["active_delta_mask"],
            metadata=bundle["metadata"],
            frame_dt=bundle["frame_dt"],
        )

    def as_step_model(
        self,
        *,
        newton_model: Any | None = None,
        device: str | torch.device | None = None,
        state_codec: NeRDStateCodec | None = None,
        post_step: Callable[[Any], None] | None = None,
        input_from_step: Callable[[Any, Any, Any, float], Any] | None = None,
        step_mode: str = "replace",
    ) -> NeRDStepModel:
        """Create the deployable learned Newton solver step.

        Pass ``newton_model`` for the common deployment path. The saved codec
        descriptor is rebuilt against that live model, including the body
        refresh required by joint-space deployment. Advanced callers may pass
        an explicit ``state_codec`` instead.

        NeRD is a full solver replacement: it predicts the next state from the
        causal history without reading the solver's output. Before applying the
        codec prediction, it copies ``state_in`` into ``state_out`` so Newton
        fields outside the codec remain current. It therefore only supports
        ``step_mode="replace"`` in ``NewtonEnv``. ``step_mode`` is accepted so a
        deployment can declare and validate its intended mode here; passing
        ``"correct"`` is rejected because the solver output would be discarded.

        A NeRD model is per-frame, so a ``NewtonEnv`` deployment must use
        ``substeps=1`` (the step model is called once per substep with the
        substep ``dt``, while NeRD was trained at ``frame_dt = dt * substeps``;
        with ``substeps>1`` the deployment dt would not match and the call
        raises).

        A model trained with application inputs also requires
        ``input_from_step``. That callback receives live Newton
        ``(state, control, contacts, dt)`` and must return the exact per-world or
        entity-aligned tensor shape used during training. Use
        ``NeRDControlInput`` for a named Newton control field. Custom loops
        that already own the feature tensor can call
        ``NeRDStepModel.step_with_inputs`` directly.
        """
        if step_mode not in ("replace", "correct"):
            raise ValueError(
                "step_mode must be 'replace' or 'correct' to match NewtonEnv"
            )
        if step_mode == "correct":
            raise ValueError(
                "NeRD is a full solver replacement and only supports "
                'step_mode="replace": it predicts the next state from the causal '
                "history and constructs state_out without the solver result, so "
                'a NewtonEnv configured with step_mode="correct" would run the '
                "solver and then silently "
                "discard its output. Deploy with step_mode='replace'."
            )
        if newton_model is not None and state_codec is not None:
            raise ValueError("pass newton_model or state_codec, not both")
        if newton_model is not None:
            state_codec = _deployment_codec(self.codec, newton_model)
            if device is None:
                device = str(newton_model.device)
        return NeRDStepModel(
            self,
            device=device,
            state_codec=state_codec,
            post_step=post_step,
            input_from_step=input_from_step,
        )

    def rollout(
        self,
        initial_state: torch.Tensor | np.ndarray,
        inputs: torch.Tensor | np.ndarray | None = None,
        *,
        steps: int | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Free-run from encoded Newton state without allocating Newton buffers.

        This is the common held-out evaluation path. Specialized evaluators may
        add physical metrics appropriate to joints, bodies, or particles.
        ``inputs`` is required when the model was trained with non-empty
        application inputs.
        """
        torch_device = _runtime_device(device, initial_state, self.active_delta_mask)
        current = torch.as_tensor(
            initial_state, dtype=torch.float32, device=torch_device
        )
        expected = (current.shape[0], *self.codec.state_shape)
        if tuple(current.shape) != expected:
            raise ValueError(
                f"initial_state must have shape {expected}, got {tuple(current.shape)}"
            )
        if inputs is None:
            if steps is None:
                raise ValueError("steps is required for autonomous rollout")
            if _input_width(self.external_input_shape) > 0:
                raise ValueError(
                    "inputs are required because this model was trained with them"
                )
            input_tensor = torch.empty(
                (current.shape[0], steps, *self.external_input_shape),
                dtype=torch.float32,
                device=torch_device,
            )
        else:
            input_tensor = torch.as_tensor(
                inputs, dtype=torch.float32, device=torch_device
            )
            if input_tensor.ndim < 3 or input_tensor.shape[0] != current.shape[0]:
                raise ValueError(
                    "inputs must have shape [trajectories, steps, *external_input_shape]"
                )
            if steps is None:
                steps = int(input_tensor.shape[1])
            expected_inputs = (current.shape[0], steps, *self.external_input_shape)
            if tuple(input_tensor.shape) != expected_inputs:
                raise ValueError(
                    f"inputs must have shape {expected_inputs}, "
                    f"got {tuple(input_tensor.shape)}"
                )
        if steps is None or steps < 0:
            raise ValueError("steps must be non-negative")

        model = _model_for_device(self.model, torch_device).eval()
        normalizers = self.normalizers.to(torch_device)
        active = self.active_delta_mask.to(torch_device)
        history: deque[torch.Tensor] = deque(maxlen=self.config.context_frames)
        states = [current]
        with torch.inference_mode():
            for frame in range(steps):
                token = _append_inputs(
                    self.codec.encode_state(current).unsqueeze(1),
                    input_tensor[:, frame : frame + 1],
                )[:, 0]
                history.append(
                    ((token - normalizers.input_mean) / normalizers.input_std).detach()
                )
                context = torch.stack(tuple(history), dim=1)
                delta = _predict_delta(
                    model, context, self.codec.prediction_shape, normalizers, active
                )
                current = self.codec.delta_to_state(current, delta)
                states.append(current)
        return torch.stack(tuple(states), dim=1)

    def predict_next(
        self,
        state_history: torch.Tensor | np.ndarray,
        input_history: torch.Tensor | np.ndarray | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Predict one next state from a teacher-forced or deployed history."""
        torch_device = _runtime_device(device, state_history, self.active_delta_mask)
        states = torch.as_tensor(
            state_history, dtype=torch.float32, device=torch_device
        )
        expected_tail = self.codec.state_shape
        if (
            states.ndim != len(expected_tail) + 2
            or tuple(states.shape[2:]) != expected_tail
            or states.shape[1] == 0
            or states.shape[1] > self.config.context_frames
        ):
            raise ValueError(
                "state_history must have shape [batch, time, *state_shape] with "
                f"1 <= time <= {self.config.context_frames}; got {tuple(states.shape)}"
            )
        if input_history is None:
            if _input_width(self.external_input_shape) > 0:
                raise ValueError(
                    "input_history is required because this model was trained with inputs"
                )
            inputs = states.new_empty(
                (states.shape[0], states.shape[1], *self.external_input_shape)
            )
        else:
            inputs = torch.as_tensor(
                input_history, dtype=torch.float32, device=torch_device
            )
        expected_inputs = (
            states.shape[0],
            states.shape[1],
            *self.external_input_shape,
        )
        if tuple(inputs.shape) != expected_inputs:
            raise ValueError(
                f"input_history must have shape {expected_inputs}, "
                f"got {tuple(inputs.shape)}"
            )

        model_input = _append_inputs(self.codec.encode_state(states), inputs)
        normalizers = self.normalizers.to(torch_device)
        normalized = (model_input - normalizers.input_mean) / normalizers.input_std
        model = _model_for_device(self.model, torch_device).eval()
        with torch.inference_mode():
            delta = _predict_delta(
                model,
                normalized,
                self.codec.prediction_shape,
                normalizers,
                self.active_delta_mask.to(torch_device),
            )
        return self.codec.delta_to_state(states[:, -1], delta)

    def evaluate(
        self,
        dataset: NeRDDataset,
        *,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        device: str | torch.device | None = None,
    ) -> NeRDRolloutEvaluation:
        """Free-run against held-out Newton trajectories.

        The default raw-state RMSE is suitable only when all channels have
        comparable units. Supply ``metric`` for rigid-body, mixed-unit, or
        task-specific state.
        """
        return evaluate_nerd(self, dataset, metric=metric, device=device)


@dataclass
class NeRDRolloutEvaluation:
    """Generic free-running result against held-out Newton trajectories.

    Attributes
    ----------
    error_by_frame : numpy.ndarray
        Per-frame error averaged over finite trajectories. It has length
        ``steps - 1`` and excludes frame 0, so ``error_by_frame[i]`` aligns with
        ``predictions[:, i + 1]`` and ``truth[:, i + 1]``.
    final_error : float
        Error at the last predicted frame.
    mean_error : float
        Mean error over predicted transitions, excluding frame 0.
    finite_trajectory_fraction : float
        Fraction of trajectories that stayed finite across all frames.
    predictions : torch.Tensor
        Free-running predictions with all ``steps`` frames, including the
        supplied initial state at frame 0.
    """

    error_by_frame: np.ndarray
    final_error: float
    mean_error: float
    finite_trajectory_fraction: float
    predictions: torch.Tensor


def evaluate_nerd(
    trained: TrainedNeRDModel,
    trajectories: NeRDDataset,
    *,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    device: str | torch.device | None = None,
) -> NeRDRolloutEvaluation:
    """Free-run a generic NeRD model against held-out Newton trajectories.

    ``metric`` receives predicted and true states shaped
    ``[trajectories, *state_shape]`` and returns one scalar error per
    trajectory. The default is raw-state RMSE; use a physical metric for mixed
    units, quaternion state, or task-specific quantities.
    """
    if (
        trajectories.codec.compatibility_signature()
        != trained.codec.compatibility_signature()
    ):
        raise ValueError(
            "held-out trajectories use a semantically incompatible NeRD state codec"
        )
    predictions = trained.rollout(
        trajectories.states[:, 0],
        trajectories.inputs,
        device=device,
    )
    truth = trajectories.states.to(predictions.device)
    finite = torch.isfinite(predictions).flatten(start_dim=2).all(dim=2)
    finite_trajectory_fraction = float(finite.all(dim=1).float().mean())
    metric = metric or _state_rmse
    errors = []
    for frame in range(1, truth.shape[1]):
        frame_error = metric(predictions[:, frame], truth[:, frame])
        if frame_error.shape != (truth.shape[0],):
            raise ValueError(
                f"metric must return shape ({truth.shape[0]},), "
                f"got {tuple(frame_error.shape)}"
            )
        # Average over finite trajectories only so a single diverged rollout
        # does not collapse the headline metric to inf; the diverged share is
        # still reported via finite_trajectory_fraction. A frame with no finite
        # trajectory at all is reported as inf.
        frame_finite = finite[:, frame]
        if bool(frame_finite.any()):
            errors.append(frame_error[frame_finite].mean())
        else:
            errors.append(
                torch.full(
                    (), float("inf"), dtype=frame_error.dtype, device=frame_error.device
                )
            )
    error_by_frame = torch.stack(tuple(errors))
    return NeRDRolloutEvaluation(
        error_by_frame=error_by_frame.detach().cpu().numpy(),
        final_error=float(error_by_frame[-1]),
        mean_error=float(error_by_frame.mean()),
        finite_trajectory_fraction=finite_trajectory_fraction,
        predictions=predictions.detach(),
    )


class NeRDStepModel:
    """Run a trained generic NeRD model through Newton's ``NewtonStepModel`` contract."""

    def __init__(
        self,
        trained: TrainedNeRDModel,
        *,
        device: str | torch.device | None = None,
        state_codec: NeRDStateCodec | None = None,
        post_step: Callable[[Any], None] | None = None,
        input_from_step: Callable[[Any, Any, Any, float], Any] | None = None,
    ) -> None:
        self.codec = state_codec or trained.codec
        if (
            self.codec.compatibility_signature()
            != trained.codec.compatibility_signature()
        ):
            raise ValueError(
                "deployment state_codec is semantically incompatible with the trained codec"
            )
        if isinstance(self.codec, NeRDJointStateCodec) and self.codec.model is None:
            raise ValueError(
                "joint-state deployment requires a live state_codec built from the "
                "deployment Newton model"
            )
        self.config = trained.config
        self.external_input_shape = trained.external_input_shape
        self.device = resolve_device(
            device if device is not None else trained.active_delta_mask.device
        )
        self.model = _model_for_device(trained.model, self.device).eval()
        self.normalizers = trained.normalizers.to(self.device)
        self.active_delta_mask = trained.active_delta_mask.to(self.device)
        self.history: deque[torch.Tensor] = deque(maxlen=trained.config.context_frames)
        self.post_step = post_step
        self.input_from_step = input_from_step
        self.frame_dt = trained.frame_dt
        if self.frame_dt is None:
            warnings.warn(
                "NeRD deployment cannot validate the timestep because no training "
                "frame_dt was recorded; a mismatched deployment dt will produce "
                "silently wrong physics. Train from a NeRDDataset/NeRDProblem that "
                "carries frame_dt to enable this check.",
                stacklevel=2,
            )

    def reset(self) -> None:
        """Clear the causal history."""
        self.history.clear()

    def _validate_dt(self, dt: float) -> None:
        """Reject deployment at a frame duration different from training."""
        if self.frame_dt is not None and not math.isclose(
            dt, self.frame_dt, rel_tol=1.0e-6, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"NeRD is a per-frame model trained at frame_dt={self.frame_dt:g}, "
                f"but the deployment called it with dt={dt:g}. A NewtonEnv calls "
                "the step model once per substep with the substep dt, so NeRD must "
                "be deployed with substeps=1 (then dt == frame_dt). If substeps>1, "
                f"frame_dt = dt * substeps and this mismatch is expected; reconfigure "
                "the env with substeps=1."
            )

    def _step(
        self,
        state_in: Any,
        state_out: Any,
        inputs: Any = None,
    ) -> None:
        """Predict and write one learned frame."""
        current = self.codec.read(state_in)
        if current.device != self.device:
            raise ValueError(
                f"NeRD step device {self.device} does not match Newton state "
                f"device {current.device}"
            )
        frame_inputs = _input_tensor(
            inputs,
            self.codec.batch_size,
            self.external_input_shape,
            self.device,
        )
        model_input = _append_inputs(
            self.codec.encode_state(current).unsqueeze(1), frame_inputs.unsqueeze(1)
        )[:, 0]
        norm = self.normalizers
        self.history.append(((model_input - norm.input_mean) / norm.input_std).detach())
        context = torch.stack(tuple(self.history), dim=1)
        with torch.inference_mode():
            delta = _predict_delta(
                self.model,
                context,
                self.codec.prediction_shape,
                norm,
                self.active_delta_mask,
            )
        next_state = self.codec.delta_to_state(current, delta)
        with torch_warp_stream(self.device):
            _copy_newton_object(state_out, state_in)
            self.codec.write(state_out, next_state)
            self.codec.finalize_state(state_out)
            if self.post_step is not None:
                self.post_step(state_out)

    def step_with_inputs(
        self,
        state_in: Any,
        state_out: Any,
        inputs: Any,
        *,
        dt: float,
    ) -> None:
        """Advance one learned frame from an already prepared feature tensor.

        Use this method in custom simulation loops and benchmarks that already
        own the exact per-world features used during training. A
        ``NewtonEnv`` should call the object normally through ``__call__``, with
        ``input_from_step`` configured to extract those
        features from Newton state, control, or contacts.
        """
        self._validate_dt(dt)
        self._step(state_in, state_out, inputs)

    def __call__(
        self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float
    ) -> None:
        """Advance one frame through the Newton ``NewtonStepModel`` interface."""
        self._validate_dt(dt)
        if _input_width(self.external_input_shape) == 0:
            inputs = None
        elif self.input_from_step is None:
            raise ValueError(
                "input_from_step is required for a NeRD model trained with inputs"
            )
        else:
            inputs = self.input_from_step(state_in, control, contacts, dt)
        self._step(state_in, state_out, inputs)


# Backward-compatible private alias for internal imports predating the public
# deployment type.
_NeRDStepModel = NeRDStepModel
