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

"""Run offline neural co-design of an articulated gripper."""

from __future__ import annotations

import argparse
from pathlib import Path

import gripper_workflow
import torch


def parse_args() -> argparse.Namespace:
    """Parse offline gripper co-design options."""
    parser = argparse.ArgumentParser(
        description="PhysicsNeMo offline neural geometry co-design of a Newton gripper"
    )
    output_dir = Path(__file__).parent / "outputs" / "gripper"
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=output_dir / "gripper_offline_dataset.npz",
    )
    parser.add_argument("--regenerate-dataset", action="store_true")
    parser.add_argument("--design-samples", type=int, default=96)
    parser.add_argument("--pose-count", type=int, default=12)
    parser.add_argument("--pose-shortlist", type=int, default=8)
    parser.add_argument("--point-count", type=int, default=192)
    parser.add_argument("--sim-batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--train-batch-size", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--training-patience", type=int, default=80)
    parser.add_argument("--point-features", type=int, default=128)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--training-restarts", type=int, default=3)
    parser.add_argument("--optimization-starts", type=int, default=24)
    parser.add_argument("--optimization-steps", type=int, default=260)
    parser.add_argument("--optimization-lr", type=float, default=0.015)
    parser.add_argument("--optimization-radius", type=float, default=0.02)
    parser.add_argument("--finalists", type=int, default=48)
    parser.add_argument("--sampled-finalists", type=int, default=8)
    parser.add_argument("--sim-time", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=36)
    parser.add_argument("--disturbance-force", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=31)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--newton-device", default=default_device)
    parser.add_argument("--torch-device", default=default_device)
    return parser.parse_args()


if __name__ == "__main__":
    print(gripper_workflow.run(parse_args()))
