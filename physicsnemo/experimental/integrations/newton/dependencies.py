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

"""Lazy import for Newton, the one optional integration dependency.

Warp is a hard PhysicsNeMo dependency (``warp-lang`` in ``pyproject.toml``, and
``physicsnemo`` itself imports it), so modules here ``import warp`` directly.
Newton is optional, so it is imported on demand through :func:`require_newton`
to keep ``import physicsnemo.experimental.integrations.newton`` working without Newton."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from physicsnemo.core.version_check import is_package_available


def require_newton() -> Any:
    """Import and return Newton, raising a helpful error if it is unavailable."""

    # Use the central availability check (MOD-011) instead of a bare
    # try/except for detection, but keep the integration-specific install
    # hint. Fall back to find_spec so an editable/metadata-less install is
    # still recognized, mirroring physicsnemo.core.version_check.OptionalImport.
    available = is_package_available("newton") or (
        importlib.util.find_spec("newton") is not None
    )
    if not available:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "Newton is required for physicsnemo.experimental.integrations.newton runtime use. "
            "Install it with `uv sync --extra newton`, or install PhysicsNeMo "
            "with the Newton extra plus the CUDA backend matching your system "
            '(for example, `pip install "nvidia-physicsnemo[cu12,newton]"`).'
        )
    return importlib.import_module("newton")
