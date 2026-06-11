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

"""Scene-visualization helpers shared by the Newton example renderers.

These utilities cover the parts of "render a Newton scene to a video" that every
example renderer repeats:

* framing a Z-up camera on a bounding box (``frame_bounding_box``,
  ``aim_camera``),
* opening Newton's GL viewer headlessly and grabbing frames (``headless_viewer``,
  ``capture_frame``),
* composing panels side by side (``stack_horizontal``), annotating them
  (``draw_text``), and writing an animated GIF (``save_gif``).

They are deliberately scene-agnostic: each example still owns its own scene
construction, what to draw each frame, and any scene-specific overlays. The heavy
optional dependencies (Newton's GL viewer, Warp, and Pillow) are imported lazily
inside the functions that use them, so importing this module never pulls in
Newton or a display stack.

Like the rest of the ``experimental/integrations/newton`` subpackage, these
helpers use prose-style docstrings rather than the NumPy-style
``Parameters``/``Returns`` sections used elsewhere in PhysicsNeMo.
"""

from __future__ import annotations

import importlib
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

__all__ = [
    "aim_camera",
    "capture_frame",
    "draw_text",
    "frame_bounding_box",
    "headless_viewer",
    "save_gif",
    "stack_horizontal",
]

# A frame is either a Pillow image or an ``(H, W, 3|4)`` ``uint8`` array; a
# ``Vec3Like`` is anything ``np.asarray`` turns into a 3-vector. These are
# deliberately loose, unvalidated aliases (not statically enforced types):
# every consumer either calls ``np.asarray(...)`` or routes through
# ``_as_image``, which accept both shapes via duck typing.
Frame = Any
Vec3Like = Any


def _require_pillow(submodule: str = "Image") -> Any:
    """Import and return a ``PIL`` submodule, with install guidance on failure.

    Pillow is an optional dependency (pulled in by the ``newton`` extra), so the
    five image helpers in this module import it lazily through here to surface an
    actionable message instead of a bare ``ModuleNotFoundError: No module named
    'PIL'``.
    """
    try:
        return importlib.import_module(f"PIL.{submodule}")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
        if exc.name is not None and exc.name.split(".")[0] != "PIL":
            raise
        raise RuntimeError(
            "Pillow is required for physicsnemo.experimental.integrations.newton "
            "visualization. Install it with `uv sync --extra newton`, or install "
            "PhysicsNeMo with the Newton extra (for example, `pip install "
            '"nvidia-physicsnemo[cu12,newton]"`).'
        ) from exc


def look_at(position: Vec3Like, target: Vec3Like) -> tuple[float, float]:
    """Pitch/yaw (degrees) so a Z-up camera at ``position`` looks at ``target``.

    Inverts Newton's Z-up ``Camera.get_front``, which builds the view direction as
    ``front = (cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))``. For a desired
    direction ``f`` this gives ``pitch = asin(f_z)`` and ``yaw = atan2(f_y, f_x)``.
    """
    front = np.asarray(target, np.float64) - np.asarray(position, np.float64)
    front /= max(float(np.linalg.norm(front)), 1.0e-9)
    pitch = math.degrees(math.asin(float(np.clip(front[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(front[1]), float(front[0])))
    return pitch, yaw


def frame_bounding_box(
    lo: Vec3Like,
    hi: Vec3Like,
    *,
    azim: float,
    elev: float,
    fov: float = 45.0,
    margin: float = 1.3,
) -> tuple[Any, float, float]:
    """Camera ``(position, pitch, yaw)`` that frames ``[lo, hi]`` from ``azim``/``elev``.

    ``position`` is a ``warp.vec3`` ready to splat into ``viewer.set_camera``.
    ``margin`` > 1 leaves padding around the box; a wider ``fov`` pulls the camera
    in closer. ``azim``/``elev`` are in degrees.
    """
    import warp as wp

    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    center = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo))
    a, e = math.radians(azim), math.radians(elev)
    front = np.array(
        [math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]
    )
    dist = (0.5 * max(diag, 1.0e-3)) / math.tan(math.radians(0.5 * fov)) * margin
    position = center - front * dist
    pitch, yaw = look_at(position, center)
    return wp.vec3(*position.tolist()), pitch, yaw


def aim_camera(
    viewer: Any,
    lo: Vec3Like,
    hi: Vec3Like,
    *,
    azim: float,
    elev: float,
    fov: float = 45.0,
    margin: float = 1.3,
) -> None:
    """Frame ``[lo, hi]`` and apply it to ``viewer`` via ``set_camera``.

    ``set_camera`` must run after ``set_model`` because ``set_model`` rebuilds the
    camera.
    """
    viewer.set_camera(
        *frame_bounding_box(lo, hi, azim=azim, elev=elev, fov=fov, margin=margin)
    )


def headless_viewer(width: int, height: int, *, vsync: bool = False) -> Any:
    """Open Newton's GL viewer in headless mode for offscreen rendering.

    The viewer's compute device is selected by Warp/the model device (e.g.
    ``--newton-device cuda``), not by this function.
    """
    viewer_module = importlib.import_module("newton.viewer")

    return viewer_module.ViewerGL(
        width=width, height=height, headless=True, vsync=vsync
    )


def capture_frame(viewer: Any) -> "PILImage":
    """Grab the current viewer frame as an RGB ``PIL.Image`` (copied off the GPU).

    Newton releases before 1.2.2 can segfault when a CPU viewer captures a frame
    on a host where CUDA is available. For those releases, this helper raises an
    actionable error before entering the unsafe upstream readback path.
    """
    import warp as wp

    Image = _require_pillow("Image")

    device = getattr(viewer, "device", None)
    if (
        _newton_cpu_capture_is_unsafe()
        and device is not None
        and not device.is_cuda
        and wp.is_cuda_available()
    ):
        raise RuntimeError(
            "capture_frame() needs a CUDA viewer device: ViewerGL.get_frame() "
            "is unsafe on a non-CUDA device in Newton releases before 1.2.2 when "
            "CUDA is available. Build the model on CUDA so the viewer runs on CUDA, "
            "e.g. pass --newton-device cuda, or upgrade Newton. "
            f"Got viewer.device={device!r}."
        )

    frame = np.asarray(viewer.get_frame().numpy())
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return Image.fromarray(np.ascontiguousarray(frame.copy()))


def _newton_cpu_capture_is_unsafe() -> bool:
    """Return whether the installed Newton predates safe CPU frame readback."""
    try:
        return Version(version("newton")) < Version("1.2.2")
    except (PackageNotFoundError, InvalidVersion):
        return True


def _as_image(frame: Frame) -> "PILImage":
    Image = _require_pillow("Image")

    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    return frame


def stack_horizontal(
    frames: Sequence[Frame],
    *,
    gap: int = 0,
    background: tuple[int, int, int] = (16, 16, 16),
) -> "PILImage":
    """Compose frames left-to-right, top-aligned, with an optional ``gap`` between.

    Each frame may be a ``PIL.Image`` or an ``(H, W, 3|4)`` ``uint8`` array. The
    canvas height matches the tallest panel; ``gap`` pixels of ``background`` fill
    the seam between panels.
    """
    images = [_as_image(frame).convert("RGB") for frame in frames]
    if not images:
        raise ValueError("stack_horizontal needs at least one frame")

    Image = _require_pillow("Image")

    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def draw_text(
    frame: Frame,
    text: str,
    *,
    xy: tuple[int, int] = (8, 8),
    color: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int, int] | None = None,
    font: Any = None,
) -> "PILImage":
    """Draw a single line of ``text`` on ``frame``, returning a new RGB ``PIL.Image``.

    ``frame`` may be a ``PIL.Image`` or array. Pass ``background`` (an RGBA tuple)
    to draw a translucent box sized to the text behind it.
    """
    ImageDraw = _require_pillow("ImageDraw")

    image = _as_image(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    if background is not None:
        left, top, right, bottom = draw.textbbox(xy, text, font=font)
        draw.rectangle((left - 4, top - 2, right + 4, bottom + 2), fill=background)
    draw.text(xy, text, fill=color, font=font)
    return image


def save_gif(
    frames: Sequence[Frame],
    path: str | Path,
    *,
    fps: float = 24.0,
    palette: bool = False,
    optimize: bool = False,
) -> Path:
    """Write ``frames`` as a looping animated GIF and return the output path.

    Frames may be ``PIL.Image`` objects or arrays and must all share the same
    size. ``palette=True`` quantizes each frame to its own adaptive palette first
    (smaller files, slightly lossy color), which suits flat-shaded scenes; on
    color-rich content (gradients, anti-aliased/shaded renders) the independent
    per-frame palettes can introduce inter-frame color shimmer, so leave
    ``palette=False`` there. ``fps`` must be positive; GIF frame durations are
    integer milliseconds, so the per-frame duration is rounded to the nearest ms.
    """
    if fps <= 0:
        raise ValueError("save_gif needs a positive fps")
    path = Path(path)
    images = [_as_image(frame) for frame in frames]
    if not images:
        raise ValueError("save_gif needs at least one frame")
    expected_size = images[0].size
    for index, image in enumerate(images):
        if image.size != expected_size:
            raise ValueError(
                "save_gif needs all frames to share the same size: frame "
                f"{index} is {image.size}, expected {expected_size}"
            )
    if palette:
        Image = _require_pillow("Image")

        images = [image.convert("P", palette=Image.ADAPTIVE) for image in images]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000.0 / float(fps)),
        loop=0,
        optimize=optimize,
    )
    return path
