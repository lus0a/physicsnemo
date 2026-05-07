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

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

wp.init()
wp.config.quiet = True


def _normalize_torch(vector: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def _as_vec3(
    value: torch.Tensor | Sequence[float], *, name: str, device
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        value = torch.tensor(value, device=device, dtype=torch.float32)
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {tuple(value.shape)}")
    return value


def _optional_tensor_arg(value: torch.Tensor | Sequence[float] | None, *, device):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    return torch.as_tensor(value, device=device)


def _camera_basis(
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    *,
    device,
) -> torch.Tensor:
    eye = _as_vec3(eye, name="eye", device=device)
    center = _as_vec3(center, name="center", device=device)
    up = _as_vec3(up, name="up", device=device)
    forward_raw = center - eye
    if forward_raw.device.type == "cpu" and bool(
        (forward_raw.norm() <= 1.0e-12).item()
    ):
        raise ValueError("eye and center must not be equal")
    forward = _normalize_torch(forward_raw)
    up_hint = _normalize_torch(up)
    right_raw = torch.linalg.cross(up_hint, forward, dim=0)
    if right_raw.device.type == "cpu" and bool((right_raw.norm() <= 1.0e-12).item()):
        raise ValueError("up must not be parallel to the camera direction")
    right = _normalize_torch(right_raw)
    camera_up = _normalize_torch(torch.linalg.cross(forward, right, dim=0))
    return torch.stack([eye, forward, right, camera_up]).contiguous()


def _bounds_tensor(
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    *,
    device,
) -> torch.Tensor:
    bounds_min = _as_vec3(bounds_min, name="bounds_min", device=device)
    bounds_max = _as_vec3(bounds_max, name="bounds_max", device=device)
    if bounds_min.device.type == "cpu" and bool(
        torch.any(bounds_max <= bounds_min).item()
    ):
        raise ValueError("bounds_max must be greater than bounds_min in all dimensions")
    return torch.stack([bounds_min, bounds_max]).contiguous()


def _color_tensor(
    color: torch.Tensor | None,
    *,
    device,
    shape_name: str,
    expected_rank: int,
) -> torch.Tensor:
    if color is None:
        return torch.zeros((1,) * (expected_rank - 1) + (4,), device=device)
    if color.ndim != expected_rank or color.shape[-1] not in (3, 4):
        raise ValueError(
            f"{shape_name} must have shape (..., 3) or (..., 4), got {tuple(color.shape)}"
        )
    color = color.to(device=device)
    if color.dtype == torch.uint8:
        color = color.to(torch.float32) / 255.0
    else:
        color = color.to(torch.float32)
    if color.shape[-1] == 3:
        alpha = torch.ones(*color.shape[:-1], 1, device=device, dtype=torch.float32)
        color = torch.cat([color, alpha], dim=-1)
    return color.contiguous().clamp(0.0, 1.0)


def _uniform_color_tensor(
    surface_color: torch.Tensor | None,
    *,
    device,
) -> torch.Tensor:
    if surface_color is None:
        color = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=device)
    else:
        color = torch.as_tensor(surface_color, device=device)
        if color.shape not in ((3,), (4,)):
            raise ValueError(
                f"surface_color must have shape (3,) or (4,), got {tuple(color.shape)}"
            )
        if color.dtype == torch.uint8:
            color = color.to(torch.float32) / 255.0
        else:
            color = color.to(torch.float32)
        if color.shape == (3,):
            color = torch.cat([color, torch.ones(1, device=device)])
        color = color.reshape(1, 4)
    return color.contiguous().clamp(0.0, 1.0)


def _light_tensor(light_direction: torch.Tensor | None, *, device) -> torch.Tensor:
    if light_direction is None:
        light_direction = torch.tensor([-0.45, 0.75, -1.0], device=device)
    light_direction = _as_vec3(
        light_direction, name="light_direction", device=device
    ).reshape(1, 3)
    return _normalize_torch(light_direction).contiguous()


@wp.func
def _normalize_vec3(vector: wp.vec3) -> wp.vec3:
    length = wp.length(vector)
    if length <= 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    return vector / length


@wp.func
def _clamp_int(value: int, lo: int, hi: int) -> int:
    return wp.min(wp.max(value, lo), hi)


@wp.func
def _make_ray_direction(
    tid: int,
    width: int,
    height: int,
    camera: wp.array(dtype=wp.vec3),
    tan_half_fov: wp.float32,
    aspect: wp.float32,
) -> wp.vec3:
    y = tid / width
    x = tid - y * width
    px = ((wp.float32(x) + 0.5) / wp.float32(width)) * 2.0 - 1.0
    py = 1.0 - (((wp.float32(y) + 0.5) / wp.float32(height)) * 2.0)
    px = px * tan_half_fov * aspect
    py = py * tan_half_fov
    return _normalize_vec3(camera[1] + px * camera[2] + py * camera[3])


@wp.func
def _sample_field_trilinear(
    field: wp.array3d(dtype=wp.float32),
    point: wp.vec3,
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.float32:
    sx = (
        (point[0] - bounds_min[0])
        / (bounds_max[0] - bounds_min[0])
        * wp.float32(nx - 1)
    )
    sy = (
        (point[1] - bounds_min[1])
        / (bounds_max[1] - bounds_min[1])
        * wp.float32(ny - 1)
    )
    sz = (
        (point[2] - bounds_min[2])
        / (bounds_max[2] - bounds_min[2])
        * wp.float32(nz - 1)
    )

    i0 = _clamp_int(int(wp.floor(sx)), 0, nx - 2)
    j0 = _clamp_int(int(wp.floor(sy)), 0, ny - 2)
    k0 = _clamp_int(int(wp.floor(sz)), 0, nz - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    fx = wp.min(wp.max(sx - wp.float32(i0), 0.0), 1.0)
    fy = wp.min(wp.max(sy - wp.float32(j0), 0.0), 1.0)
    fz = wp.min(wp.max(sz - wp.float32(k0), 0.0), 1.0)

    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def _sample_color_trilinear(
    colors: wp.array4d(dtype=wp.float32),
    point: wp.vec3,
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec4:
    sx = (
        (point[0] - bounds_min[0])
        / (bounds_max[0] - bounds_min[0])
        * wp.float32(nx - 1)
    )
    sy = (
        (point[1] - bounds_min[1])
        / (bounds_max[1] - bounds_min[1])
        * wp.float32(ny - 1)
    )
    sz = (
        (point[2] - bounds_min[2])
        / (bounds_max[2] - bounds_min[2])
        * wp.float32(nz - 1)
    )

    i0 = _clamp_int(int(wp.floor(sx)), 0, nx - 2)
    j0 = _clamp_int(int(wp.floor(sy)), 0, ny - 2)
    k0 = _clamp_int(int(wp.floor(sz)), 0, nz - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    fx = wp.min(wp.max(sx - wp.float32(i0), 0.0), 1.0)
    fy = wp.min(wp.max(sy - wp.float32(j0), 0.0), 1.0)
    fz = wp.min(wp.max(sz - wp.float32(k0), 0.0), 1.0)

    out = wp.vec4(0.0, 0.0, 0.0, 0.0)
    for channel in range(4):
        c000 = colors[i0, j0, k0, channel]
        c100 = colors[i1, j0, k0, channel]
        c010 = colors[i0, j1, k0, channel]
        c110 = colors[i1, j1, k0, channel]
        c001 = colors[i0, j0, k1, channel]
        c101 = colors[i1, j0, k1, channel]
        c011 = colors[i0, j1, k1, channel]
        c111 = colors[i1, j1, k1, channel]

        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        out[channel] = c0 * (1.0 - fz) + c1 * fz
    return out


@wp.func
def _field_gradient(
    field: wp.array3d(dtype=wp.float32),
    point: wp.vec3,
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec3:
    dx = (bounds_max[0] - bounds_min[0]) / wp.float32(nx - 1)
    dy = (bounds_max[1] - bounds_min[1]) / wp.float32(ny - 1)
    dz = (bounds_max[2] - bounds_min[2]) / wp.float32(nz - 1)
    gx = (
        _sample_field_trilinear(
            field,
            point + wp.vec3(0.5 * dx, 0.0, 0.0),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
        - _sample_field_trilinear(
            field,
            point - wp.vec3(0.5 * dx, 0.0, 0.0),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
    ) / dx
    gy = (
        _sample_field_trilinear(
            field,
            point + wp.vec3(0.0, 0.5 * dy, 0.0),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
        - _sample_field_trilinear(
            field,
            point - wp.vec3(0.0, 0.5 * dy, 0.0),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
    ) / dy
    gz = (
        _sample_field_trilinear(
            field,
            point + wp.vec3(0.0, 0.0, 0.5 * dz),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
        - _sample_field_trilinear(
            field,
            point - wp.vec3(0.0, 0.0, 0.5 * dz),
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
    ) / dz
    return wp.vec3(gx, gy, gz)


@wp.func
def _axis_intersection(
    origin: wp.float32,
    direction: wp.float32,
    lo: wp.float32,
    hi: wp.float32,
) -> wp.vec3:
    if wp.abs(direction) < 1.0e-12:
        if origin < lo or origin > hi:
            return wp.vec3(1.0, 0.0, 0.0)
        return wp.vec3(0.0, -3.402823e38, 3.402823e38)

    inv_d = 1.0 / direction
    t0 = (lo - origin) * inv_d
    t1 = (hi - origin) * inv_d
    return wp.vec3(0.0, wp.min(t0, t1), wp.max(t0, t1))


@wp.func
def _ray_box_intersection(
    origin: wp.vec3,
    direction: wp.vec3,
    bounds_min: wp.vec3,
    bounds_max: wp.vec3,
) -> wp.vec3:
    x = _axis_intersection(origin[0], direction[0], bounds_min[0], bounds_max[0])
    y = _axis_intersection(origin[1], direction[1], bounds_min[1], bounds_max[1])
    z = _axis_intersection(origin[2], direction[2], bounds_min[2], bounds_max[2])

    miss = x[0] + y[0] + z[0]
    t_near = wp.max(0.0, wp.max(x[1], wp.max(y[1], z[1])))
    t_far = wp.min(x[2], wp.min(y[2], z[2]))
    if miss > 0.0 or t_far < t_near:
        return wp.vec3(0.0, 0.0, -1.0)
    return wp.vec3(1.0, t_near, t_far)


@wp.func
def _shade(
    color: wp.vec4,
    normal: wp.vec3,
    light_direction: wp.vec3,
    ambient: wp.float32,
) -> wp.vec4:
    diffuse = wp.max(wp.dot(normal, light_direction), 0.0)
    intensity = ambient + (1.0 - ambient) * diffuse
    return wp.vec4(
        color[0] * intensity,
        color[1] * intensity,
        color[2] * intensity,
        color[3],
    )


@wp.func
def _jet_colormap(value: wp.float32) -> wp.vec3:
    r = wp.min(4.0 * value - 1.5, -4.0 * value + 4.5)
    g = wp.min(4.0 * value - 0.5, -4.0 * value + 3.5)
    b = wp.min(4.0 * value + 0.5, -4.0 * value + 2.5)
    return wp.vec3(
        wp.min(wp.max(r, 0.0), 1.0),
        wp.min(wp.max(g, 0.0), 1.0),
        wp.min(wp.max(b, 0.0), 1.0),
    )


@wp.func
def _sample_seed_trilinear(
    seed: wp.array3d(dtype=wp.float32),
    pos: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.float32:
    i0 = _clamp_int(int(wp.floor(pos[0])), 0, nx - 1)
    j0 = _clamp_int(int(wp.floor(pos[1])), 0, ny - 1)
    k0 = _clamp_int(int(wp.floor(pos[2])), 0, nz - 1)
    i1 = _clamp_int(i0 + 1, 0, nx - 1)
    j1 = _clamp_int(j0 + 1, 0, ny - 1)
    k1 = _clamp_int(k0 + 1, 0, nz - 1)

    fx = wp.min(wp.max(pos[0] - wp.float32(i0), 0.0), 1.0)
    fy = wp.min(wp.max(pos[1] - wp.float32(j0), 0.0), 1.0)
    fz = wp.min(wp.max(pos[2] - wp.float32(k0), 0.0), 1.0)

    c000 = seed[i0, j0, k0]
    c100 = seed[i1, j0, k0]
    c010 = seed[i0, j1, k0]
    c110 = seed[i1, j1, k0]
    c001 = seed[i0, j0, k1]
    c101 = seed[i1, j0, k1]
    c011 = seed[i0, j1, k1]
    c111 = seed[i1, j1, k1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def _sample_vector_trilinear(
    vector_field: wp.array4d(dtype=wp.float32),
    pos: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec3:
    i0 = _clamp_int(int(wp.floor(pos[0])), 0, nx - 1)
    j0 = _clamp_int(int(wp.floor(pos[1])), 0, ny - 1)
    k0 = _clamp_int(int(wp.floor(pos[2])), 0, nz - 1)
    i1 = _clamp_int(i0 + 1, 0, nx - 1)
    j1 = _clamp_int(j0 + 1, 0, ny - 1)
    k1 = _clamp_int(k0 + 1, 0, nz - 1)

    fx = wp.min(wp.max(pos[0] - wp.float32(i0), 0.0), 1.0)
    fy = wp.min(wp.max(pos[1] - wp.float32(j0), 0.0), 1.0)
    fz = wp.min(wp.max(pos[2] - wp.float32(k0), 0.0), 1.0)

    result = wp.vec3(0.0, 0.0, 0.0)
    for channel in range(3):
        c000 = vector_field[i0, j0, k0, channel]
        c100 = vector_field[i1, j0, k0, channel]
        c010 = vector_field[i0, j1, k0, channel]
        c110 = vector_field[i1, j1, k0, channel]
        c001 = vector_field[i0, j0, k1, channel]
        c101 = vector_field[i1, j0, k1, channel]
        c011 = vector_field[i0, j1, k1, channel]
        c111 = vector_field[i1, j1, k1, channel]

        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        result[channel] = c0 * (1.0 - fz) + c1 * fz
    return result


@wp.func
def _project_point(
    point: wp.vec3,
    camera: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
) -> wp.vec4:
    rel = point - camera[0]
    z = wp.dot(rel, camera[1])
    x = wp.dot(rel, camera[2])
    y = wp.dot(rel, camera[3])
    if z <= 1.0e-12:
        return wp.vec4(0.0, 0.0, z, 0.0)
    screen_x = (x / (z * tan_half_fov * aspect) + 1.0) * 0.5 * wp.float32(width)
    screen_y = (1.0 - (y / (z * tan_half_fov) + 1.0) * 0.5) * wp.float32(height)
    return wp.vec4(screen_x, screen_y, z, 1.0)


@wp.kernel
def _scalar_field_to_rgba_kernel(
    field: wp.array3d(dtype=wp.float32),
    vmin: wp.float32,
    vmax: wp.float32,
    max_opacity: wp.float32,
    opacity_threshold: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    rgba_volume: wp.array4d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    value = (field[i, j, k] - vmin) / (vmax - vmin)
    value = wp.min(wp.max(value, 0.0), 1.0)
    color = _jet_colormap(value)

    alpha = value
    if alpha < opacity_threshold:
        alpha = 0.0
    alpha = wp.min(wp.max(alpha * max_opacity, 0.0), 1.0)

    rgba_volume[i, j, k, 0] = wp.uint8(color[0] * 255.0)
    rgba_volume[i, j, k, 1] = wp.uint8(color[1] * 255.0)
    rgba_volume[i, j, k, 2] = wp.uint8(color[2] * 255.0)
    rgba_volume[i, j, k, 3] = wp.uint8(alpha * 255.0)


@wp.kernel
def _line_integral_convolution_kernel(
    vector_field: wp.array4d(dtype=wp.float32),
    seed: wp.array3d(dtype=wp.float32),
    step_size: wp.float32,
    num_steps: int,
    contrast: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    line_integral: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    pos = wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))

    total = seed[i, j, k]
    total_weight = wp.float32(1.0)

    for direction_sign in range(2):
        direction_scale = wp.float32(1.0)
        if direction_sign == 1:
            direction_scale = -1.0

        current = pos
        for step in range(num_steps):
            vector = _sample_vector_trilinear(vector_field, current, nx, ny, nz)
            vector_length = wp.length(vector)
            if vector_length <= 1.0e-6:
                break
            vector = direction_scale * vector / vector_length

            mid = current + 0.5 * step_size * vector
            mid_vector = _sample_vector_trilinear(vector_field, mid, nx, ny, nz)
            mid_length = wp.length(mid_vector)
            if mid_length <= 1.0e-6:
                break
            mid_vector = direction_scale * mid_vector / mid_length
            current = current + step_size * mid_vector

            if (
                current[0] < 0.0
                or current[0] > wp.float32(nx - 1)
                or current[1] < 0.0
                or current[1] > wp.float32(ny - 1)
                or current[2] < 0.0
                or current[2] > wp.float32(nz - 1)
            ):
                break

            normalized_step = wp.float32(step + 1) / wp.float32(num_steps + 1)
            weight = 1.0 - normalized_step
            total += _sample_seed_trilinear(seed, current, nx, ny, nz) * weight
            total_weight += weight

    value = total / wp.max(total_weight, 1.0e-6)
    value = wp.min(wp.max(value, 0.0), 1.0)
    value = (value - 0.5) * contrast + 0.5
    line_integral[i, j, k] = wp.min(wp.max(value, 0.0), 1.0)


@wp.kernel
def _vector_field_to_rgba_kernel(
    vector_field: wp.array4d(dtype=wp.float32),
    lic_field: wp.array3d(dtype=wp.float32),
    vmin: wp.float32,
    vmax: wp.float32,
    max_opacity: wp.float32,
    lic_threshold: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    rgba_volume: wp.array4d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    vx = vector_field[i, j, k, 0]
    vy = vector_field[i, j, k, 1]
    vz = vector_field[i, j, k, 2]
    magnitude = wp.sqrt(vx * vx + vy * vy + vz * vz)
    normalized = wp.min(wp.max((magnitude - vmin) / (vmax - vmin), 0.0), 1.0)
    color = _jet_colormap(normalized)

    lic_value = wp.min(wp.max(lic_field[i, j, k], 0.0), 1.0)
    if lic_value < lic_threshold:
        lic_value = 0.0
    alpha = wp.min(wp.max(lic_value * normalized * max_opacity, 0.0), 1.0)

    rgba_volume[i, j, k, 0] = wp.uint8(color[0] * 255.0)
    rgba_volume[i, j, k, 1] = wp.uint8(color[1] * 255.0)
    rgba_volume[i, j, k, 2] = wp.uint8(color[2] * 255.0)
    rgba_volume[i, j, k, 3] = wp.uint8(alpha * 255.0)


@wp.kernel
def _volume_render_kernel(
    rgba_volume: wp.array4d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    bounds: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    step_size: wp.float32,
    max_steps: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    opacity_threshold: wp.float32,
    depth_threshold: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    ray_origin = camera[0]
    ray_direction = _make_ray_direction(
        tid, width, height, camera, tan_half_fov, aspect
    )
    bounds_min = bounds[0]
    bounds_max = bounds[1]
    intersection = _ray_box_intersection(
        ray_origin, ray_direction, bounds_min, bounds_max
    )

    if intersection[0] <= 0.0:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        return

    accum = wp.vec4(0.0, 0.0, 0.0, 0.0)
    first_depth = wp.float32(3.402823e38)
    t = intersection[1]
    for _ in range(max_steps):
        if t > intersection[2] or accum[3] >= opacity_threshold:
            break
        sample = _sample_color_trilinear(
            rgba_volume,
            ray_origin + t * ray_direction,
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
        sample_alpha = wp.min(wp.max(sample[3], 0.0), 1.0)
        if sample_alpha > 0.0:
            opacity = (1.0 - accum[3]) * sample_alpha
            accum[0] += sample[0] * opacity
            accum[1] += sample[1] * opacity
            accum[2] += sample[2] * opacity
            accum[3] += opacity
            if first_depth >= 3.0e38 and accum[3] >= depth_threshold:
                first_depth = t
        t += step_size

    if accum[3] <= 0.0:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        return

    rgba[tid] = wp.vec4(
        accum[0] / accum[3],
        accum[1] / accum[3],
        accum[2] / accum[3],
        accum[3],
    )
    depth[tid] = first_depth


@wp.func
def _write_depth_tested_pixel(
    x: int,
    y: int,
    z: wp.float32,
    color: wp.vec4,
    width: int,
    height: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
):
    if x >= 0 and x < width and y >= 0 and y < height:
        index = y * width + x
        old_depth = wp.atomic_min(depth, index, z)
        if z <= old_depth:
            rgba[index] = color


@wp.kernel
def _point_cloud_depth_kernel(
    points: wp.array2d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    near: wp.float32,
    far: wp.float32,
    point_size: int,
    num_points: int,
    depth_scale: wp.float32,
    winners: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    point = wp.vec3(points[tid, 0], points[tid, 1], points[tid, 2])
    projected = _project_point(point, camera, width, height, tan_half_fov, aspect)
    z = projected[2]
    if z <= near or z >= far:
        return

    radius = point_size / 2
    center_x = int(projected[0])
    center_y = int(projected[1])
    key = wp.int64(z * depth_scale) * wp.int64(num_points) + wp.int64(tid)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = center_x + dx
            y = center_y + dy
            if x >= 0 and x < width and y >= 0 and y < height:
                wp.atomic_min(winners, y * width + x, key)


@wp.kernel
def _point_cloud_resolve_kernel(
    points: wp.array2d(dtype=wp.float32),
    colors: wp.array2d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    uniform_color: wp.array(dtype=wp.vec4),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    has_point_colors: bool,
    num_points: int,
    empty_key: wp.int64,
    winners: wp.array(dtype=wp.int64),
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    key = winners[tid]
    if key == empty_key:
        return

    point_id = int(key % wp.int64(num_points))
    point = wp.vec3(points[point_id, 0], points[point_id, 1], points[point_id, 2])
    projected = _project_point(point, camera, width, height, tan_half_fov, aspect)

    color = uniform_color[0]
    if has_point_colors:
        color = wp.vec4(
            colors[point_id, 0],
            colors[point_id, 1],
            colors[point_id, 2],
            colors[point_id, 3],
        )
    rgba[tid] = color
    depth[tid] = projected[2]


@wp.func
def _draw_line_depth_tested(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    z0: wp.float32,
    z1: wp.float32,
    width: int,
    height: int,
    color: wp.vec4,
    thickness: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
):
    dx = wp.abs(x1 - x0)
    dy = wp.abs(y1 - y0)
    sx = wp.int32(1)
    if x0 > x1:
        sx = -1
    sy = wp.int32(1)
    if y0 > y1:
        sy = -1
    err = dx - dy
    steps = wp.max(dx, dy)
    radius = thickness / 2
    x = x0
    y = y0

    for step in range(8192):
        if step > steps:
            break
        alpha = wp.float32(0.0)
        if steps > 0:
            alpha = wp.float32(step) / wp.float32(steps)
        z = z0 * (1.0 - alpha) + z1 * alpha
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                _write_depth_tested_pixel(
                    x + ox, y + oy, z, color, width, height, rgba, depth
                )

        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


@wp.kernel
def _wireframe_render_kernel(
    edges: wp.array2d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    uniform_color: wp.array(dtype=wp.vec4),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    near: wp.float32,
    far: wp.float32,
    line_thickness: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p0 = wp.vec3(edges[tid, 0], edges[tid, 1], edges[tid, 2])
    p1 = wp.vec3(edges[tid, 3], edges[tid, 4], edges[tid, 5])
    s0 = _project_point(p0, camera, width, height, tan_half_fov, aspect)
    s1 = _project_point(p1, camera, width, height, tan_half_fov, aspect)

    if s0[2] <= near or s0[2] >= far or s1[2] <= near or s1[2] >= far:
        return

    _draw_line_depth_tested(
        int(s0[0]),
        int(s0[1]),
        int(s1[0]),
        int(s1[1]),
        s0[2],
        s1[2],
        width,
        height,
        uniform_color[0],
        line_thickness,
        rgba,
        depth,
    )


@wp.kernel
def _isosurface_render_kernel(
    field: wp.array3d(dtype=wp.float32),
    color_field: wp.array4d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    bounds: wp.array(dtype=wp.vec3),
    uniform_color: wp.array(dtype=wp.vec4),
    light: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    threshold: wp.float32,
    step_size: wp.float32,
    max_steps: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    ambient: wp.float32,
    has_color_field: bool,
    nx: int,
    ny: int,
    nz: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
    normal_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ray_origin = camera[0]
    ray_direction = _make_ray_direction(
        tid, width, height, camera, tan_half_fov, aspect
    )
    bounds_min = bounds[0]
    bounds_max = bounds[1]
    intersection = _ray_box_intersection(
        ray_origin, ray_direction, bounds_min, bounds_max
    )

    if intersection[0] <= 0.0:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    t_far = intersection[2]
    prev_t = intersection[1]
    prev_point = ray_origin + prev_t * ray_direction
    prev_value = _sample_field_trilinear(
        field, prev_point, bounds_min, bounds_max, nx, ny, nz
    )

    found = bool(False)
    hit_t = wp.float32(3.402823e38)

    for _ in range(max_steps):
        if found:
            break
        next_t = prev_t + step_size
        if next_t > t_far:
            break

        next_point = ray_origin + next_t * ray_direction
        next_value = _sample_field_trilinear(
            field, next_point, bounds_min, bounds_max, nx, ny, nz
        )
        if (prev_value - threshold) * (next_value - threshold) <= 0.0:
            denom = next_value - prev_value
            if wp.abs(denom) < 1.0e-7:
                if denom < 0.0:
                    denom = -1.0e-7
                else:
                    denom = 1.0e-7
            alpha = wp.min(wp.max((threshold - prev_value) / denom, 0.0), 1.0)
            hit_t = prev_t + alpha * step_size
            found = True

        prev_t = next_t
        prev_value = next_value

    if not found:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    hit_point = ray_origin + hit_t * ray_direction
    normal = _normalize_vec3(
        _field_gradient(field, hit_point, bounds_min, bounds_max, nx, ny, nz)
    )
    if wp.dot(normal, ray_direction) > 0.0:
        normal = -normal

    color = uniform_color[0]
    if has_color_field:
        color = _sample_color_trilinear(
            color_field, hit_point, bounds_min, bounds_max, nx, ny, nz
        )

    rgba[tid] = _shade(color, normal, light[0], ambient)
    depth[tid] = hit_t
    normal_out[tid] = normal


@wp.kernel
def _mesh_raycast_kernel(
    mesh_id: wp.uint64,
    color_values: wp.array2d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    uniform_color: wp.array(dtype=wp.vec4),
    light: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    max_distance: wp.float32,
    ambient: wp.float32,
    color_mode: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
    normal_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ray_origin = camera[0]
    ray_direction = _make_ray_direction(
        tid, width, height, camera, tan_half_fov, aspect
    )
    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_direction, max_distance)

    if not query.result:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    normal = _normalize_vec3(query.normal)
    if wp.dot(normal, ray_direction) > 0.0:
        normal = -normal

    color = uniform_color[0]
    if color_mode == 1:
        mesh = wp.mesh_get(mesh_id)
        i0 = mesh.indices[3 * query.face + 0]
        i1 = mesh.indices[3 * query.face + 1]
        i2 = mesh.indices[3 * query.face + 2]
        w0 = query.u
        w1 = query.v
        w2 = 1.0 - query.u - query.v
        color = wp.vec4(
            w0 * color_values[i0, 0]
            + w1 * color_values[i1, 0]
            + w2 * color_values[i2, 0],
            w0 * color_values[i0, 1]
            + w1 * color_values[i1, 1]
            + w2 * color_values[i2, 1],
            w0 * color_values[i0, 2]
            + w1 * color_values[i1, 2]
            + w2 * color_values[i2, 2],
            w0 * color_values[i0, 3]
            + w1 * color_values[i1, 3]
            + w2 * color_values[i2, 3],
        )
    elif color_mode == 2:
        color = wp.vec4(
            color_values[query.face, 0],
            color_values[query.face, 1],
            color_values[query.face, 2],
            color_values[query.face, 3],
        )

    rgba[tid] = _shade(color, normal, light[0], ambient)
    depth[tid] = query.t
    normal_out[tid] = normal


def _validate_image_shape(image_height: int, image_width: int) -> None:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be strictly positive")


def _validate_fov(fov_y_degrees: float) -> None:
    if fov_y_degrees <= 0.0 or fov_y_degrees >= 180.0:
        raise ValueError("fov_y_degrees must lie in the open interval (0, 180)")


def _validate_ambient(ambient: float) -> None:
    if ambient < 0.0 or ambient > 1.0:
        raise ValueError("ambient must lie in the closed interval [0, 1]")


def _empty_render_outputs(
    image_height: int,
    image_width: int,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgba = torch.empty(
        (image_height, image_width, 4), device=device, dtype=torch.float32
    )
    depth = torch.empty((image_height, image_width), device=device, dtype=torch.float32)
    normal = torch.empty(
        (image_height, image_width, 3), device=device, dtype=torch.float32
    )
    return rgba, depth, normal


def _empty_image_outputs(
    image_height: int,
    image_width: int,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgba = torch.zeros(
        (image_height, image_width, 4), device=device, dtype=torch.float32
    )
    depth = torch.full(
        (image_height, image_width), 3.402823e38, device=device, dtype=torch.float32
    )
    return rgba, depth


def _validate_transfer_range(vmin: float, vmax: float) -> None:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")


def _validate_opacity(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in the closed interval [0, 1]")


def _validate_clip_range(near: float, far: float) -> None:
    if not math.isfinite(near) or not math.isfinite(far) or near <= 0.0 or far <= near:
        raise ValueError("near and far must satisfy 0 < near < far")


def _validate_vector_field(vector_field: torch.Tensor) -> None:
    if vector_field.ndim != 4 or vector_field.shape[-1] != 3:
        raise ValueError(
            "vector_field must have shape (nx, ny, nz, 3), got "
            f"{tuple(vector_field.shape)}"
        )
    if any(size < 2 for size in vector_field.shape[:3]):
        raise ValueError("vector_field must have at least two samples per dimension")


def _normalize_rgba_volume(rgba_volume: torch.Tensor) -> torch.Tensor:
    if rgba_volume.ndim != 4 or rgba_volume.shape[-1] != 4:
        raise ValueError(
            "rgba_volume must have shape (nx, ny, nz, 4), got "
            f"{tuple(rgba_volume.shape)}"
        )
    if any(size < 2 for size in rgba_volume.shape[:3]):
        raise ValueError("rgba_volume must have at least two samples per dimension")
    if rgba_volume.dtype == torch.uint8:
        rgba_volume = rgba_volume.to(torch.float32) / 255.0
    else:
        rgba_volume = rgba_volume.to(torch.float32)
    return rgba_volume.contiguous().clamp(0.0, 1.0)


@torch.library.custom_op("physicsnemo::scalar_field_to_rgba_warp", mutates_args=())
def scalar_field_to_rgba_impl(
    field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    opacity_threshold: float = 0.1,
) -> torch.Tensor:
    """Launch the Warp scalar-to-RGBA transfer custom op."""
    if field.ndim != 3:
        raise ValueError(
            f"field must have shape (nx, ny, nz), got {tuple(field.shape)}"
        )
    _validate_transfer_range(vmin, vmax)
    _validate_opacity(max_opacity, name="max_opacity")
    _validate_opacity(opacity_threshold, name="opacity_threshold")

    field_fp32 = field.to(dtype=torch.float32).contiguous()
    rgba_volume = torch.empty(*field.shape, 4, device=field.device, dtype=torch.uint8)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(field_fp32)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            _scalar_field_to_rgba_kernel,
            dim=tuple(int(size) for size in field.shape),
            inputs=[
                wp.from_torch(field_fp32, dtype=wp.float32),
                float(vmin),
                float(vmax),
                float(max_opacity),
                float(opacity_threshold),
                int(field.shape[0]),
                int(field.shape[1]),
                int(field.shape[2]),
            ],
            outputs=[wp.from_torch(rgba_volume, dtype=wp.uint8)],
            device=wp_device,
            stream=wp_stream,
        )
    return rgba_volume


@scalar_field_to_rgba_impl.register_fake
def _(
    field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    opacity_threshold: float = 0.1,
) -> torch.Tensor:
    return torch.empty(*field.shape, 4, device=field.device, dtype=torch.uint8)


@torch.library.custom_op("physicsnemo::line_integral_convolution_warp", mutates_args=())
def line_integral_convolution_impl(
    vector_field: torch.Tensor,
    seed: torch.Tensor,
    step_size: float = 0.5,
    num_steps: int = 20,
    contrast: float = 1.4,
) -> torch.Tensor:
    """Launch the Warp line integral convolution custom op."""
    _validate_vector_field(vector_field)
    if seed.shape != vector_field.shape[:3]:
        raise ValueError(
            "seed must have shape matching vector_field spatial dimensions, got "
            f"{tuple(seed.shape)} and {tuple(vector_field.shape[:3])}"
        )
    if step_size <= 0.0:
        raise ValueError("step_size must be strictly positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be strictly positive")
    if contrast <= 0.0:
        raise ValueError("contrast must be strictly positive")

    vector_fp32 = vector_field.to(dtype=torch.float32).contiguous()
    seed_fp32 = seed.to(device=vector_field.device, dtype=torch.float32).contiguous()
    line_integral = torch.empty_like(seed_fp32)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(vector_fp32)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            _line_integral_convolution_kernel,
            dim=tuple(int(size) for size in seed.shape),
            inputs=[
                wp.from_torch(vector_fp32, dtype=wp.float32),
                wp.from_torch(seed_fp32, dtype=wp.float32),
                float(step_size),
                int(num_steps),
                float(contrast),
                int(seed.shape[0]),
                int(seed.shape[1]),
                int(seed.shape[2]),
            ],
            outputs=[wp.from_torch(line_integral, dtype=wp.float32)],
            device=wp_device,
            stream=wp_stream,
        )
    return line_integral


@line_integral_convolution_impl.register_fake
def _(
    vector_field: torch.Tensor,
    seed: torch.Tensor,
    step_size: float = 0.5,
    num_steps: int = 20,
    contrast: float = 1.4,
) -> torch.Tensor:
    return torch.empty_like(seed, dtype=torch.float32)


@torch.library.custom_op("physicsnemo::vector_field_to_rgba_warp", mutates_args=())
def vector_field_to_rgba_impl(
    vector_field: torch.Tensor,
    lic_field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    lic_threshold: float = 0.5,
) -> torch.Tensor:
    """Launch the Warp vector LIC-to-RGBA transfer custom op."""
    _validate_vector_field(vector_field)
    if lic_field.shape != vector_field.shape[:3]:
        raise ValueError(
            "lic_field must have shape matching vector_field spatial dimensions"
        )
    _validate_transfer_range(vmin, vmax)
    _validate_opacity(max_opacity, name="max_opacity")
    _validate_opacity(lic_threshold, name="lic_threshold")

    vector_fp32 = vector_field.to(dtype=torch.float32).contiguous()
    lic_fp32 = lic_field.to(
        device=vector_field.device, dtype=torch.float32
    ).contiguous()
    rgba_volume = torch.empty(
        *vector_field.shape[:3], 4, device=vector_field.device, dtype=torch.uint8
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(vector_fp32)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            _vector_field_to_rgba_kernel,
            dim=tuple(int(size) for size in vector_field.shape[:3]),
            inputs=[
                wp.from_torch(vector_fp32, dtype=wp.float32),
                wp.from_torch(lic_fp32, dtype=wp.float32),
                float(vmin),
                float(vmax),
                float(max_opacity),
                float(lic_threshold),
                int(vector_field.shape[0]),
                int(vector_field.shape[1]),
                int(vector_field.shape[2]),
            ],
            outputs=[wp.from_torch(rgba_volume, dtype=wp.uint8)],
            device=wp_device,
            stream=wp_stream,
        )
    return rgba_volume


@vector_field_to_rgba_impl.register_fake
def _(
    vector_field: torch.Tensor,
    lic_field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    lic_threshold: float = 0.5,
) -> torch.Tensor:
    return torch.empty(
        *vector_field.shape[:3], 4, device=vector_field.device, dtype=torch.uint8
    )


@torch.library.custom_op("physicsnemo::volume_render_warp", mutates_args=())
def volume_render_impl(
    rgba_volume: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    step_size: float = 0.01,
    max_steps: int = 512,
    opacity_threshold: float = 0.95,
    depth_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Warp RGBA volume rendering custom op."""
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    if step_size <= 0.0:
        raise ValueError("step_size must be strictly positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be strictly positive")
    _validate_opacity(opacity_threshold, name="opacity_threshold")
    _validate_opacity(depth_threshold, name="depth_threshold")

    device = rgba_volume.device
    rgba_volume_fp32 = _normalize_rgba_volume(rgba_volume)
    camera = _camera_basis(eye, center, up, device=device)
    bounds = _bounds_tensor(bounds_min, bounds_max, device=device)
    rgba, depth = _empty_image_outputs(image_height, image_width, device=device)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(rgba_volume_fp32)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            _volume_render_kernel,
            dim=image_height * image_width,
            inputs=[
                wp.from_torch(rgba_volume_fp32, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(bounds, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(step_size),
                int(max_steps),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(opacity_threshold),
                float(depth_threshold),
                int(rgba_volume.shape[0]),
                int(rgba_volume.shape[1]),
                int(rgba_volume.shape[2]),
            ],
            outputs=[
                wp.from_torch(rgba.reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(depth.reshape(-1), dtype=wp.float32),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    depth = torch.where(depth >= 3.0e38, torch.full_like(depth, torch.inf), depth)
    return rgba, depth


@volume_render_impl.register_fake
def _(
    rgba_volume: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    step_size: float = 0.01,
    max_steps: int = 512,
    opacity_threshold: float = 0.95,
    depth_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _empty_image_outputs(image_height, image_width, device=rgba_volume.device)


@torch.library.custom_op("physicsnemo::point_cloud_render_warp", mutates_args=())
def point_cloud_render_impl(
    points: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    point_colors: torch.Tensor | None = None,
    point_color: torch.Tensor | None = None,
    point_size: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Warp point cloud rendering custom op."""
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (num_points, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one point")
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    if point_size <= 0:
        raise ValueError("point_size must be strictly positive")
    _validate_clip_range(near, far)

    device = points.device
    points_fp32 = points.to(dtype=torch.float32).contiguous()
    colors = torch.zeros((1, 4), device=device, dtype=torch.float32)
    has_point_colors = point_colors is not None
    if point_colors is not None:
        if point_color is not None:
            raise ValueError("Pass either point_colors or point_color, not both")
        if point_colors.shape[0] != points.shape[0]:
            raise ValueError("point_colors must have one color per point")
        colors = _color_tensor(
            point_colors, device=device, shape_name="point_colors", expected_rank=2
        )
    uniform_color = _uniform_color_tensor(point_color, device=device)
    camera = _camera_basis(eye, center, up, device=device)
    rgba, depth = _empty_image_outputs(image_height, image_width, device=device)
    empty_key = torch.iinfo(torch.int64).max
    max_depth_key = float(empty_key // max(int(points.shape[0]), 1) - 1)
    depth_scale = min(1.0e6, max_depth_key / float(far))
    winners = torch.full(
        (image_height, image_width), empty_key, device=device, dtype=torch.int64
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_fp32)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            _point_cloud_depth_kernel,
            dim=int(points.shape[0]),
            inputs=[
                wp.from_torch(points_fp32, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(near),
                float(far),
                int(point_size),
                int(points.shape[0]),
                float(depth_scale),
            ],
            outputs=[wp.from_torch(winners.reshape(-1), dtype=wp.int64)],
            device=wp_device,
            stream=wp_stream,
        )
        wp.launch(
            _point_cloud_resolve_kernel,
            dim=image_height * image_width,
            inputs=[
                wp.from_torch(points_fp32, dtype=wp.float32),
                wp.from_torch(colors, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(uniform_color, dtype=wp.vec4),
                int(image_width),
                int(image_height),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                bool(has_point_colors),
                int(points.shape[0]),
                int(empty_key),
                wp.from_torch(winners.reshape(-1), dtype=wp.int64),
            ],
            outputs=[
                wp.from_torch(rgba.reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(depth.reshape(-1), dtype=wp.float32),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    depth = torch.where(depth >= 3.0e38, torch.full_like(depth, torch.inf), depth)
    return rgba, depth


@point_cloud_render_impl.register_fake
def _(
    points: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    point_colors: torch.Tensor | None = None,
    point_color: torch.Tensor | None = None,
    point_size: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _empty_image_outputs(image_height, image_width, device=points.device)


@torch.library.custom_op("physicsnemo::wireframe_render_warp", mutates_args=())
def wireframe_render_impl(
    edges: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    line_color: torch.Tensor | None = None,
    line_thickness: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Warp wireframe rendering custom op."""
    if edges.ndim == 3:
        if edges.shape[1:] != (2, 3):
            raise ValueError(
                "edges must have shape (num_edges, 2, 3) or (num_edges, 6)"
            )
        edges = edges.reshape(edges.shape[0], 6)
    elif edges.ndim != 2 or edges.shape[-1] != 6:
        raise ValueError("edges must have shape (num_edges, 2, 3) or (num_edges, 6)")
    if edges.shape[0] == 0:
        raise ValueError("edges must contain at least one edge")
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    if line_thickness <= 0:
        raise ValueError("line_thickness must be strictly positive")
    _validate_clip_range(near, far)

    device = edges.device
    edges_fp32 = edges.to(dtype=torch.float32).contiguous()
    uniform_color = _uniform_color_tensor(line_color, device=device)
    camera = _camera_basis(eye, center, up, device=device)
    rgba, depth = _empty_image_outputs(image_height, image_width, device=device)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(edges_fp32)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            _wireframe_render_kernel,
            dim=int(edges.shape[0]),
            inputs=[
                wp.from_torch(edges_fp32, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(uniform_color, dtype=wp.vec4),
                int(image_width),
                int(image_height),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(near),
                float(far),
                int(line_thickness),
            ],
            outputs=[
                wp.from_torch(rgba.reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(depth.reshape(-1), dtype=wp.float32),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    depth = torch.where(depth >= 3.0e38, torch.full_like(depth, torch.inf), depth)
    return rgba, depth


@wireframe_render_impl.register_fake
def _(
    edges: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    line_color: torch.Tensor | None = None,
    line_thickness: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _empty_image_outputs(image_height, image_width, device=edges.device)


@torch.library.custom_op("physicsnemo::isosurface_render_warp", mutates_args=())
def isosurface_render_impl(
    field: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    threshold: float = 0.0,
    step_size: float = 0.01,
    max_steps: int = 512,
    color_field: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the Warp isosurface rendering custom op."""
    if field.ndim != 3:
        raise ValueError(
            f"field must have shape (nx, ny, nz), got {tuple(field.shape)}"
        )
    if any(size < 2 for size in field.shape):
        raise ValueError("field must have at least two samples in each dimension")
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    _validate_ambient(ambient)
    if step_size <= 0.0:
        raise ValueError("step_size must be strictly positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be strictly positive")

    device = field.device
    field_fp32 = field.to(device=device, dtype=torch.float32).contiguous()
    camera = _camera_basis(eye, center, up, device=device)
    bounds = _bounds_tensor(bounds_min, bounds_max, device=device)
    color_field_fp32 = _color_tensor(
        color_field, device=device, shape_name="color_field", expected_rank=4
    )
    if color_field is not None and color_field.shape[:3] != field.shape:
        raise ValueError(
            f"color_field spatial shape must match field, got {tuple(color_field.shape[:3])}"
            f" and {tuple(field.shape)}"
        )
    uniform_color = _uniform_color_tensor(surface_color, device=device)
    light = _light_tensor(light_direction, device=device)

    rgba, depth, normal = _empty_render_outputs(
        image_height, image_width, device=device
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(field_fp32)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            _isosurface_render_kernel,
            dim=image_height * image_width,
            inputs=[
                wp.from_torch(field_fp32, dtype=wp.float32),
                wp.from_torch(color_field_fp32, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(bounds, dtype=wp.vec3),
                wp.from_torch(uniform_color, dtype=wp.vec4),
                wp.from_torch(light, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(threshold),
                float(step_size),
                int(max_steps),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(ambient),
                color_field is not None,
                int(field_fp32.shape[0]),
                int(field_fp32.shape[1]),
                int(field_fp32.shape[2]),
            ],
            outputs=[
                wp.from_torch(rgba.reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(depth.reshape(-1), dtype=wp.float32),
                wp.from_torch(normal.reshape(-1, 3), dtype=wp.vec3),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    depth = torch.where(depth >= 3.0e38, torch.full_like(depth, torch.inf), depth)
    return rgba, depth, normal


@isosurface_render_impl.register_fake
def _(
    field: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    threshold: float = 0.0,
    step_size: float = 0.01,
    max_steps: int = 512,
    color_field: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _empty_render_outputs(image_height, image_width, device=field.device)


@torch.library.custom_op("physicsnemo::mesh_raycast_warp", mutates_args=())
def mesh_raycast_impl(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    vertex_colors: torch.Tensor | None = None,
    face_colors: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
    max_distance: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the Warp mesh raycast rendering custom op."""
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[-1] != 3:
        raise ValueError(
            "mesh_vertices must have shape (num_vertices, 3), got "
            f"{tuple(mesh_vertices.shape)}"
        )
    if mesh_vertices.shape[0] == 0:
        raise ValueError("mesh_vertices must contain at least one vertex")
    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError("mesh_indices must have shape (num_faces, 3)")
        mesh_indices = mesh_indices.reshape(-1)
    elif mesh_indices.ndim != 1:
        raise ValueError("mesh_indices must be 1D or have shape (num_faces, 3)")
    if mesh_indices.numel() == 0 or mesh_indices.numel() % 3 != 0:
        raise ValueError("mesh_indices must contain complete triangle faces")
    if vertex_colors is not None and face_colors is not None:
        raise ValueError("Pass either vertex_colors or face_colors, not both")
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    _validate_ambient(ambient)
    if max_distance <= 0.0:
        raise ValueError("max_distance must be strictly positive")

    device = mesh_vertices.device
    mesh_vertices_fp32 = mesh_vertices.to(dtype=torch.float32).contiguous()
    mesh_indices_i32 = mesh_indices.to(device=device, dtype=torch.int32).contiguous()
    camera = _camera_basis(eye, center, up, device=device)
    uniform_color = _uniform_color_tensor(surface_color, device=device)
    light = _light_tensor(light_direction, device=device)

    color_mode = 0
    color_values = torch.zeros((1, 4), device=device, dtype=torch.float32)
    if vertex_colors is not None:
        if vertex_colors.shape[0] != mesh_vertices.shape[0]:
            raise ValueError("vertex_colors must have one color per mesh vertex")
        color_values = _color_tensor(
            vertex_colors, device=device, shape_name="vertex_colors", expected_rank=2
        )
        color_mode = 1
    elif face_colors is not None:
        num_faces = mesh_indices_i32.numel() // 3
        if face_colors.shape[0] != num_faces:
            raise ValueError("face_colors must have one color per mesh face")
        color_values = _color_tensor(
            face_colors, device=device, shape_name="face_colors", expected_rank=2
        )
        color_mode = 2

    rgba, depth, normal = _empty_render_outputs(
        image_height, image_width, device=device
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(mesh_vertices_fp32)
    with wp.ScopedStream(wp_stream):
        wp_vertices = wp.from_torch(mesh_vertices_fp32, dtype=wp.vec3)
        wp_indices = wp.from_torch(mesh_indices_i32, dtype=wp.int32)
        mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)
        wp.launch(
            _mesh_raycast_kernel,
            dim=image_height * image_width,
            inputs=[
                mesh.id,
                wp.from_torch(color_values, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(uniform_color, dtype=wp.vec4),
                wp.from_torch(light, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(max_distance),
                float(ambient),
                int(color_mode),
            ],
            outputs=[
                wp.from_torch(rgba.reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(depth.reshape(-1), dtype=wp.float32),
                wp.from_torch(normal.reshape(-1, 3), dtype=wp.vec3),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    depth = torch.where(depth >= 3.0e38, torch.full_like(depth, torch.inf), depth)
    return rgba, depth, normal


@mesh_raycast_impl.register_fake
def _(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    vertex_colors: torch.Tensor | None = None,
    face_colors: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
    max_distance: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _empty_render_outputs(image_height, image_width, device=mesh_vertices.device)


def isosurface_render_warp(
    field: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    threshold: float = 0.0,
    step_size: float = 0.01,
    max_steps: int = 512,
    color_field: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and render an isosurface with Warp."""
    device = field.device
    return isosurface_render_impl(
        field,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        _as_vec3(bounds_min, name="bounds_min", device=device),
        _as_vec3(bounds_max, name="bounds_max", device=device),
        threshold,
        step_size,
        max_steps,
        color_field,
        _optional_tensor_arg(surface_color, device=device),
        _optional_tensor_arg(light_direction, device=device),
        ambient,
    )


def mesh_raycast_warp(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    vertex_colors: torch.Tensor | None = None,
    face_colors: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
    max_distance: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and raycast a mesh with Warp."""
    device = mesh_vertices.device
    return mesh_raycast_impl(
        mesh_vertices,
        mesh_indices,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        vertex_colors,
        face_colors,
        _optional_tensor_arg(surface_color, device=device),
        _optional_tensor_arg(light_direction, device=device),
        ambient,
        max_distance,
    )


def scalar_field_to_rgba_warp(
    field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    opacity_threshold: float = 0.1,
) -> torch.Tensor:
    """Map a scalar field to an RGBA volume with Warp."""
    return scalar_field_to_rgba_impl(
        field,
        vmin,
        vmax,
        max_opacity,
        opacity_threshold,
    )


def line_integral_convolution_warp(
    vector_field: torch.Tensor,
    seed: torch.Tensor,
    step_size: float = 0.5,
    num_steps: int = 20,
    contrast: float = 1.4,
) -> torch.Tensor:
    """Compute line integral convolution with Warp."""
    return line_integral_convolution_impl(
        vector_field,
        seed,
        step_size,
        num_steps,
        contrast,
    )


def vector_field_to_rgba_warp(
    vector_field: torch.Tensor,
    lic_field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    lic_threshold: float = 0.5,
) -> torch.Tensor:
    """Map vector magnitude and LIC values to RGBA with Warp."""
    return vector_field_to_rgba_impl(
        vector_field,
        lic_field,
        vmin,
        vmax,
        max_opacity,
        lic_threshold,
    )


def volume_render_warp(
    rgba_volume: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    step_size: float = 0.01,
    max_steps: int = 512,
    opacity_threshold: float = 0.95,
    depth_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and render an RGBA volume with Warp."""
    device = rgba_volume.device
    return volume_render_impl(
        rgba_volume,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        _as_vec3(bounds_min, name="bounds_min", device=device),
        _as_vec3(bounds_max, name="bounds_max", device=device),
        step_size,
        max_steps,
        opacity_threshold,
        depth_threshold,
    )


def point_cloud_render_warp(
    points: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    point_colors: torch.Tensor | None = None,
    point_color: torch.Tensor | None = None,
    point_size: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and rasterize a point cloud with Warp."""
    device = points.device
    return point_cloud_render_impl(
        points,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        point_colors,
        _optional_tensor_arg(point_color, device=device),
        point_size,
        near,
        far,
    )


def wireframe_render_warp(
    edges: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    line_color: torch.Tensor | None = None,
    line_thickness: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and rasterize wireframe segments with Warp."""
    device = edges.device
    return wireframe_render_impl(
        edges,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        _optional_tensor_arg(line_color, device=device),
        line_thickness,
        near,
        far,
    )


__all__ = [
    "isosurface_render_warp",
    "line_integral_convolution_warp",
    "mesh_raycast_warp",
    "point_cloud_render_warp",
    "scalar_field_to_rgba_warp",
    "vector_field_to_rgba_warp",
    "volume_render_warp",
    "wireframe_render_warp",
]
