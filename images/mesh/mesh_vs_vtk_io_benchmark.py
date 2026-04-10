"""PhysicsNeMo-Mesh I/O Benchmark: VTU vs PMSH format comparison.

Quantitatively compares disk size and deserialization time between
VTK's VTU/VTP format and PhysicsNeMo-Mesh's native memmap format (.pmsh),
including both interior and boundary mesh components.

Three formats are compared:
  1. VTU (all fields)           - the raw VTK files as-is
  2. VTU (physics fields only)  - VTK selective loading, same fields as PMSH
  3. PhysicsNeMo-Mesh (.pmsh)   - the optimized memmap format

For VTU files using inline binary encoding (where selective loading provides
no I/O benefit), the "trimmed" bar equals the full VTU bar - annotated with
a dagger (†) to indicate this limitation.

Usage:
    uv run benchmark.py
"""

import functools
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista

### Force line-buffered output (so progress appears immediately when piped)
print = functools.partial(print, flush=True)  # noqa: A001

### Configuration ###########################################################

N_TRIALS: int = 3

_DATASETS_ROOT = Path("/home/psharpe/coreai_modulus_cae/datasets")
_HIGHLIFT_VTU_ROOT = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae"
    "/users/nashton/cadence/HiLiftAeroML-refined"
)


@dataclass(frozen=True)
class SampleConfig:
    """Paths to matched VTU and PMSH mesh files for one sample.

    Each sample typically has multiple mesh components (interior + boundary).
    Both tuples must list corresponding components in the same order.

    Args:
        vtu_paths: VTU/VTP files (interior mesh, boundary mesh, etc.).
        pmsh_paths: Corresponding PMSH memmap directories.
    """

    vtu_paths: tuple[Path, ...]
    pmsh_paths: tuple[Path, ...]


DATASETS: dict[str, list[SampleConfig]] = {
    "ShiftSUV": [
        SampleConfig(
            vtu_paths=(
                _DATASETS_ROOT / "shift_suv/SUV/AeroSUV_full_scale_estate_transient/run_00004/merged_volumes.vtu",
                _DATASETS_ROOT / "shift_suv/SUV/AeroSUV_full_scale_estate_transient/run_00004/merged_surfaces.vtp",
            ),
            pmsh_paths=(
                _DATASETS_ROOT / "shift_suv_pnm_mesh/SUV/estate/test/run_00004/merged_volumes.vtu.pmsh",
                _DATASETS_ROOT / "shift_suv_pnm_mesh/SUV/estate/test/run_00004/merged_surfaces.vtp.pmsh",
            ),
        ),
    ],
    "DrivAerML": [
        SampleConfig(
            vtu_paths=(
                _DATASETS_ROOT / "drivaer_aws/drivaer_data_full/run_1/volume_1.vtu",
                _DATASETS_ROOT / "drivaer_aws/drivaer_data_full/run_1/boundary_1.vtp",
            ),
            pmsh_paths=(
                _DATASETS_ROOT / "drivaer_aws/drivaer_data_pnm_mesh/train/run_1/volume_1.vtu.pmsh",
                _DATASETS_ROOT / "drivaer_aws/drivaer_data_pnm_mesh/train/run_1/boundary_1.vtp.pmsh",
            ),
        ),
    ],
    "HiLiftAeroML": [
        SampleConfig(
            vtu_paths=(
                _HIGHLIFT_VTU_ROOT / "geo_LHC001_AoA_6/volume_geo_LHC001_AoA_6.vtu",
                _HIGHLIFT_VTU_ROOT / "geo_LHC001_AoA_6/boundary_geo_LHC001_AoA_6.vtu",
            ),
            pmsh_paths=(
                _DATASETS_ROOT / "highlift_pnm_mesh/AoA_6/geo_LHC001_AoA_6/volume_geo_LHC001_AoA_6.vtu.pmsh",
                _DATASETS_ROOT / "highlift_pnm_mesh/AoA_6/geo_LHC001_AoA_6/boundary_geo_LHC001_AoA_6.vtu.pmsh",
            ),
        ),
    ],
}


### Result Types #############################################################


@dataclass(frozen=True)
class FormatResult:
    """Benchmark results for a single format on one dataset.

    Attributes:
        disk_size_bytes: Total disk size across all mesh components.
        load_times_s: Wall-clock load times (all components combined per trial).
        n_points: Total vertices across all mesh components.
        n_cells: Total cells across all mesh components.
        n_fields: Total point_data keys across all mesh components.
        is_selective_load: Whether VTK selective loading was actually used
            (False for inline binary VTU where selective loading is a no-op).
    """

    disk_size_bytes: int
    load_times_s: tuple[float, ...]
    n_points: int
    n_cells: int
    n_fields: int
    is_selective_load: bool = True

    @property
    def median_load_time_s(self) -> float:
        """Median load time across trials (robust to outliers)."""
        return statistics.median(self.load_times_s)


### Measurement ##############################################################


def measure_disk_size(path: Path) -> int:
    """Total bytes on disk for a file or directory (recursive for directories)."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _manifold_dim_for(path: Path) -> int | str:
    """Infer manifold_dim from filename: interior/volume meshes become point clouds."""
    if "volume" in path.stem.lower():
        return 0
    return "auto"


def _uses_appended_format(path: Path) -> bool:
    """Check whether a VTU/VTP file uses appended binary (vs inline binary).

    Appended binary supports efficient selective array loading because the
    reader can seek directly to desired arrays using byte offsets. Inline
    binary requires scanning the entire file regardless of which arrays are
    requested.
    """
    with open(path, "rb") as f:
        header = f.read(4096)
    return b'format="appended"' in header or b"format='appended'" in header


def _evict_page_cache(path: Path) -> None:
    """Advise the kernel to drop a file's pages from the page cache.

    Flushes dirty buffers first (so FADV_DONTNEED can actually release
    pages), then uses posix_fadvise(FADV_DONTNEED) on each file.
    """
    os.sync()
    targets = list(path.rglob("*")) if path.is_dir() else [path]
    for target in targets:
        if not target.is_file():
            continue
        fd = os.open(str(target), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def load_vtu(path: Path) -> Mesh:
    """Load a VTU/VTP file via PyVista and convert to PhysicsNeMo Mesh.

    pv.read() parses the VTK XML format (C++ under the hood, produces numpy
    arrays), then from_pyvista() wraps the arrays as torch tensors (zero-copy
    where possible).

    Interior meshes (detected by "volume" in the filename) are loaded as
    point clouds (manifold_dim=0), matching the PMSH conversion pipeline.
    Boundary meshes keep their natural manifold dimension (surface triangles).
    """
    pv_mesh = pv.read(str(path))
    mdim = _manifold_dim_for(path)
    return from_pyvista(pv_mesh, manifold_dim=mdim, warn_on_lost_data=False)


def load_vtu_selective(path: Path, keep_point_fields: set[str]) -> Mesh:
    """Load a VTU/VTP file with only selected point arrays via VTK reader.

    Uses the VTK XML reader's array selection API to disable unwanted arrays
    before reading. Only effective for appended-binary VTU files.
    """
    if str(path).endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()

    ### Diagnostic: report field name matching
    n_point = reader.GetNumberOfPointArrays()
    vtu_names = {reader.GetPointArrayName(i) for i in range(n_point)}
    enabled = sorted(vtu_names & keep_point_fields)
    disabled = sorted(vtu_names - keep_point_fields)
    unmatched = sorted(keep_point_fields - vtu_names)
    print(f"      Selective load {path.name}: "
          f"{len(enabled)}/{len(vtu_names)} point arrays enabled")
    if enabled:
        print(f"        Enabled:  {enabled}")
    if disabled:
        print(f"        Disabled: {disabled}")
    if unmatched:
        print(f"        WARNING - PMSH fields not in VTU: {unmatched}")

    ### Enable only matching point arrays
    for i in range(n_point):
        name = reader.GetPointArrayName(i)
        reader.SetPointArrayStatus(name, int(name in keep_point_fields))

    ### Disable all cell arrays
    for i in range(reader.GetNumberOfCellArrays()):
        reader.SetCellArrayStatus(reader.GetCellArrayName(i), 0)

    reader.Update()
    pv_mesh = pv.wrap(reader.GetOutput())
    mdim = _manifold_dim_for(path)
    return from_pyvista(pv_mesh, manifold_dim=mdim, warn_on_lost_data=False)


def load_pmsh(path: Path) -> Mesh:
    """Load a PMSH memmap directory and materialize into contiguous memory.

    Mesh.load() opens memmap file handles (near-instant, no disk reads).
    .clone() forces the actual read from the OS page cache / disk into
    freshly-allocated contiguous CPU memory - this is the fair comparison
    point against pv.read(), which reads everything eagerly.
    """
    mesh = Mesh.load(path)
    return mesh.clone()


def _extract_mesh_metadata(meshes: Sequence[Mesh]) -> tuple[int, int, int]:
    """Sum (n_points, n_cells, n_fields) across a list of meshes."""
    return (
        sum(m.n_points for m in meshes),
        sum(m.n_cells for m in meshes),
        sum(len(m.point_data.keys()) for m in meshes),
    )


### Subprocess Isolation ######################################################


@dataclass(frozen=True)
class _SubprocessResult:
    """Timing and metadata returned from a subprocess cold load."""

    elapsed: float
    n_points: int
    n_cells: int
    n_fields: int


def _build_load_code(
    load_fn_name: str,
    paths: Sequence[Path],
    *,
    field_sets: Sequence[set[str]] | None = None,
) -> str:
    """Generate Python code for a subprocess to load meshes and report timing.

    The code imports the named load function from benchmark.py, calls it on
    each path, measures wall-clock time, and prints a JSON result line. Any
    other output (e.g. diagnostic prints from load_vtu_selective) precedes
    the JSON line.
    """
    if field_sets is not None:
        loads = ", ".join(
            f"{load_fn_name}(Path({str(p)!r}), {fs!r})"
            for p, fs in zip(paths, field_sets)
        )
    else:
        loads = ", ".join(
            f"{load_fn_name}(Path({str(p)!r}))" for p in paths
        )
    return (
        "import time, gc, json\n"
        "from pathlib import Path\n"
        f"from benchmark import {load_fn_name}, _extract_mesh_metadata\n"
        "gc.collect()\n"
        "t0 = time.perf_counter()\n"
        f"meshes = [{loads}]\n"
        "elapsed = time.perf_counter() - t0\n"
        "meta = _extract_mesh_metadata(meshes)\n"
        'print(json.dumps({"elapsed": elapsed,'
        ' "n_points": meta[0], "n_cells": meta[1], "n_fields": meta[2]}))\n'
    )


def _cold_load_subprocess(
    load_code: str,
    evict_paths: Sequence[Path],
    *,
    verbose: bool = False,
) -> _SubprocessResult:
    """Run a load in a fresh subprocess after evicting page cache.

    Guarantees cold-disk I/O by flushing dirty buffers, advising
    FADV_DONTNEED on all data files, then spawning a clean Python
    process that performs the load and reports timing via JSON on stdout.

    Args:
        load_code: Python source (from _build_load_code) to execute.
        evict_paths: Files/directories to evict from page cache.
        verbose: If True, forward subprocess diagnostic output to stdout.
    """
    for p in evict_paths:
        _evict_page_cache(p)
    result = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c", load_code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )
    if result.returncode != 0:
        print(f"    [subprocess error] return code {result.returncode}")
        if result.stdout:
            print(f"    [stdout] {result.stdout[:1000]}")
        if result.stderr:
            print(f"    [stderr] {result.stderr[:1000]}")
        raise RuntimeError(
            f"Subprocess failed (rc={result.returncode}): {result.stderr[:200]}"
        )
    lines = result.stdout.strip().split("\n")
    data = json.loads(lines[-1])
    if verbose and len(lines) > 1:
        for line in lines[:-1]:
            print(f"    {line}")
    return _SubprocessResult(
        elapsed=data["elapsed"],
        n_points=data["n_points"],
        n_cells=data["n_cells"],
        n_fields=data["n_fields"],
    )


### Formatting ###############################################################


def human_bytes(n: int | float) -> str:
    """Format a byte count as a human-readable string (binary prefixes)."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PiB"


### Benchmarking #############################################################


def benchmark_dataset(
    dataset_name: str,
    samples: list[SampleConfig],
    *,
    n_trials: int,
) -> dict[str, FormatResult]:
    """Benchmark VTU, VTU (trimmed), and PMSH loading for all samples.

    Each timed load runs in a fresh subprocess with page cache eviction
    before spawn, ensuring true cold-disk I/O measurements. An initial
    in-process PMSH load discovers field names for selective VTU loading
    and prints verification diagnostics; it is not used for timing.

    For inline-binary VTU files, the "trimmed" result is identical to the
    full VTU result because the format cannot skip unused arrays.

    Returns:
        Dict with keys "VTU", "VTU (trimmed)", "PMSH".
    """
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name} ({len(samples)} sample(s))")
    print(f"{'=' * 60}")

    pmsh_sizes: list[int] = []
    pmsh_times: list[float] = []
    pmsh_meta: tuple[int, int, int] = (0, 0, 0)

    vtu_sizes: list[int] = []
    vtu_times: list[float] = []
    vtu_meta: tuple[int, int, int] = (0, 0, 0)

    trimmed_sizes: list[int] = []
    trimmed_times: list[float] = []
    trimmed_meta: tuple[int, int, int] = (0, 0, 0)
    trimmed_is_selective = True

    for i, sample in enumerate(samples):
        print(f"\n  Sample {i + 1}/{len(samples)}:")
        for label, paths in [("VTU", sample.vtu_paths), ("PMSH", sample.pmsh_paths)]:
            for p in paths:
                print(f"    {label}: {p}")

        ### Disk sizes
        pmsh_disk = sum(measure_disk_size(p) for p in sample.pmsh_paths)
        pmsh_sizes.append(pmsh_disk)
        vtu_disk = sum(measure_disk_size(p) for p in sample.vtu_paths)
        vtu_sizes.append(vtu_disk)
        trimmed_sizes.append(vtu_disk)

        ### In-process PMSH load: discover field names + verification
        gc.collect()
        pmsh_meshes = [load_pmsh(p) for p in sample.pmsh_paths]
        pmsh_field_sets = [set(m.point_data.keys()) for m in pmsh_meshes]

        if i == 0:
            print("\n    PMSH mesh repr (first component):")
            for line in repr(pmsh_meshes[0]).splitlines():
                print(f"      {line}")

            tensor_bytes = 0
            for m in pmsh_meshes:
                tensor_bytes += m.points.nelement() * m.points.element_size()
                for key in m.point_data.keys():
                    t = m.point_data[key]
                    tensor_bytes += t.nelement() * t.element_size()
            print(f"\n    Verify: points type = {type(pmsh_meshes[0].points).__name__}")
            print(f"    Verify: materialized tensor data = {tensor_bytes / 2**30:.3f} GiB")

            t0 = time.perf_counter()
            all_sums = []
            for m in pmsh_meshes:
                all_sums.append(m.points.sum())
                for key in m.point_data.keys():
                    all_sums.append(m.point_data[key].sum())
            sum_time = time.perf_counter() - t0
            checksum = sum(s.item() for s in all_sums)
            print(f"    Verify: all-fields checksum = {checksum:.6e} ({sum_time:.3f} s)")

            for j, fields in enumerate(pmsh_field_sets):
                print(f"    PMSH component {j} fields: {sorted(fields)}")

        del pmsh_meshes
        gc.collect()

        ### Build subprocess code strings
        all_appended = all(_uses_appended_format(p) for p in sample.vtu_paths)
        pmsh_code = _build_load_code("load_pmsh", sample.pmsh_paths)
        vtu_code = _build_load_code("load_vtu", sample.vtu_paths)
        trimmed_code = (
            _build_load_code(
                "load_vtu_selective", sample.vtu_paths,
                field_sets=pmsh_field_sets,
            )
            if all_appended
            else None
        )
        all_sample_paths = list(sample.vtu_paths) + list(sample.pmsh_paths)

        ### Cold trials via subprocess isolation
        sample_vtu_times: list[float] = []
        sample_trimmed_times: list[float] = []

        for trial in range(n_trials):
            sr = _cold_load_subprocess(pmsh_code, all_sample_paths)
            pmsh_times.append(sr.elapsed)
            if trial == 0:
                pmsh_meta = (sr.n_points, sr.n_cells, sr.n_fields)
            print(f"    PMSH trial {trial + 1}/{n_trials}: {sr.elapsed:.3f} s")

            if trimmed_code is not None:
                sr = _cold_load_subprocess(
                    trimmed_code, all_sample_paths, verbose=(trial == 0),
                )
                if trial == 0:
                    trimmed_meta = (sr.n_points, sr.n_cells, sr.n_fields)
                sample_trimmed_times.append(sr.elapsed)
                print(f"    VTU (trimmed) trial {trial + 1}/{n_trials}: {sr.elapsed:.3f} s")

            sr = _cold_load_subprocess(vtu_code, all_sample_paths)
            if trial == 0:
                vtu_meta = (sr.n_points, sr.n_cells, sr.n_fields)
            sample_vtu_times.append(sr.elapsed)
            print(f"    VTU full trial {trial + 1}/{n_trials}: {sr.elapsed:.3f} s")

        vtu_times.extend(sample_vtu_times)

        if all_appended:
            trimmed_times.extend(sample_trimmed_times)
        else:
            trimmed_is_selective = False
            trimmed_times.extend(sample_vtu_times)
            trimmed_meta = vtu_meta
            print(
                "    VTU (trimmed): same as full VTU "
                "(\u2020 inline binary cannot skip arrays)"
            )

    return {
        "VTU": FormatResult(
            disk_size_bytes=int(statistics.mean(vtu_sizes)),
            load_times_s=tuple(vtu_times),
            n_points=vtu_meta[0],
            n_cells=vtu_meta[1],
            n_fields=vtu_meta[2],
        ),
        "VTU (trimmed)": FormatResult(
            disk_size_bytes=int(statistics.mean(trimmed_sizes)),
            load_times_s=tuple(trimmed_times),
            n_points=trimmed_meta[0],
            n_cells=trimmed_meta[1],
            n_fields=trimmed_meta[2],
            is_selective_load=trimmed_is_selective,
        ),
        "PMSH": FormatResult(
            disk_size_bytes=int(statistics.mean(pmsh_sizes)),
            load_times_s=tuple(pmsh_times),
            n_points=pmsh_meta[0],
            n_cells=pmsh_meta[1],
            n_fields=pmsh_meta[2],
        ),
    }


def run_benchmarks(
    datasets: dict[str, list[SampleConfig]],
    *,
    n_trials: int = N_TRIALS,
) -> dict[str, dict[str, FormatResult]]:
    """Run benchmarks across all configured datasets.

    Returns:
        dict[dataset_name, dict[format_name, FormatResult]].
    """
    return {
        name: benchmark_dataset(name, samples, n_trials=n_trials)
        for name, samples in datasets.items()
    }


### Output ###################################################################

FORMAT_KEYS = ("VTU", "VTU (trimmed)", "PMSH")


def print_results_table(results: dict[str, dict[str, FormatResult]]) -> None:
    """Print a formatted comparison table to stdout."""
    print(f"\n{'=' * 100}")
    print("  BENCHMARK RESULTS")
    print(f"{'=' * 100}\n")

    header = (
        f"{'Dataset':<16} {'Format':<16} {'Disk Size':>12} "
        f"{'Load (s)':>14} {'Points':>12} {'Cells':>12} {'Fields':>8}"
    )
    print(header)
    print("-" * len(header))

    for dataset_name, fmt_results in results.items():
        vtu = fmt_results["VTU"]
        trimmed = fmt_results["VTU (trimmed)"]
        pmsh = fmt_results["PMSH"]

        dagger = "" if trimmed.is_selective_load else " \u2020"

        for fmt, r in [("VTU", vtu), (f"VTU (trimmed){dagger}", trimmed), ("PMSH", pmsh)]:
            print(
                f"{dataset_name:<16} {fmt:<16} "
                f"{human_bytes(r.disk_size_bytes):>12} "
                f"{r.median_load_time_s:>14.3f} "
                f"{r.n_points:>12,} {r.n_cells:>12,} {r.n_fields:>8}"
            )

        size_ratio = (
            vtu.disk_size_bytes / pmsh.disk_size_bytes
            if pmsh.disk_size_bytes > 0
            else float("inf")
        )
        time_ratio = (
            vtu.median_load_time_s / pmsh.median_load_time_s
            if pmsh.median_load_time_s > 0
            else float("inf")
        )
        print(
            f"{'':>16} {'VTU/PMSH ratio':>16} "
            f"{size_ratio:>11.1f}x "
            f"{'':>8}{time_ratio:>7.1f}x faster"
        )
        print()

    print(f"Note: Each format is cold-loaded {N_TRIALS}x per sample (subprocess isolation, cache eviction).")
    print("      \u2020 = VTU inline binary format does not support selective field loading.")
    print("      PMSH supports direct GPU loading: Mesh.load(path).to('cuda')")
    print(
        "      VTU requires CPU-side parsing first: "
        "pv.read() -> from_pyvista() -> .to('cuda')"
    )


def plot_comparison_chart(
    results: dict[str, dict[str, FormatResult]],
    output_path: Path = Path("benchmark_results.png"),
) -> None:
    """Generate a 2-panel comparison chart (disk size + load time).

    Disk size panel: 2 bars (VTU, PhysicsNeMo-Mesh).
    Load time panel: 2 bars, but for datasets where VTU supports selective
    field loading (appended binary), the VTU bar is stacked to show the
    physics-fields-only portion vs extra-fields overhead.
    """
    datasets = list(results.keys())
    n = len(datasets)
    x = np.arange(n)
    bar_width = 0.35

    vtu_sizes_gb = [results[d]["VTU"].disk_size_bytes / 1e9 for d in datasets]
    pmsh_sizes_gb = [results[d]["PMSH"].disk_size_bytes / 1e9 for d in datasets]

    vtu_times = [results[d]["VTU"].median_load_time_s for d in datasets]
    trim_times = [results[d]["VTU (trimmed)"].median_load_time_s for d in datasets]
    pmsh_times = [results[d]["PMSH"].median_load_time_s for d in datasets]
    is_selective = [results[d]["VTU (trimmed)"].is_selective_load for d in datasets]

    fig, (ax_size, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5))
    vtu_color = "#999999"
    vtu_trim_color = "#666666"
    pmsh_color = "#76B900"

    ### Left panel: disk size (2 bars, simple)
    ax_size.bar(
        x - bar_width / 2, vtu_sizes_gb, bar_width, label="VTU", color=vtu_color
    )
    ax_size.bar(
        x + bar_width / 2, pmsh_sizes_gb, bar_width,
        label="PhysicsNeMo-Mesh *", color=pmsh_color,
    )
    ax_size.set_ylabel("Disk Size (GB)")
    ax_size.set_title("Disk Size per Sample (interior + boundary)")
    ax_size.set_xticks(x)
    ax_size.set_xticklabels(datasets)
    ax_size.legend(fontsize=9)

    for i in range(n):
        ax_size.annotate(
            f"{vtu_sizes_gb[i]:.1f} GB",
            xy=(x[i] - bar_width / 2, vtu_sizes_gb[i]),
            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9,
        )
        ratio = vtu_sizes_gb[i] / pmsh_sizes_gb[i] if pmsh_sizes_gb[i] > 0 else 0
        ax_size.annotate(
            f"{pmsh_sizes_gb[i]:.1f} GB\n({ratio:.1f}\u00d7\nsmaller)",
            xy=(x[i] + bar_width / 2, pmsh_sizes_gb[i]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontweight="bold", fontsize=9,
        )

    ### Right panel: load time (2 bars, stacked VTU where selective loading applies)
    has_any_selective = any(is_selective)

    for i in range(n):
        vx = x[i] - bar_width / 2

        if is_selective[i]:
            extra_time = vtu_times[i] - trim_times[i]
            ax_time.bar(
                vx, trim_times[i], bar_width, color=vtu_trim_color,
                label="VTU (physics fields only)" if i == 0 or not any(is_selective[:i]) else "",
            )
            ax_time.bar(
                vx, extra_time, bar_width, bottom=trim_times[i], color=vtu_color,
                label="VTU (extra fields overhead)" if i == 0 or not any(is_selective[:i]) else "",
            )
            ax_time.annotate(
                f"{vtu_times[i]:.1f} s total",
                xy=(vx, vtu_times[i]),
                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8,
            )
            ax_time.annotate(
                f"physics fields\nonly: {trim_times[i]:.1f} s",
                xy=(vx, trim_times[i]),
                xytext=(0, -5), textcoords="offset points",
                ha="center", va="top", fontsize=7, fontstyle="italic",
            )
        else:
            ax_time.bar(
                vx, vtu_times[i], bar_width, color=vtu_color,
                label="VTU" if not has_any_selective and i == 0 else "",
            )
            ax_time.annotate(
                f"{vtu_times[i]:.1f} s",
                xy=(vx, vtu_times[i]),
                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9,
            )

    ax_time.bar(
        x + bar_width / 2, pmsh_times, bar_width,
        label="PhysicsNeMo-Mesh *", color=pmsh_color,
    )

    for i in range(n):
        ref_time = trim_times[i] if is_selective[i] else vtu_times[i]
        ratio = ref_time / pmsh_times[i] if pmsh_times[i] > 0 else 0
        ax_time.annotate(
            f"{pmsh_times[i]:.1f} s\n({ratio:.1f}\u00d7\nfaster)",
            xy=(x[i] + bar_width / 2, pmsh_times[i]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontweight="bold", fontsize=9,
        )

    ax_time.set_ylabel("Deserialization Time (s)")
    ax_time.set_title("Load Time per Sample (interior + boundary)")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(datasets)
    ax_time.legend(fontsize=8, loc="upper left")

    ### Title and footnotes
    fig.suptitle(
        "I/O Benchmark: VTU vs. PhysicsNeMo-Mesh (*.pmsh)",
        fontsize=14, fontweight="bold",
    )
    footnotes = (
        "* PhysicsNeMo-Mesh stores only physics-relevant fields: mean flow quantities "
        "(pressure, velocity, temperature, density). Fields excluded during curation include\n"
        "  normal vectors (recomputed on-the-fly), cell/node IDs (trivially reconstructable), "
        "and solver-internal quantities (RMS, Reynolds stress, etc.).\n"
        "  To isolate the effect of field trimming from format efficiency, the HiLiftAeroML "
        "load-time bar shows VTU with selective field loading (\u2020), matching the PMSH field set.\n"
        "  Cell connectivity dominates VTU file size for these meshes, so the trimming effect "
        "on load time is modest; the bulk of the speedup is attributable to the memmap format.\n"
        "\u2020 HiLiftAeroML uses VTK appended binary, which supports selective array loading. "
        "ShiftSUV and DrivAerML use inline binary, where the parser must read the full file "
        "regardless of field selection."
    )
    fig.text(
        0.02, 0.02, footnotes, ha="left", fontsize=7.5, fontstyle="italic",
        va="bottom", linespacing=1.4,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nChart saved to: {output_path.resolve()}")


### Entry Point ##############################################################


def validate_paths(datasets: dict[str, list[SampleConfig]]) -> list[str]:
    """Check that all configured paths exist and are readable.

    Returns:
        List of error messages (empty if all paths are valid).
    """
    errors: list[str] = []
    for name, samples in datasets.items():
        for i, sample in enumerate(samples):
            for p in sample.vtu_paths:
                if not p.exists():
                    errors.append(f"  {name} sample {i}: VTU not found: {p}")
                elif not p.is_file():
                    errors.append(f"  {name} sample {i}: VTU is not a file: {p}")
            for p in sample.pmsh_paths:
                if not p.exists():
                    errors.append(f"  {name} sample {i}: PMSH not found: {p}")
                elif not p.is_dir():
                    errors.append(f"  {name} sample {i}: PMSH is not a directory: {p}")
    return errors


def main() -> None:
    errors = validate_paths(DATASETS)
    if errors:
        print("ERROR: Some configured paths are invalid:\n")
        print("\n".join(errors))
        print("\nPlease update the DATASETS configuration in benchmark.py.")
        sys.exit(1)

    results = run_benchmarks(DATASETS, n_trials=N_TRIALS)
    print_results_table(results)
    plot_comparison_chart(results)


if __name__ == "__main__":
    main()
