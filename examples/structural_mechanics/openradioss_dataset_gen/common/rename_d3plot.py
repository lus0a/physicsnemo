"""
Rename OpenRadioss-generated d3plot files to the LS-DYNA convention expected
by PhysicsNeMo-Curator.

OpenRadioss (via `vortex_radioss`) emits:
    <BASE>.d3plot
    <BASE>.d3plot01
    <BASE>.d3plot02
    ...

PhysicsNeMo-Curator expects:
    d3plot
    d3plot01
    d3plot02
    ...

Replaces the previous shell script and works for any `<BASE>` (e.g.
`Bumper_Beam_AP_meshed` or `Cell_Phone_Drop`).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def rename_run(run_dir: Path, base_name: str) -> int:
    renamed = 0
    pattern = re.compile(rf"^{re.escape(base_name)}\.d3plot(.*)$")

    for entry in run_dir.iterdir():
        if not entry.is_file():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        suffix = m.group(1)
        new_name = f"d3plot{suffix}" if suffix else "d3plot"
        new_path = run_dir / new_name
        if new_path.exists():
            print(f"  skip {entry.name}: {new_name} already exists")
            continue
        print(f"  {entry.name} -> {new_name}")
        entry.rename(new_path)
        renamed += 1
    return renamed


def rename_all(dataset_dir: Path, base_name: str) -> None:
    for run_dir in sorted(dataset_dir.glob("run*")):
        if not run_dir.is_dir():
            continue
        print(f"Processing {run_dir}")
        rename_run(run_dir, base_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Dataset directory containing run*/ subfolders.",
    )
    parser.add_argument(
        "--base-name",
        required=True,
        help="Base name used in the Radioss input deck "
        "(e.g. Bumper_Beam_AP_meshed or Cell_Phone_Drop).",
    )
    args = parser.parse_args()
    if not args.dataset_dir.exists():
        raise SystemExit(f"Dataset dir {args.dataset_dir} not found")
    rename_all(args.dataset_dir, args.base_name)
    print("In-place renaming complete.")


if __name__ == "__main__":
    main()
