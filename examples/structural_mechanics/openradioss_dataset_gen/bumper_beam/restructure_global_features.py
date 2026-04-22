"""
Convert the bumper-beam `summary.json` (list of run metadata) into the
per-run dict format consumed by the crash recipe's datapipe
(`global_features.json` keyed by run ID).

Output keys (one row per run):
    geo_scale_x, geo_scale_y, geo_scale_z,
    velocity_x, velocity_y, velocity_z,
    thickness_scale,
    rwall_diameter,
    rwall_origin_x, rwall_origin_y, rwall_origin_z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _as_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def convert_global_features(input_json: Path, output_json: Path) -> None:
    with input_json.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Input JSON must be a list of entries")

    out: dict[str, dict[str, float]] = {}
    for entry in data:
        run_id = entry.get("run_id")
        if run_id is None:
            raise KeyError("Missing run_id in entry")
        if run_id in out:
            raise ValueError(f"Duplicate run_id: {run_id}")

        p = entry.get("parameters") or {}
        geo = p.get("geometry_scale") or (1.0, 1.0, 1.0)
        vel = p.get("velocity_vector") or (0.0, 0.0, 0.0)
        orig = p.get("rwall_origin") or (0.0, 0.0, 0.0)

        out[run_id] = {
            "geo_scale_x": _as_float(geo[0], 1.0),
            "geo_scale_y": _as_float(geo[1], 1.0),
            "geo_scale_z": _as_float(geo[2], 1.0),
            "velocity_x": _as_float(vel[0]),
            "velocity_y": _as_float(vel[1]),
            "velocity_z": _as_float(vel[2]),
            "thickness_scale": _as_float(p.get("thickness_scale"), 1.0),
            "rwall_diameter": _as_float(p.get("rwall_diameter")),
            "rwall_origin_x": _as_float(orig[0]),
            "rwall_origin_y": _as_float(orig[1]),
            "rwall_origin_z": _as_float(orig[2]),
        }

    with output_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} runs to {output_json}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "dataset" / "summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "global_features.json",
    )
    args = parser.parse_args()
    convert_global_features(args.input, args.output)


if __name__ == "__main__":
    main()
