"""Shared dataset-summary helpers (JSON + CSV writers)."""

from __future__ import annotations

import csv
import json
import os
from typing import Callable


def save_summaries(
    dataset_dir: str,
    all_runs_data: list,
    flatten_row: Callable[[dict], dict],
) -> None:
    """Write `summary.json` (full hierarchical log) and `summary.csv`
    (flat, ML-loader-friendly) for a dataset directory.

    `flatten_row` receives one metadata entry and returns the CSV row dict.
    """
    json_path = os.path.join(dataset_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(all_runs_data, f, indent=4)
    print(f"Summary JSON saved to: {json_path}")

    if not all_runs_data:
        return

    csv_path = os.path.join(dataset_dir, "summary.csv")
    flattened = [flatten_row(r) for r in all_runs_data]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
        writer.writeheader()
        writer.writerows(flattened)
    print(f"Summary CSV saved to: {csv_path}")
