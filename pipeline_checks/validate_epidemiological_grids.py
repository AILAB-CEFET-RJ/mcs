#!/usr/bin/env python3
"""Cross-check weather-independent epidemiological grid artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path: Path):
    return {
        "path": path,
        "cases": np.load(path / "cases_weekly.npy", mmap_mode="r"),
        "dates": np.load(path / "week_dates.npy"),
        "masks": np.load(path / "masks.npz"),
        "manifest": json.loads((path / "manifest.json").read_text(encoding="utf-8")),
        "mapping": pd.read_parquet(path / "cnes_cell_mapping.parquet"),
        "indices": {name: np.load(path / f"{name}_indices.npy")
                    for name in ("train", "validation", "development")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    artifacts = [load(Path(value)) for value in args.artifact]
    reference_dates = artifacts[0]["dates"]
    reference_weekly = artifacts[0]["cases"].sum(axis=(1, 2), dtype=np.float64)
    rows = []
    for artifact in artifacts:
        cases, dates, masks = artifact["cases"], artifact["dates"], artifact["masks"]
        assert np.array_equal(dates, reference_dates), "Candidate dates differ"
        assert pd.Timestamp(dates.max()) <= pd.Timestamp("2022-12-31"), "2023 entered artifact"
        assert np.array_equal(cases.sum(axis=(1, 2), dtype=np.float64), reference_weekly), \
            "Weekly totals differ across grids"
        observation = masks["observation"].astype(bool)
        assert np.all(cases[:, ~observation] == 0), "Cases found outside CNES observation mask"
        assert artifact["mapping"].CNES.nunique() == 1193
        assert int(masks["cnes_count"].sum()) == 1193
        all_indices = np.concatenate(list(artifact["indices"].values()))
        assert np.array_equal(np.sort(all_indices), np.arange(len(dates))), "Splits overlap or omit weeks"
        assert artifact["manifest"]["test_2023_loaded"] is False
        rows.append({
            "artifact": str(artifact["path"]),
            "cell_size_km": artifact["manifest"]["cell_size_km"],
            "weeks": len(dates), "first_week": str(dates.min()), "last_week": str(dates.max()),
            "grid_rows": cases.shape[1], "grid_cols": cases.shape[2],
            "observation_cells": int(observation.sum()), "cnes_units": 1193,
            "cases_total": float(cases.sum(dtype=np.float64)),
            "cases_outside_observation": float(cases[:, ~observation].sum(dtype=np.float64)),
            "train_cases": float(cases[artifact["indices"]["train"]].sum(dtype=np.float64)),
            "validation_cases": float(cases[artifact["indices"]["validation"]].sum(dtype=np.float64)),
            "development_cases": float(cases[artifact["indices"]["development"]].sum(dtype=np.float64)),
            "test_2023_loaded": False,
        })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps({"status": "PASS", "checks": [
            "identical weekly totals across candidates", "all cases inside CNES mask",
            "1,193 CNES preserved", "splits exhaustive and disjoint", "2023 absent"
        ], "artifacts": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
