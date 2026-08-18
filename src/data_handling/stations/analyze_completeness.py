#!/usr/bin/env python3
"""Quantify observational retention under pre-specified completeness cutoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default="0.5,0.75,0.9")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    directory = Path(args.input_dir)
    rows = []
    for path in sorted(directory.glob("*_weekly_completeness.csv")):
        network = path.name.split("_weekly_completeness")[0].upper()
        frame = pd.read_csv(path)
        frame["DATE"] = pd.to_datetime(frame.DATE)
        for completeness_column in [column for column in frame if column.startswith("COMPLETENESS_")]:
            variable = completeness_column.removeprefix("COMPLETENESS_")
            if variable not in frame:
                continue
            usable = frame[frame[variable].notna() & frame[completeness_column].notna()].copy()
            for threshold in thresholds:
                kept = usable[usable[completeness_column] >= threshold]
                rows.append({
                    "network": network, "variable": variable, "threshold": threshold,
                    "available_rows": len(usable), "retained_rows": len(kept),
                    "retained_pct": 100 * len(kept) / len(usable) if len(usable) else 0,
                    "stations_available": usable.STATION.nunique(),
                    "stations_retained": kept.STATION.nunique(),
                    "weeks_available": usable.DATE.nunique(), "weeks_retained": kept.DATE.nunique(),
                    "median_completeness": float(usable[completeness_column].median()),
                    "p10_completeness": float(usable[completeness_column].quantile(.1)),
                })
    result = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    summary = {"thresholds": thresholds, "rows": rows,
               "selection_rule": "prefer 0.75 if it retains >=80% of available rows and all networks/stations required; otherwise document lower cutoff",
               "dengue_outcomes_used": False, "test_2023_used": False}
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
