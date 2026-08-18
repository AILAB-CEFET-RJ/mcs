#!/usr/bin/env python3
"""Build weather-independent weekly dengue targets on a candidate RJ grid.

The SINAN parquet predicate is pushed down so records after ``--end`` are not
materialised. Outputs preserve raw counts, CNES/territory masks and ISO-year
split metadata; no scaling or lag engineering is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..spatial_grid.selection import CandidateGrid, assign_points
except ImportError:
    from pathlib import Path as _Path
    import sys
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from data_handling.spatial_grid.selection import CandidateGrid, assign_points


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", required=True)
    parser.add_argument("--grid-manifest", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--cnes", required=True)
    parser.add_argument("--sinan", required=True)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    paths = {name: Path(getattr(args, name.replace("-", "_"))) for name in
             ("cells", "grid_manifest", "masks", "cnes", "sinan")}
    manifest = json.loads(paths["grid_manifest"].read_text(encoding="utf-8"))
    grid = CandidateGrid(
        domain=manifest["domain"], crs=manifest["crs"],
        cell_size_m=float(manifest["cell_size_m"]),
        bounds=tuple(manifest["bounds_projected_m"]),
        rows=int(manifest["rows"]), cols=int(manifest["cols"]),
    )
    cells = pd.read_csv(paths["cells"]).sort_values(["row", "col"])
    if len(cells) != grid.total_cells:
        raise RuntimeError("Cell table and grid manifest disagree")
    saved_masks = np.load(paths["masks"])
    territory_mask = saved_masks["territory"].astype(bool)

    cnes = pd.read_parquet(paths["cnes"], columns=["CNES", "LAT", "LNG"]).copy()
    cnes["CNES"] = normalize_id(cnes["CNES"])
    cnes["LAT"] = pd.to_numeric(cnes["LAT"], errors="coerce")
    cnes["LNG"] = pd.to_numeric(cnes["LNG"], errors="coerce")
    cnes = cnes.dropna(subset=["LAT", "LNG"]).drop_duplicates("CNES")
    row, col, valid = assign_points(grid, cnes.LNG.to_numpy(float), cnes.LAT.to_numpy(float))
    if not valid.all():
        raise RuntimeError(f"CNES outside candidate grid extent: {int((~valid).sum())}")
    cnes["row"], cnes["col"] = row, col
    cnes["cell_id"] = grid.cell_id(row, col)
    cnes_count = np.zeros((grid.rows, grid.cols), dtype=np.int16)
    counts = cnes.groupby(["row", "col"]).CNES.nunique()
    for (r, c), value in counts.items():
        cnes_count[int(r), int(c)] = int(value)
    cnes_mask = cnes_count > 0
    # The experimental universe is the same 1,193 code-filtered CNES units as
    # the tabular arm. The official territory remains a structural channel,
    # but must not silently remove a unit because its recorded coordinate or
    # the candidate cell centroid falls outside the polygon.
    territory_cnes_mask = territory_mask & cnes_mask
    observation_mask = cnes_mask.copy()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    filters = [("CASES", ">", 0), ("DT_NOTIFIC", ">=", start), ("DT_NOTIFIC", "<=", end)]
    cases = pd.read_parquet(paths["sinan"], columns=["ID_UNIDADE", "DT_NOTIFIC", "CASES"], filters=filters)
    cases["ID_UNIDADE"] = normalize_id(cases["ID_UNIDADE"])
    cases["DT_NOTIFIC"] = pd.to_datetime(cases["DT_NOTIFIC"], errors="coerce")
    cases["CASES"] = pd.to_numeric(cases["CASES"], errors="coerce").fillna(0)
    cases = cases.dropna(subset=["DT_NOTIFIC"])
    source_total_before_cnes = float(cases.CASES.sum())
    mapping = cnes[["CNES", "row", "col", "cell_id"]].rename(columns={"CNES": "ID_UNIDADE"})
    cases = cases.merge(mapping, on="ID_UNIDADE", how="inner", validate="many_to_one")
    source_total = float(cases.CASES.sum())
    cases["WEEK_DATE"] = (
        cases.DT_NOTIFIC - pd.to_timedelta(cases.DT_NOTIFIC.dt.weekday, unit="D")
    ).dt.normalize()
    dates = pd.date_range(
        start - pd.Timedelta(days=start.weekday()),
        end - pd.Timedelta(days=end.weekday()), freq="W-MON",
    )
    date_index = {value: index for index, value in enumerate(dates)}
    target = np.zeros((len(dates), grid.rows, grid.cols), dtype=np.float32)
    grouped = cases.groupby(["WEEK_DATE", "row", "col"], as_index=False).CASES.sum()
    for item in grouped.itertuples(index=False):
        target[date_index[pd.Timestamp(item.WEEK_DATE)], int(item.row), int(item.col)] += float(item.CASES)
    if not np.isclose(target.sum(dtype=np.float64), source_total):
        raise RuntimeError("Case conservation failure during spatial/weekly aggregation")

    iso_year = dates.isocalendar().year.to_numpy()
    split_masks = {
        "train": (iso_year >= 2014) & (iso_year <= 2020),
        "validation": iso_year == 2021,
        "development": iso_year == 2022,
    }
    if sum(int(mask.sum()) for mask in split_masks.values()) != len(dates):
        raise RuntimeError("Weekly dates not fully and uniquely assigned to development splits")

    outside_territory = cnes.loc[~territory_mask[cnes.row, cnes.col]]
    cases_outside_territory = float(
        cases.loc[~territory_mask[cases.row, cases.col], "CASES"].sum()
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "cases_weekly.npy", target)
    np.save(out / "week_dates.npy", dates.to_numpy(dtype="datetime64[D]"))
    np.savez_compressed(out / "masks.npz", territory=territory_mask, cnes=cnes_mask,
                        territory_cnes=territory_cnes_mask,
                        observation=observation_mask, cnes_count=cnes_count)
    cnes[["CNES", "LAT", "LNG", "row", "col", "cell_id"]].to_parquet(
        out / "cnes_cell_mapping.parquet", index=False
    )
    for name, mask in split_masks.items():
        np.save(out / f"{name}_indices.npy", np.flatnonzero(mask))

    report = {
        "status": "epidemiological_base_no_weather_no_scaling_no_lags",
        "test_2023_loaded": False,
        "requested_interval": [args.start, args.end],
        "weekly_label": "Monday; aggregation Monday-Sunday",
        "first_week_partial_context": str(dates[0].date()) if dates[0] < start else None,
        "crs": grid.crs, "cell_size_km": grid.cell_size_m / 1000,
        "grid_shape": [grid.rows, grid.cols], "total_cells": grid.total_cells,
        "territory_cells": int(territory_mask.sum()), "cnes_cells": int(cnes_mask.sum()),
        "observation_mask_policy": "all cells containing one of the 1,193 experimental CNES units",
        "observation_cells": int(observation_mask.sum()),
        "territory_and_cnes_cells_for_audit": int(territory_cnes_mask.sum()),
        "cnes_units": int(cnes.CNES.nunique()),
        "cnes_units_in_observation_mask": int(cnes.CNES.nunique()),
        "cnes_units_outside_centroid_territory_mask": int(outside_territory.CNES.nunique()),
        "source_cases_before_cnes_filter": source_total_before_cnes,
        "source_cases_after_1193_cnes_filter": source_total,
        "aggregated_cases": float(target.sum(dtype=np.float64)),
        "cases_excluded_by_observation_mask": 0.0,
        "cases_outside_centroid_territory_mask_for_audit": cases_outside_territory,
        "splits": {
            name: {"weeks": int(mask.sum()), "start": str(dates[mask].min().date()),
                   "end": str(dates[mask].max().date()),
                   "cases": float(target[mask].sum(dtype=np.float64))}
            for name, mask in split_masks.items()
        },
        "inputs": {name: {"path": str(path), "sha256": hash_file(path)}
                   for name, path in paths.items()},
    }
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
