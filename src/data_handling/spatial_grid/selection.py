"""Construction and structural evaluation of metric epidemiological grids.

The target grid is intentionally independent from the ERA5 raster.  All
case-dependent masks and indicators in this module must receive training data
only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import Point, box
from shapely.prepared import prep


@dataclass(frozen=True)
class CandidateGrid:
    domain: str
    crs: str
    cell_size_m: float
    bounds: tuple[float, float, float, float]
    rows: int
    cols: int
    inclusion: str = "cell centroid within territory"

    @property
    def total_cells(self) -> int:
        return self.rows * self.cols

    def cell_id(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        return row * self.cols + col

    def manifest(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "crs": self.crs,
            "crs_wkt": CRS.from_user_input(self.crs).to_wkt(),
            "cell_size_m": self.cell_size_m,
            "bounds_projected_m": list(self.bounds),
            "rows": self.rows,
            "cols": self.cols,
            "total_cells": self.total_cells,
            "territorial_inclusion": self.inclusion,
            "row_orientation": "north-to-south",
            "column_orientation": "west-to-east",
        }


def build_grid(
    domain: str,
    crs: str,
    cell_size_km: float,
    territory,
) -> CandidateGrid:
    """Create a regular grid aligned outwards to the projected territory."""
    size = float(cell_size_km) * 1000.0
    if size <= 0:
        raise ValueError("cell_size_km must be positive")
    minx, miny, maxx, maxy = territory.bounds
    minx = np.floor(minx / size) * size
    miny = np.floor(miny / size) * size
    maxx = np.ceil(maxx / size) * size
    maxy = np.ceil(maxy / size) * size
    cols = int(round((maxx - minx) / size))
    rows = int(round((maxy - miny) / size))
    return CandidateGrid(domain, crs, size, (minx, miny, maxx, maxy), rows, cols)


def assign_points(
    grid: CandidateGrid, longitude: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project WGS84 points and return row, column and in-extent flag."""
    transform = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = transform.transform(longitude, latitude)
    minx, miny, maxx, maxy = grid.bounds
    col = np.floor((np.asarray(x) - minx) / grid.cell_size_m).astype(np.int64)
    row = np.floor((maxy - np.asarray(y)) / grid.cell_size_m).astype(np.int64)
    # Points exactly on the south/east edge belong to the last cell.
    col = np.where(col == grid.cols, grid.cols - 1, col)
    row = np.where(row == grid.rows, grid.rows - 1, row)
    valid = (row >= 0) & (row < grid.rows) & (col >= 0) & (col < grid.cols)
    return row, col, valid


def cell_frame(grid: CandidateGrid, territory) -> pd.DataFrame:
    minx, miny, maxx, maxy = grid.bounds
    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    prepared = prep(territory)
    rows: list[dict[str, Any]] = []
    for row in range(grid.rows):
        y1 = maxy - row * grid.cell_size_m
        y0 = y1 - grid.cell_size_m
        for col in range(grid.cols):
            x0 = minx + col * grid.cell_size_m
            x1 = x0 + grid.cell_size_m
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            lon, lat = transformer.transform(cx, cy)
            rows.append(
                {
                    "cell_id": row * grid.cols + col,
                    "row": row,
                    "col": col,
                    "center_x_m": cx,
                    "center_y_m": cy,
                    "longitude": lon,
                    "latitude": lat,
                    "territory_mask": bool(prepared.covers(Point(cx, cy))),
                    "territory_intersection_fraction": (
                        territory.intersection(box(x0, y0, x1, y1)).area
                        / (grid.cell_size_m**2)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or values.sum() == 0:
        return 0.0
    values = np.sort(np.maximum(values, 0))
    n = len(values)
    return float((2 * np.dot(np.arange(1, n + 1), values) / values.sum() - n - 1) / n)


def evaluate_candidate(
    grid: CandidateGrid,
    territory,
    cnes: pd.DataFrame,
    train_cases: pd.DataFrame,
    *,
    cnes_id: str = "CNES",
    case_unit: str = "ID_UNIDADE",
    case_count: str = "CASES",
    channels: int = 45,
    lookback: int = 28,
    horizon: int = 28,
    batch_size: int = 8,
    temporal_observations: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate one candidate using CNES and training cases only."""
    cells = cell_frame(grid, territory)
    n = grid.total_cells

    cnes = cnes.dropna(subset=["LAT", "LNG", cnes_id]).copy()
    # CNES identifiers are numeric.  Keeping them numeric avoids allocating
    # millions of Python strings when SINAN is large (DENG33 has >7M rows).
    cnes[cnes_id] = pd.to_numeric(cnes[cnes_id], errors="coerce")
    cnes = cnes.dropna(subset=[cnes_id])
    cnes[cnes_id] = cnes[cnes_id].astype(np.int64)
    row, col, valid = assign_points(
        grid, cnes["LNG"].to_numpy(float), cnes["LAT"].to_numpy(float)
    )
    cnes = cnes.loc[valid].copy()
    cnes["cell_id"] = grid.cell_id(row[valid], col[valid])
    per_cnes = cnes.groupby("cell_id")[cnes_id].nunique()

    required_case_columns = [case_unit, case_count]
    if "DT_NOTIFIC" in train_cases:
        required_case_columns.append("DT_NOTIFIC")
    cases = train_cases[required_case_columns].copy()
    cases[case_unit] = pd.to_numeric(cases[case_unit], errors="coerce")
    cases = cases.dropna(subset=[case_unit])
    cases[case_unit] = cases[case_unit].astype(np.int64)
    cases[case_count] = pd.to_numeric(cases[case_count], errors="coerce").fillna(0)
    cases = cases[cases[case_count] > 0].merge(
        cnes[[cnes_id, "cell_id"]].drop_duplicates(cnes_id),
        left_on=case_unit,
        right_on=cnes_id,
        how="inner",
    )
    per_cases = cases.groupby("cell_id")[case_count].sum()
    if "DT_NOTIFIC" in cases:
        cases["DT_NOTIFIC"] = pd.to_datetime(cases["DT_NOTIFIC"], errors="coerce").dt.normalize()
        nonzero_cell_times = cases.groupby(["DT_NOTIFIC", "cell_id"]).size().shape[0]
        observed_times = cases["DT_NOTIFIC"].nunique()
    else:
        nonzero_cell_times = 0
        observed_times = 0

    cells["cnes_count"] = cells["cell_id"].map(per_cnes).fillna(0).astype(int)
    cells["train_cases"] = cells["cell_id"].map(per_cases).fillna(0).astype(float)
    cells["cnes_mask"] = cells["cnes_count"] > 0
    cells["train_active_mask"] = cells["train_cases"] > 0
    territory_cells = cells["territory_mask"]
    occupied = cells.loc[cells["cnes_mask"], "cnes_count"].to_numpy()
    active_cases = cells.loc[cells["train_active_mask"], "train_cases"].to_numpy()

    unique_cnes = cnes[cnes_id].nunique()
    shared = int(
        cells.loc[cells["cnes_count"] > 1, "cnes_count"].sum()
    )
    bytes_sample = (lookback * channels + horizon) * n * 4
    total_t = int(temporal_observations or observed_times)
    territory_cell_count = int(territory_cells.sum())
    possible_cell_times = territory_cell_count * total_t
    metrics = {
        **grid.manifest(),
        "cells_inside_territory": int(territory_cells.sum()),
        "cells_outside_or_maritime": int((~territory_cells).sum()),
        "outside_or_maritime_pct": float(100 * (~territory_cells).mean()),
        "cells_with_cnes": int(cells["cnes_mask"].sum()),
        "cells_with_train_cases": int(cells["train_active_mask"].sum()),
        "epidemiologically_active_pct_of_territory": float(
            100 * cells.loc[territory_cells, "train_active_mask"].mean()
            if territory_cells.any()
            else 0
        ),
        "cnes_total_in_extent": int(unique_cnes),
        "cnes_per_occupied_cell_mean": float(occupied.mean() if len(occupied) else 0),
        "cnes_per_cell_max": int(occupied.max() if len(occupied) else 0),
        "establishments_sharing_cell_pct": float(
            100 * shared / unique_cnes if unique_cnes else 0
        ),
        "train_cases_total": float(cells["train_cases"].sum()),
        "train_cases_per_active_cell_mean": float(
            active_cases.mean() if len(active_cases) else 0
        ),
        "train_cases_per_active_cell_median": float(
            np.median(active_cases) if len(active_cases) else 0
        ),
        "train_cases_per_cell_max": float(active_cases.max() if len(active_cases) else 0),
        "always_zero_cells_pct": float(100 * (~cells["train_active_mask"]).mean()),
        "zero_inflation_cell_time_pct": float(
            100 * (1 - nonzero_cell_times / possible_cell_times)
            if possible_cell_times
            else 0
        ),
        "sparse_active_cells_le_5_cases": int((active_cases <= 5).sum()),
        "cnes_gini_all_cells": _gini(cells["cnes_count"].to_numpy()),
        "train_cases_gini_all_cells": _gini(cells["train_cases"].to_numpy()),
        "estimated_bytes_per_sample_float32": int(bytes_sample),
        "estimated_mib_per_sample_float32": float(bytes_sample / 2**20),
        "estimated_mib_per_batch_float32": float(bytes_sample * batch_size / 2**20),
        "smoke_epoch_seconds": None,
        "temporal_observations": total_t,
        "territory_cells_per_temporal_observation": float(
            territory_cells.sum() / total_t if total_t else 0
        ),
        "land_mask_policy": "same official administrative geometry as territory",
        "selection_data_policy": "CNES metadata plus training cases only; validation/test excluded",
    }
    masks = {
        "territory": cells["territory_mask"].to_numpy().reshape(grid.rows, grid.cols),
        "land": cells["territory_mask"].to_numpy().reshape(grid.rows, grid.cols),
        "cnes": cells["cnes_mask"].to_numpy().reshape(grid.rows, grid.cols),
        "train_active": cells["train_active_mask"].to_numpy().reshape(grid.rows, grid.cols),
        "cnes_count": cells["cnes_count"].to_numpy().reshape(grid.rows, grid.cols),
    }
    return metrics, cells, masks
