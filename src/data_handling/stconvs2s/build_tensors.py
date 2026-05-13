#!/usr/bin/env python3
"""
Build STConvS2S-ready tensors for dengue forecasting.

Inputs:
  --grid    : data/processed/stations/RJ_grid_weekly.npz
  --cnes    : data/processed/cnes/STRJ2401.parquet
  --cases   : data/datasets/RJ_WEEKLY/dataset_ids.pickle
  --out-dir : data/datasets/RJ_STCONVS2S
  --lookback: 4  (weeks of historical met + cases context)
  --train-end, --val-end: ISO split dates

Outputs per split (train/val/test):
  X_met.npy    : float32 (n_samples, lookback, 9, H, W)
  X_cases.npy  : float32 (n_samples, lookback, n_units)
  y.npy        : float32 (n_samples, n_units)   log1p scale
  dates.npy    : str (n_samples,)  target week Monday
  unit_ids.npy : str (n_units,)
scaler.npz: StandardScaler params (fit on train only)
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────── constants ────────────────────────────────────────────
MET_CHANNELS   = ["TEM_AVG", "TEM_MIN", "TEM_MAX", "DEW_AVG", "HUM_AVG",
                  "PRECIP", "TEMP_RANGE", "WEEK_SIN", "WEEK_COS"]
N_MET          = len(MET_CHANNELS)         # 9
LOG1P_PRECIP   = True                      # log1p on PRECIP before z-scoring
PRECIP_IDX     = 5                         # channel index of PRECIP
WEEK_SIN_IDX   = 7                         # not normalized
WEEK_COS_IDX   = 8


# ─────────────────────── helpers ──────────────────────────────────────────────

def iso_week_monday(date_str):
    """Convert any date string to its ISO-week Monday as 'YYYY-MM-DD'."""
    d = pd.Timestamp(date_str)
    return (d - pd.Timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def add_derived_channels(grid, channels):
    """
    Append TEMP_RANGE, WEEK_SIN, WEEK_COS to grid.

    Parameters
    ----------
    grid     : (n_weeks, C, H, W)  ERA5/station merged grid
    channels : list[str]  length C

    Returns
    -------
    grid_ext     : (n_weeks, C+3, H, W)
    channels_ext : list[str]
    """
    n_weeks, C, H, W = grid.shape
    channels = list(channels)

    tem_max_idx = channels.index("TEM_MAX")
    tem_min_idx = channels.index("TEM_MIN")
    temp_range = (grid[:, tem_max_idx] - grid[:, tem_min_idx])[:, np.newaxis]  # (n_weeks,1,H,W)

    # WEEK_SIN/COS: broadcast over H, W
    week_sin = np.zeros((n_weeks, 1, H, W), dtype=np.float32)
    week_cos = np.zeros((n_weeks, 1, H, W), dtype=np.float32)

    # We need week dates to compute ISO week number — caller must pass them
    # Return placeholders; caller fills in after calling this function.
    grid_ext = np.concatenate([grid, temp_range, week_sin, week_cos], axis=1)
    channels_ext = channels + ["TEMP_RANGE", "WEEK_SIN", "WEEK_COS"]
    return grid_ext.astype(np.float32), channels_ext


def fill_week_trig(grid_ext, week_dates, week_sin_idx=7, week_cos_idx=8):
    """Fill WEEK_SIN/WEEK_COS channels from actual ISO week numbers."""
    for wi, wd in enumerate(week_dates):
        iso_week = pd.Timestamp(str(wd)).isocalendar().week
        angle = 2 * np.pi * iso_week / 52.0
        grid_ext[wi, week_sin_idx] = np.float32(np.sin(angle))
        grid_ext[wi, week_cos_idx] = np.float32(np.cos(angle))
    return grid_ext


# ─────────────────────── main logic ───────────────────────────────────────────

def build_cases_matrix(cases_dict, unit_ids, all_weeks):
    """
    Build dense cases matrix aligned to all_weeks and unit_ids.

    Returns
    -------
    cases_mat : (n_weeks, n_units) float32  raw counts
    """
    n_weeks  = len(all_weeks)
    n_units  = len(unit_ids)
    week_to_idx = {w: i for i, w in enumerate(all_weeks)}
    unit_to_idx = {u: i for i, u in enumerate(unit_ids)}

    cases_mat = np.zeros((n_weeks, n_units), dtype=np.float32)

    for split_name, split_data in cases_dict.items():
        dates_raw = split_data["DATE"]
        ids       = split_data["ID_UNIDADE"]
        y_true    = split_data["Y_TRUE"]
        for d, uid, y in zip(dates_raw, ids, y_true):
            week_mon = iso_week_monday(str(d)[:10])
            wi = week_to_idx.get(week_mon)
            ui = unit_to_idx.get(str(uid))
            if wi is not None and ui is not None:
                cases_mat[wi, ui] = float(y)

    return cases_mat


def build_windows(met_grid, cases_mat, all_weeks, lookback):
    """
    Create sliding windows of shape:
      X_met   : (n_samples, lookback, 9, H, W)
      X_cases : (n_samples, lookback, n_units)
      y       : (n_samples, n_units)   log1p scale
      dates   : (n_samples,)  target week strings

    Window: [t-lookback … t-1] → y[t]
    """
    n_weeks, C, H, W = met_grid.shape
    n_units = cases_mat.shape[1]

    X_met_list   = []
    X_cases_list = []
    y_list       = []
    dates_list   = []

    for t in range(lookback, n_weeks):
        X_met_list.append(met_grid[t - lookback:t])         # (lookback, C, H, W)
        X_cases_list.append(cases_mat[t - lookback:t])      # (lookback, n_units)
        y_list.append(np.log1p(cases_mat[t]))               # (n_units,)
        dates_list.append(all_weeks[t])

    X_met   = np.stack(X_met_list,   axis=0).astype(np.float32)
    X_cases = np.stack(X_cases_list, axis=0).astype(np.float32)
    y       = np.stack(y_list,       axis=0).astype(np.float32)
    dates   = np.array(dates_list)

    return X_met, X_cases, y, dates


def split_by_date(X_met, X_cases, y, dates, train_end, val_end):
    """Split arrays into train/val/test by target date."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts   = pd.Timestamp(val_end)
    dates_ts     = pd.to_datetime(dates)

    train_mask = dates_ts <= train_end_ts
    val_mask   = (dates_ts > train_end_ts) & (dates_ts <= val_end_ts)
    test_mask  = dates_ts > val_end_ts

    def _sel(mask):
        return (X_met[mask], X_cases[mask], y[mask], dates[mask])

    return _sel(train_mask), _sel(val_mask), _sel(test_mask)


def fit_scaler(X_met_train, X_cases_train, met_channels):
    """
    Fit StandardScaler on train split.
    Returns scalar parameters as dicts.

    PRECIP: log1p first, then StandardScaler.
    WEEK_SIN / WEEK_COS: not normalized.
    Cases: log1p first (already done in y), then StandardScaler.
    """
    # Met channels
    n_samples, lookback, C, H, W = X_met_train.shape
    met_flat = X_met_train.reshape(-1, C, H * W)  # (n_samples*lookback, C, H*W)

    means = np.zeros(C, dtype=np.float64)
    stds  = np.ones(C, dtype=np.float64)

    skip_norm = {WEEK_SIN_IDX, WEEK_COS_IDX}

    for ci in range(C):
        if ci in skip_norm:
            continue
        vals = met_flat[:, ci, :].ravel()
        if ci == PRECIP_IDX:
            vals = np.log1p(np.maximum(vals, 0))
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        means[ci] = float(valid.mean())
        stds[ci]  = float(valid.std()) if valid.std() > 0 else 1.0

    # Cases (already log1p applied in build_windows)
    cases_flat = X_cases_train.reshape(-1)  # already log1p
    cases_mean = float(cases_flat.mean())
    cases_std  = float(cases_flat.std()) if cases_flat.std() > 0 else 1.0

    return {
        "met_means":    means,
        "met_stds":     stds,
        "cases_mean":   cases_mean,
        "cases_std":    cases_std,
        "met_channels": np.array(met_channels),
    }


def apply_scaler(X_met, X_cases, scaler):
    """Apply fitted scaler to a split."""
    X_met   = X_met.copy()
    X_cases = X_cases.copy()

    means = scaler["met_means"]
    stds  = scaler["met_stds"]
    skip  = {WEEK_SIN_IDX, WEEK_COS_IDX}

    for ci in range(X_met.shape[2]):
        if ci in skip:
            continue
        if ci == PRECIP_IDX:
            X_met[:, :, ci] = np.log1p(np.maximum(X_met[:, :, ci], 0))
        X_met[:, :, ci] = (X_met[:, :, ci] - means[ci]) / stds[ci]

    # Cases: already log1p in build_windows
    X_cases = (X_cases - scaler["cases_mean"]) / scaler["cases_std"]

    return X_met.astype(np.float32), X_cases.astype(np.float32)


def save_split(out_dir: Path, name: str, X_met, X_cases, y, dates, unit_ids):
    split_dir = out_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "X_met.npy",    X_met)
    np.save(split_dir / "X_cases.npy",  X_cases)
    np.save(split_dir / "y.npy",        y)
    np.save(split_dir / "dates.npy",    dates)
    np.save(split_dir / "unit_ids.npy", unit_ids)
    log.info("[%s] X_met=%s  X_cases=%s  y=%s  dates=%s→%s",
             name, X_met.shape, X_cases.shape, y.shape, dates[0], dates[-1])


def main():
    parser = argparse.ArgumentParser(description="Build STConvS2S tensors")
    parser.add_argument("--grid",      required=True)
    parser.add_argument("--cnes",      required=True)
    parser.add_argument("--cases",     required=True)
    parser.add_argument("--out-dir",   required=True)
    parser.add_argument("--lookback",  type=int, default=4)
    parser.add_argument("--train-end", required=True, help="YYYY-MM-DD inclusive train end")
    parser.add_argument("--val-end",   required=True, help="YYYY-MM-DD inclusive val end")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load grid ──────────────────────────────────────────────────────────
    log.info("Loading grid from %s", args.grid)
    era5 = np.load(args.grid, allow_pickle=True)
    grid        = era5["era5_weekly"]    # (n_weeks, 6, H, W)
    week_dates  = list(era5["week_dates"])
    # Convert np.str_ to plain str
    era5_chs    = [str(c) for c in era5["channels"]]
    week_dates  = [str(d) for d in week_dates]
    # Rename RH_AVG -> HUM_AVG if present (ERA5 naming vs target naming)
    era5_chs = [("HUM_AVG" if c == "RH_AVG" else c) for c in era5_chs]
    H, W        = grid.shape[2], grid.shape[3]
    log.info("Grid: %d weeks, H=%d W=%d, channels=%s", len(week_dates), H, W, era5_chs)

    # ── Add derived channels ──────────────────────────────────────────────
    grid_ext, ext_channels = add_derived_channels(grid, era5_chs)
    grid_ext = fill_week_trig(grid_ext, week_dates)
    log.info("Extended channels: %s", ext_channels)
    assert ext_channels == MET_CHANNELS, f"Channel mismatch: {ext_channels} vs {MET_CHANNELS}"

    # ── Load cases ────────────────────────────────────────────────────────
    log.info("Loading cases from %s", args.cases)
    with open(args.cases, "rb") as f:
        cases_dict = pickle.load(f)

    # Determine unit IDs: intersection of cases and CNES
    log.info("Loading CNES from %s", args.cnes)
    cnes = pd.read_parquet(args.cnes)
    cnes_id_col = "CNES" if "CNES" in cnes.columns else cnes.columns[0]
    cnes_ids = set(cnes[cnes_id_col].astype(str).str.strip())

    all_case_ids = set()
    for split_data in cases_dict.values():
        all_case_ids.update(str(x) for x in split_data["ID_UNIDADE"])

    common_ids = sorted(all_case_ids & cnes_ids)
    if not common_ids:
        log.warning("No common IDs between cases (%d) and CNES (%d) — using all case IDs",
                    len(all_case_ids), len(cnes_ids))
        common_ids = sorted(all_case_ids)

    unit_ids = np.array(common_ids)
    n_units  = len(unit_ids)
    log.info("Units: %d (cases=%d, CNES=%d, common=%d)",
             n_units, len(all_case_ids), len(cnes_ids), len(common_ids))

    # ── Build dense cases matrix aligned to grid weeks ────────────────────
    cases_mat = build_cases_matrix(cases_dict, common_ids, week_dates)
    log.info("Cases matrix: %s  nonzero=%.1f%%",
             cases_mat.shape, 100 * (cases_mat > 0).mean())

    # ── Sliding windows ───────────────────────────────────────────────────
    X_met, X_cases, y, dates = build_windows(grid_ext, cases_mat, week_dates, args.lookback)
    log.info("Windows: X_met=%s  X_cases=%s  y=%s", X_met.shape, X_cases.shape, y.shape)

    # ── Split ─────────────────────────────────────────────────────────────
    (Xm_tr, Xc_tr, y_tr, d_tr), \
    (Xm_va, Xc_va, y_va, d_va), \
    (Xm_te, Xc_te, y_te, d_te) = split_by_date(
        X_met, X_cases, y, dates, args.train_end, args.val_end
    )
    log.info("Split sizes: train=%d val=%d test=%d", len(d_tr), len(d_va), len(d_te))

    # ── Fit scaler on train ───────────────────────────────────────────────
    scaler = fit_scaler(Xm_tr, Xc_tr, MET_CHANNELS)
    log.info("Scaler fitted: met_means=%s", np.round(scaler["met_means"], 3))

    # ── Normalize ─────────────────────────────────────────────────────────
    Xm_tr, Xc_tr = apply_scaler(Xm_tr, Xc_tr, scaler)
    Xm_va, Xc_va = apply_scaler(Xm_va, Xc_va, scaler)
    Xm_te, Xc_te = apply_scaler(Xm_te, Xc_te, scaler)

    # Cases in X_cases were log1p'd in build_windows; apply z-score
    # (already done inside apply_scaler for X_cases)

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez_compressed(
        out_dir / "scaler.npz",
        met_means=scaler["met_means"],
        met_stds=scaler["met_stds"],
        cases_mean=np.array([scaler["cases_mean"]]),
        cases_std=np.array([scaler["cases_std"]]),
        met_channels=scaler["met_channels"],
    )
    log.info("Scaler saved → %s", out_dir / "scaler.npz")

    save_split(out_dir, "train", Xm_tr, Xc_tr, y_tr, d_tr, unit_ids)
    save_split(out_dir, "val",   Xm_va, Xc_va, y_va, d_va, unit_ids)
    save_split(out_dir, "test",  Xm_te, Xc_te, y_te, d_te, unit_ids)

    log.info("All splits saved to %s", out_dir)


if __name__ == "__main__":
    main()
