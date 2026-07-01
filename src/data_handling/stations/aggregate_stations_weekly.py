#!/usr/bin/env python3
"""
Aggregate INMET + REDEMET daily malha1 files to weekly (ISO-week Monday) CSV.

Outputs: {out_dir}/RJ_stations_weekly.csv with columns:
  WEEK_DATE, STATION, NAME, LAT, LNG,
  TEM_AVG, TEM_MIN, TEM_MAX, HUM_AVG, DEW_AVG, RAIN

Usage:
  python src/data_handling/stations/aggregate_stations_weekly.py \
      --city           RJ \
      --inmet-malha1   data/raw/inmet/malha1_inmet_diario.csv \
      --redemet-malha1 data/raw/redemet/malha1_redemet_diario.csv \
      --out-dir        data/processed/stations
"""

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _iso_week_monday(dates: pd.Series) -> pd.Series:
    """Return the ISO-week Monday for each date."""
    dt = pd.to_datetime(dates)
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")


def load_inmet(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DT_MEDICAO"] = pd.to_datetime(df["DT_MEDICAO"], errors="coerce")
    df = df.dropna(subset=["DT_MEDICAO"])
    for col in ["TEM_MIN", "TEM_MAX", "TEM_AVG", "HUM_AVG", "DEW_AVG", "RAIN"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SOURCE"] = "INMET"
    return df


def load_redemet(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DT_MEDICAO"] = pd.to_datetime(df["DT_MEDICAO"], errors="coerce")
    df = df.dropna(subset=["DT_MEDICAO"])
    # REDEMET has no precipitation — leave as NaN
    if "RAIN" not in df.columns:
        df["RAIN"] = np.nan
    if "HUM_AVG" not in df.columns:
        df["HUM_AVG"] = np.nan
    for col in ["TEM_MIN", "TEM_MAX", "TEM_AVG", "HUM_AVG", "DEW_AVG", "RAIN"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SOURCE"] = "REDEMET"
    return df


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (station, ISO-week Monday) and aggregate met variables."""
    df = df.copy()
    df["WEEK_DATE"] = _iso_week_monday(df["DT_MEDICAO"])

    def _rain_sum(x):
        return x.sum(min_count=1)

    agg_spec = {}
    for col in ["TEM_AVG", "HUM_AVG", "DEW_AVG"]:
        if col in df.columns:
            agg_spec[col] = "mean"
    if "TEM_MIN" in df.columns:
        agg_spec["TEM_MIN"] = "min"
    if "TEM_MAX" in df.columns:
        agg_spec["TEM_MAX"] = "max"
    if "RAIN" in df.columns:
        agg_spec["RAIN"] = _rain_sum
    for meta_col in ["NAME", "LAT", "LNG", "SOURCE"]:
        if meta_col in df.columns:
            agg_spec[meta_col] = "first"

    grp = df.groupby(["STATION", "WEEK_DATE"], as_index=False).agg(agg_spec)
    grp["WEEK_DATE"] = grp["WEEK_DATE"].dt.strftime("%Y-%m-%d")
    return grp


def main():
    parser = argparse.ArgumentParser(description="Aggregate stations to weekly CSV")
    parser.add_argument("--city",            required=True)
    parser.add_argument("--inmet-malha1",    required=True)
    parser.add_argument("--redemet-malha1",  required=True)
    parser.add_argument("--out-dir",         default="data/processed/stations")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    log.info("Loading INMET daily from %s", args.inmet_malha1)
    inmet = load_inmet(args.inmet_malha1)
    log.info("  %d rows, %d stations: %s",
             len(inmet), inmet["STATION"].nunique(), sorted(inmet["STATION"].unique()))

    log.info("Loading REDEMET daily from %s", args.redemet_malha1)
    redemet = load_redemet(args.redemet_malha1)
    log.info("  %d rows, %d stations: %s",
             len(redemet), redemet["STATION"].nunique(), sorted(redemet["STATION"].unique()))

    combined = pd.concat([inmet, redemet], ignore_index=True)
    log.info("Combined: %d rows, %d stations", len(combined), combined["STATION"].nunique())

    weekly = aggregate_to_weekly(combined)
    log.info("Weekly: %d rows, %d stations, %d weeks",
             len(weekly), weekly["STATION"].nunique(), weekly["WEEK_DATE"].nunique())

    out_path = Path(args.out_dir) / f"{args.city}_stations_weekly.csv"
    weekly.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
