#!/usr/bin/env python3
"""Rebuild weekly observational sources with per-channel completeness."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def monday(values):
    values = pd.to_datetime(values, errors="coerce")
    return (values - pd.to_timedelta(values.dt.weekday, unit="D")).dt.normalize()


def existing_metadata(path, prefix):
    frame = pd.read_csv(path)
    frame = frame[frame.STATION.astype(str).str.startswith(prefix)]
    return frame.drop_duplicates("STATION").set_index("STATION")[["LAT", "LNG"]]


def build_web(zip_path, existing_path):
    metadata = existing_metadata(existing_path, "WS_")
    pattern = re.compile(
        r"(?P<date>\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}-\d{2}\s+"
        r"(?P<m15>\S+).*\s+(?P<station>\d+)\s*$"
    )
    records = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match_id = re.search(r"ID_ESTACAO_(\d+)\.txt$", name)
            if not match_id:
                continue
            station = f"WS_{int(match_id.group(1))}"
            if station not in metadata.index:
                continue
            daily = {}
            with archive.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace")
                next(text, None)
                for line in text:
                    match = pattern.search(line)
                    if not match:
                        continue
                    value = match.group("m15").replace(",", ".")
                    try:
                        rain = float(value)
                    except ValueError:
                        continue
                    if rain >= 0:
                        item = daily.setdefault(match.group("date"), [0.0, 0])
                        item[0] += rain
                        item[1] += 1
            coord = metadata.loc[station]
            records.extend((day, station, coord.LAT, coord.LNG, value[0], value[1])
                           for day, value in daily.items())
    frame = pd.DataFrame(records, columns=["DATE", "STATION", "LAT", "LNG", "RAIN", "OBS_VALID_RAIN"])
    frame["DATE"] = monday(frame.DATE)
    weekly = frame.groupby(["DATE", "STATION", "LAT", "LNG"], as_index=False).agg(
        RAIN=("RAIN", "sum"), OBS_VALID_RAIN=("OBS_VALID_RAIN", "sum"))
    weekly["OBS_EXPECTED_RAIN"] = 7 * 96
    weekly["COMPLETENESS_RAIN"] = weekly.OBS_VALID_RAIN / weekly.OBS_EXPECTED_RAIN
    return weekly


def build_alert_raw(zip_path):
    file_pattern = re.compile(r"(?:^|/)(.+)_(\d{6})_Plv\.txt$", re.IGNORECASE)
    chunks = []
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if file_pattern.search(Path(name).name)]
        for name in names:
            slug = file_pattern.search(Path(name).name).group(1).lower()
            with archive.open(name) as raw:
                frame = pd.read_csv(raw, sep=r"\s{2,}", engine="python", skiprows=5,
                    names=["day", "time", "m15", "h01", "h04", "h24", "h96"],
                    encoding="utf-8")
            frame["DATE"] = pd.to_datetime(
                frame.day.astype(str) + " " + frame.time.astype(str),
                format="%d/%m/%Y %H:%M:%S", errors="coerce")
            frame["RAIN"] = pd.to_numeric(frame.m15, errors="coerce")
            frame.loc[frame.RAIN < 0, "RAIN"] = np.nan
            frame = frame.dropna(subset=["DATE"])
            frame["VALID"] = frame.RAIN.notna().astype(int)
            frame["RAIN"] = frame.RAIN.fillna(0)
            frame["DATE"] = monday(frame.DATE)
            chunks.append(frame.groupby("DATE", as_index=False).agg(
                RAIN=("RAIN", "sum"), OBS_VALID_RAIN=("VALID", "sum")).assign(SLUG=slug))
    return pd.concat(chunks, ignore_index=True).groupby(["DATE", "SLUG"], as_index=False).agg(
        RAIN=("RAIN", "sum"), OBS_VALID_RAIN=("OBS_VALID_RAIN", "sum"))


def match_alert(raw, existing_path):
    existing = pd.read_csv(existing_path)
    existing["DATE"] = pd.to_datetime(existing.DATE)
    candidates = []
    for slug, left in raw.groupby("SLUG"):
        for station, right in existing.groupby("STATION"):
            joined = left[["DATE", "RAIN"]].merge(right[["DATE", "RAIN"]], on="DATE", suffixes=("_raw", "_old"))
            if len(joined) < 20:
                continue
            error = (joined.RAIN_raw - joined.RAIN_old).abs()
            candidates.append((float(error.mean()), float(error.max()), -len(joined), slug, station))
    mapping, used_slugs, used_stations = {}, set(), set()
    for mean_error, max_error, neg_n, slug, station in sorted(candidates):
        if slug in used_slugs or station in used_stations:
            continue
        if max_error > 1e-6:
            continue
        mapping[slug] = station
        used_slugs.add(slug); used_stations.add(station)
    metadata = existing.drop_duplicates("STATION").set_index("STATION")[["LAT", "LNG"]]
    matched = raw[raw.SLUG.isin(mapping)].copy()
    matched["STATION"] = matched.SLUG.map(mapping)
    matched = matched.join(metadata, on="STATION")
    matched["OBS_EXPECTED_RAIN"] = 7 * 96
    matched["COMPLETENESS_RAIN"] = matched.OBS_VALID_RAIN / matched.OBS_EXPECTED_RAIN
    return matched.drop(columns="SLUG"), mapping


def build_inmet(path):
    frame = pd.read_parquet(path).rename(columns={"CD_ESTACAO": "STATION", "DT_MEDICAO": "DATE",
        "CHUVA": "RAIN", "VL_LATITUDE": "LAT", "VL_LONGITUDE": "LNG"})
    frame["STATION"] = "INMET_" + frame.STATION.astype(str)
    frame["DATE"] = monday(frame.DATE)
    variables = ["TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN"]
    for variable in variables:
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
    frame.loc[frame.RAIN < 0, "RAIN"] = np.nan
    for variable in variables:
        frame[f"VALID_{variable}"] = frame[variable].notna().astype(int)
    functions = {"TEM_AVG": "mean", "TEM_MIN": "min", "TEM_MAX": "max",
                 "RAIN": lambda values: values.sum(min_count=1),
                 **{f"VALID_{variable}": "sum" for variable in variables}}
    weekly = frame.groupby(["DATE", "STATION", "LAT", "LNG"], as_index=False).agg(functions)
    for variable in variables:
        weekly[f"OBS_VALID_{variable}"] = weekly.pop(f"VALID_{variable}")
        weekly[f"OBS_EXPECTED_{variable}"] = 7
        weekly[f"COMPLETENESS_{variable}"] = weekly[f"OBS_VALID_{variable}"] / 7
    return weekly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--web-zip", required=True)
    parser.add_argument("--alert-zip", required=True)
    parser.add_argument("--inmet", required=True)
    parser.add_argument("--existing-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    existing = Path(args.existing_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    web = build_web(Path(args.web_zip), existing / "websirenes_weekly.csv")
    alert_raw = build_alert_raw(Path(args.alert_zip))
    alert, mapping = match_alert(alert_raw, existing / "alertario_weekly.csv")
    inmet = build_inmet(Path(args.inmet))
    outputs = {"websirenes": web, "alertario": alert, "inmet": inmet}
    for name, frame in outputs.items():
        frame.sort_values(["DATE", "STATION"]).to_csv(out / f"{name}_weekly_completeness.csv", index=False)
    report = {name: {"rows": len(frame), "stations": frame.STATION.nunique(),
                     "start": str(frame.DATE.min().date()), "end": str(frame.DATE.max().date())}
              for name, frame in outputs.items()}
    report["alertario_slug_mapping"] = mapping
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
