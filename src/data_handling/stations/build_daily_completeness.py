#!/usr/bin/env python3
"""Rebuild daily RJ station products with channel-specific completeness."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def metadata(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame.STATION.astype(str).str.startswith(prefix)].drop_duplicates("STATION").set_index("STATION")[["LAT", "LNG"]]


def web_daily(zip_path: Path, existing: Path) -> pd.DataFrame:
    meta = metadata(existing, "WS_")
    pattern = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}-\d{2}\s+(?P<m15>\S+).*\s+(?P<station>\d+)\s*$")
    records = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match_id = re.search(r"ID_ESTACAO_(\d+)\.txt$", name)
            if not match_id: continue
            station = f"WS_{int(match_id.group(1))}"
            if station not in meta.index: continue
            daily = {}
            with archive.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace"); next(text, None)
                for line in text:
                    match = pattern.search(line)
                    if not match: continue
                    try: value = float(match.group("m15").replace(",", "."))
                    except ValueError: continue
                    if value >= 0:
                        item = daily.setdefault(match.group("date"), [0.0, 0]); item[0] += value; item[1] += 1
            coord = meta.loc[station]
            records.extend((day, station, coord.LAT, coord.LNG, value[0], value[1]) for day, value in daily.items())
    frame = pd.DataFrame(records, columns=["DATE", "STATION", "LAT", "LNG", "RAIN", "OBS_VALID_RAIN"])
    frame["OBS_EXPECTED_RAIN"] = 96; frame["COMPLETENESS_RAIN"] = frame.OBS_VALID_RAIN / 96
    return frame


def alert_daily(zip_path: Path, existing: Path, weekly_manifest: Path) -> pd.DataFrame:
    meta = metadata(existing, "AR_")
    mapping = json.loads(weekly_manifest.read_text(encoding="utf-8"))["alertario_slug_mapping"]
    file_pattern = re.compile(r"(?:^|/)(.+)_(\d{6})_Plv\.txt$", re.IGNORECASE)
    chunks = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match = file_pattern.search(Path(name).name)
            if not match or match.group(1).lower() not in mapping: continue
            station = mapping[match.group(1).lower()]
            with archive.open(name) as raw:
                lines = io.TextIOWrapper(raw, encoding="utf-8", errors="replace").read().splitlines()
            header = next((line for line in lines[:10] if "min" in line.lower() and "Dia" in line), "")
            minute_labels = [int(value) for value in re.findall(r"(\d{2})\s*min", header, flags=re.I)]
            if not minute_labels:
                continue
            interval = min(minute_labels)
            value_index = minute_labels.index(interval)
            records = []
            for line in lines[5:]:
                parts = line.split()
                if len(parts) < 3 or not re.fullmatch(r"\d{2}/\d{2}/\d{4}", parts[0]):
                    continue
                values = parts[2:]
                if values and not re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?|ND", values[0], flags=re.I):
                    values = values[1:]  # optional HBV marker
                if value_index >= len(values):
                    continue
                raw_value = values[value_index].replace(",", ".")
                try: rain = float(raw_value)
                except ValueError: rain = np.nan
                if rain < 0: rain = np.nan
                date = pd.to_datetime(parts[0] + " " + parts[1], format="%d/%m/%Y %H:%M:%S", errors="coerce")
                records.append((date.normalize() if pd.notna(date) else pd.NaT, rain))
            frame = pd.DataFrame(records, columns=["DATE", "RAIN"]).dropna(subset=["DATE"])
            frame["VALID"] = frame.RAIN.notna().astype(int)
            daily = frame.groupby("DATE", as_index=False).agg(RAIN=("RAIN", lambda x: x.sum(min_count=1)), OBS_VALID_RAIN=("VALID", "sum"))
            daily["OBS_EXPECTED_RAIN"] = 1440 // interval
            coord = meta.loc[station]; daily["STATION"] = station; daily["LAT"] = coord.LAT; daily["LNG"] = coord.LNG
            chunks.append(daily)
    result = pd.concat(chunks, ignore_index=True).groupby(["DATE", "STATION", "LAT", "LNG"], as_index=False).agg(
        RAIN=("RAIN", "sum"), OBS_VALID_RAIN=("OBS_VALID_RAIN", "sum"), OBS_EXPECTED_RAIN=("OBS_EXPECTED_RAIN", "sum"))
    result["COMPLETENESS_RAIN"] = result.OBS_VALID_RAIN / result.OBS_EXPECTED_RAIN
    return result


def inmet_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).rename(columns={"CD_ESTACAO": "STATION", "DT_MEDICAO": "DATE", "TEM_INS": "TEM_AVG",
        "CHUVA": "RAIN", "VL_LATITUDE": "LAT", "VL_LONGITUDE": "LNG"})
    frame["STATION"] = "INMET_" + frame.STATION.astype(str); frame["DATE"] = pd.to_datetime(frame.DATE).dt.normalize()
    variables = ["TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN"]
    for variable in variables:
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce"); frame[f"VALID_{variable}"] = frame[variable].notna().astype(int)
    frame.loc[frame.RAIN < 0, "RAIN"] = np.nan
    counts_per_day = frame.groupby(["DATE", "STATION"]).size()
    expected = 24 if float(counts_per_day.median()) > 1 else 1
    functions = {"TEM_AVG": "mean", "TEM_MIN": "min", "TEM_MAX": "max", "RAIN": lambda x: x.sum(min_count=1),
                 **{f"VALID_{v}": "sum" for v in variables}}
    daily = frame.groupby(["DATE", "STATION", "LAT", "LNG"], as_index=False).agg(functions)
    for variable in variables:
        daily[f"OBS_VALID_{variable}"] = daily.pop(f"VALID_{variable}")
        daily[f"OBS_EXPECTED_{variable}"] = expected
        daily[f"COMPLETENESS_{variable}"] = daily[f"OBS_VALID_{variable}"] / expected
    return daily


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--web-zip", required=True); p.add_argument("--alert-zip", required=True)
    p.add_argument("--inmet", required=True); p.add_argument("--weekly-dir", required=True); p.add_argument("--out-dir", required=True)
    args = p.parse_args(); weekly, out = Path(args.weekly_dir), Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "websirenes": web_daily(Path(args.web_zip), weekly / "websirenes_weekly_completeness.csv"),
        "alertario": alert_daily(Path(args.alert_zip), weekly / "alertario_weekly_completeness.csv", weekly / "manifest.json"),
        "inmet": inmet_daily(Path(args.inmet)),
    }
    report = {}
    for name, frame in outputs.items():
        frame = frame.sort_values(["DATE", "STATION"]); frame.to_csv(out / f"{name}_daily_completeness.csv", index=False)
        report[name] = {"rows": len(frame), "stations": int(frame.STATION.nunique()), "start": str(frame.DATE.min()), "end": str(frame.DATE.max())}
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report))


if __name__ == "__main__": main()
