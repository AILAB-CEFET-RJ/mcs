#!/usr/bin/env python3
"""Prepare RJ observational weather sources in the canonical fusion schema.

WebSirenes and AlertaRio are read directly from their ZIP archives.  Their
15-minute increment (m15) is summed; overlapping h01/h04/h24 accumulations are
deliberately ignored to avoid double counting.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("prepare_rj_local_sources")


def _period(date: pd.Series, frequency: str) -> pd.Series:
    date = pd.to_datetime(date, errors="coerce")
    if frequency == "daily":
        return date.dt.normalize()
    return (date - pd.to_timedelta(date.dt.weekday, unit="D")).dt.normalize()


def _finish(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    frame["DATE"] = _period(frame["DATE"], frequency)
    frame["RAIN"] = pd.to_numeric(frame["RAIN"], errors="coerce")
    frame.loc[frame["RAIN"] < 0, "RAIN"] = np.nan
    frame = frame.dropna(subset=["DATE", "STATION", "LAT", "LNG"])
    result = (
        frame.groupby(["DATE", "STATION", "LAT", "LNG"], as_index=False)["RAIN"]
        .sum(min_count=1)
        .sort_values(["DATE", "STATION"])
    )
    result["DATE"] = result["DATE"].dt.strftime("%Y-%m-%d")
    return result


def prepare_websirenes(zip_path: Path, coords_path: Path, frequency: str) -> pd.DataFrame:
    coords = pd.read_csv(coords_path).rename(
        columns={"id_estacao": "STATION", "latitude": "LAT", "longitude": "LNG"}
    )
    coords["STATION"] = pd.to_numeric(coords["STATION"], errors="coerce")
    coords = coords.dropna(subset=["STATION", "LAT", "LNG"]).set_index("STATION")
    chunks = []
    pattern = re.compile(
        r"(?P<date>\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}-\d{2}\s+"
        r"(?P<m15>\S+).*\s+(?P<station>\d+)\s*$"
    )
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            name for name in archive.namelist()
            if re.search(r"ID_ESTACAO_\d+\.txt$", name)
        ]
        usable = [
            name for name in names
            if int(re.search(r"(\d+)\.txt$", name).group(1)) in coords.index
        ]
        LOG.info("WebSirenes: %d/%d arquivos possuem coordenadas", len(usable), len(names))
        for index, name in enumerate(usable, start=1):
            station = int(re.search(r"(\d+)\.txt$", name).group(1))
            daily: dict[str, list[float]] = {}
            with archive.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace")
                next(text, None)
                for line in text:
                    match = pattern.search(line)
                    if not match:
                        continue
                    value = match.group("m15").replace(",", ".")
                    if value.lower() == "null":
                        continue
                    try:
                        rain = float(value)
                    except ValueError:
                        continue
                    if rain >= 0:
                        daily.setdefault(match.group("date"), []).append(rain)
            coord = coords.loc[station]
            chunks.append(
                pd.DataFrame(
                    {
                        "DATE": list(daily),
                        "STATION": f"WS_{station}",
                        "LAT": float(coord["LAT"]),
                        "LNG": float(coord["LNG"]),
                        "RAIN": [sum(values) for values in daily.values()],
                    }
                )
            )
            if index % 10 == 0 or index == len(usable):
                LOG.info("WebSirenes: %d/%d estações processadas", index, len(usable))
    return _finish(pd.concat(chunks, ignore_index=True), frequency)


def prepare_alertario(zip_path: Path, coords_path: Path, frequency: str) -> pd.DataFrame:
    coords = pd.read_csv(coords_path).rename(
        columns={"estacao_desc": "slug", "id": "station_id", "latitude": "LAT", "longitude": "LNG"}
    )
    coords["slug"] = coords["slug"].astype(str).str.strip().str.lower()
    lookup = coords.set_index("slug")
    chunks = []
    file_pattern = re.compile(r"(?:^|/)(.+)_(\d{6})_Plv\.txt$", re.IGNORECASE)
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            name for name in archive.namelist()
            if file_pattern.search(Path(name).name)
        ]
        LOG.info("AlertaRio: %d arquivos mensais encontrados", len(names))
        for index, name in enumerate(names, start=1):
            match = file_pattern.search(Path(name).name)
            slug = match.group(1).lower()
            if slug not in lookup.index:
                continue
            with archive.open(name) as raw:
                frame = pd.read_csv(
                    raw,
                    sep=r"\s{2,}",
                    engine="python",
                    skiprows=5,
                    names=["day", "time", "m15", "h01", "h04", "h24", "h96"],
                    encoding="utf-8",
                )
            coord = lookup.loc[slug]
            frame = pd.DataFrame(
                {
                    "DATE": pd.to_datetime(
                        frame["day"].astype(str) + " " + frame["time"].astype(str),
                        format="%d/%m/%Y %H:%M:%S",
                        errors="coerce",
                    ),
                    "STATION": f"AR_{int(coord['station_id'])}",
                    "LAT": float(coord["LAT"]),
                    "LNG": float(coord["LNG"]),
                    "RAIN": frame["m15"],
                }
            )
            # Collapse 15-minute rows immediately.  Keeping every observation
            # from all 5,490 files would consume roughly a gigabyte of memory.
            frame["RAIN"] = pd.to_numeric(frame["RAIN"], errors="coerce")
            frame.loc[frame["RAIN"] < 0, "RAIN"] = np.nan
            frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce").dt.normalize()
            frame = (
                frame.dropna(subset=["DATE"])
                .groupby(["DATE", "STATION", "LAT", "LNG"], as_index=False)["RAIN"]
                .sum(min_count=1)
            )
            chunks.append(frame)
            if index % 250 == 0:
                LOG.info("AlertaRio: %d/%d arquivos processados", index, len(names))
    return _finish(pd.concat(chunks, ignore_index=True), frequency)


def prepare_inmet(path: Path, frequency: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=[
            "CD_ESTACAO", "DT_MEDICAO", "TEM_MIN", "TEM_MAX", "TEM_AVG",
            "CHUVA", "VL_LATITUDE", "VL_LONGITUDE",
        ],
    ).rename(
        columns={
            "CD_ESTACAO": "STATION", "DT_MEDICAO": "DATE",
            "CHUVA": "RAIN", "VL_LATITUDE": "LAT", "VL_LONGITUDE": "LNG",
        }
    )
    frame["STATION"] = "INMET_" + frame["STATION"].astype(str)
    frame["DATE"] = _period(frame["DATE"], frequency)
    for column in ["TEM_MIN", "TEM_MAX", "TEM_AVG", "RAIN"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.loc[frame["RAIN"] < 0, "RAIN"] = np.nan
    functions = {"TEM_MIN": "min", "TEM_MAX": "max", "TEM_AVG": "mean", "RAIN": lambda x: x.sum(min_count=1)}
    result = frame.groupby(["DATE", "STATION", "LAT", "LNG"], as_index=False).agg(functions)
    result["DATE"] = result["DATE"].dt.strftime("%Y-%m-%d")
    return result.sort_values(["DATE", "STATION"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--websirenes-zip", required=True)
    parser.add_argument("--websirenes-coords", required=True)
    parser.add_argument("--alertario-zip", required=True)
    parser.add_argument("--alertario-coords", required=True)
    parser.add_argument("--inmet", required=True)
    parser.add_argument("--frequency", choices=["daily", "weekly"], default="weekly")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "websirenes": prepare_websirenes(Path(args.websirenes_zip), Path(args.websirenes_coords), args.frequency),
        "alertario": prepare_alertario(Path(args.alertario_zip), Path(args.alertario_coords), args.frequency),
        "inmet": prepare_inmet(Path(args.inmet), args.frequency),
    }
    for name, frame in sources.items():
        path = out / f"{name}_{args.frequency}.csv"
        frame.to_csv(path, index=False)
        LOG.info(
            "%s salvo: %s linhas, %s estações, %s a %s -> %s",
            name, f"{len(frame):,}", frame["STATION"].nunique(),
            frame["DATE"].min(), frame["DATE"].max(), path,
        )


if __name__ == "__main__":
    main()
