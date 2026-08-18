#!/usr/bin/env python3
"""Fusão hierárquica de fontes meteorológicas sobre uma grade ERA5.

O ERA5 é sempre a cobertura inicial. Cada fonte observacional opcional é
interpolada por IDW e substitui apenas células/instantes/canais em que possui
dado válido. ``--priority`` define a ordem: por padrão,
INMET > AlertaRio > WebSirenes; ERA5 permanece como fallback.

Formato de cada fonte (CSV):
    DATE, STATION, LAT, LNG e quaisquer colunas canônicas disponíveis entre
    TEM_AVG,TEM_MIN,TEM_MAX,DEW_AVG,RH_AVG,RAIN.

Exemplo:
    python arboseer/src/data_handling/stations/fuse_weather_sources.py \
      --era5 RJ_era5_daily.npz --out RJ_weather_fused_daily.npz \
      --source ALERTARIO=alertario_daily.csv \
      --source WEBSIRENES=websirenes_daily.csv \
      --source AIRPORTS=redemet_daily.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .interpolate_to_grid import idw_weights, interpolate_channel
except ImportError:  # direct script execution
    from interpolate_to_grid import idw_weights, interpolate_channel

LOG = logging.getLogger("fuse_weather_sources")

DEFAULT_PRIORITY = ("INMET", "ALERTARIO", "WEBSIRENES")

CHANNEL_ALIASES = {
    "TEM_AVG": ("TEM_AVG",),
    "TEM_MIN": ("TEM_MIN",),
    "TEM_MAX": ("TEM_MAX",),
    "DEW_AVG": ("DEW_AVG",),
    "RH_AVG": ("RH_AVG", "HUM_AVG"),
    "RAIN": ("RAIN", "PRECIP"),
}


def _pick(npz: np.lib.npyio.NpzFile, names: tuple[str, ...]):
    for name in names:
        if name in npz.files:
            return npz[name], name
    raise KeyError(f"Nenhuma chave {names} encontrada; disponíveis: {npz.files}")


def _read_source(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Fonte inválida '{spec}'; use NOME=caminho.csv")
    name, raw_path = spec.split("=", 1)
    name = name.strip().upper()
    path = Path(raw_path.strip())
    if not name or not raw_path.strip():
        raise ValueError(f"Fonte inválida '{spec}'")
    return name, path


def _canonical_column(frame: pd.DataFrame, channel: str) -> str | None:
    for candidate in CHANNEL_ALIASES.get(channel, (channel,)):
        if candidate in frame.columns:
            return candidate
    return None


def fuse(args: argparse.Namespace) -> dict:
    era5 = np.load(args.era5, allow_pickle=True)
    baseline, data_key = _pick(era5, ("weather", "era5_daily", "era5_weekly"))
    dates, dates_key = _pick(era5, ("dates", "day_dates", "week_dates"))
    all_channels = [str(value) for value in era5["channels"]]
    requested = set(args.enabled_channel or all_channels)
    requested.difference_update(args.disable_channel)
    unknown = requested.difference(all_channels)
    if unknown:
        raise ValueError(f"Canais meteorológicos desconhecidos: {sorted(unknown)}")
    channel_indices = [i for i, name in enumerate(all_channels) if name in requested]
    if not channel_indices:
        raise ValueError("Todos os canais meteorológicos foram desligados")
    channels = [all_channels[i] for i in channel_indices]
    if "target_lat" in era5 and "target_lon" in era5:
        lats = np.asarray(era5["target_lat"], dtype=float)
        lons = np.asarray(era5["target_lon"], dtype=float)
    else:
        lats = np.asarray(era5["lat"], dtype=float)
        lons = np.asarray(era5["lon"], dtype=float)
    result = np.asarray(baseline, dtype=np.float32)[:, channel_indices].copy()
    if result.ndim != 4:
        raise ValueError(f"Grade ERA5 deve ser (T,C,H,W); recebido {result.shape}")

    date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d")
    date_to_index = {value: index for index, value in enumerate(date_strings)}
    source_specs = [_read_source(value) for value in args.source]
    priority = [value.strip().upper() for value in args.priority.split(",") if value.strip()]
    if len(priority) != len(set(priority)):
        raise ValueError(f"Prioridade contém fontes repetidas: {priority}")
    rank = {name: index for index, name in enumerate(priority)}
    source_specs.sort(key=lambda item: (rank.get(item[0], len(rank)), item[0]))
    source_names = [name for name, _ in source_specs]
    # 0=ERA5; 1..N seguem a prioridade declarada.
    provenance = np.zeros(result.shape, dtype=np.uint8)
    coverage: dict[str, dict[str, int | float]] = {}
    influence: dict[str, dict[str, object]] = {}

    for source_code, (source_name, path) in enumerate(source_specs, start=1):
        if not path.exists():
            if args.allow_missing:
                LOG.warning("%s ausente (%s); ERA5/fonte seguinte será usada", source_name, path)
                coverage[source_name] = {"available": False, "filled_values": 0}
                continue
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        required = {"DATE", "STATION", "LAT", "LNG"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{source_name}: colunas obrigatórias ausentes: {sorted(missing)}")
        frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["DATE", "STATION", "LAT", "LNG"])
        metadata = frame.drop_duplicates("STATION").set_index("STATION")[["LAT", "LNG"]]
        stations = metadata.index.tolist()
        if not stations:
            LOG.warning("%s não possui estações utilizáveis", source_name)
            coverage[source_name] = {"available": True, "filled_values": 0}
            continue

        weights = idw_weights(
            lats,
            lons,
            metadata["LAT"].to_numpy(float),
            metadata["LNG"].to_numpy(float),
            alpha=args.alpha,
            cutoff_km=args.cutoff_km,
        )
        station_cell_counts = (weights > 0).sum(axis=(0, 1)).astype(int)
        source_cell_union = int((weights > 0).any(axis=2).sum())
        influence[source_name] = {
            "cutoff_km": float(args.cutoff_km),
            "grid_cells_in_union": source_cell_union,
            "grid_cells_total": int(weights.shape[0] * weights.shape[1]),
            "stations_total": len(stations),
            "stations_reaching_grid": int((station_cell_counts > 0).sum()),
            "cells_per_reaching_station": {
                "min": int(station_cell_counts[station_cell_counts > 0].min()) if (station_cell_counts > 0).any() else 0,
                "median": float(np.median(station_cell_counts[station_cell_counts > 0])) if (station_cell_counts > 0).any() else 0.0,
                "max": int(station_cell_counts.max()) if len(station_cell_counts) else 0,
            },
            "stations": [
                {
                    "station": str(station),
                    "lat": float(metadata.loc[station, "LAT"]),
                    "lon": float(metadata.loc[station, "LNG"]),
                    "cells_within_cutoff": int(station_cell_counts[index]),
                }
                for index, station in enumerate(stations)
            ],
        }
        filled = 0
        eligible = 0
        for channel_index, channel in enumerate(channels):
            source_column = _canonical_column(frame, channel)
            if source_column is None:
                continue
            values = np.full((len(dates), len(stations)), np.nan, dtype=np.float32)
            station_to_index = {station: index for index, station in enumerate(stations)}
            subset = frame[["DATE", "STATION", source_column]].copy()
            subset[source_column] = pd.to_numeric(subset[source_column], errors="coerce")
            subset = subset.dropna(subset=[source_column])
            # Múltiplas observações já agregadas no mesmo período são reduzidas
            # pela média; precipitação deve chegar previamente acumulada.
            subset = subset.groupby(["DATE", "STATION"], as_index=False)[source_column].mean()
            for row in subset.itertuples(index=False):
                time_index = date_to_index.get(row[0])
                station_index = station_to_index.get(row[1])
                if time_index is not None and station_index is not None:
                    values[time_index, station_index] = float(row[2])

            interpolated, _ = interpolate_channel(
                np.arange(len(dates)), values, weights
            )
            valid = np.isfinite(interpolated)
            unclaimed = provenance[:, channel_index] == 0
            use = valid & unclaimed
            eligible += int(valid.sum())
            result[:, channel_index][use] = interpolated[use]
            provenance[:, channel_index][use] = source_code
            filled += int(use.sum())

        coverage[source_name] = {
            "available": True,
            "stations": len(stations),
            "eligible_values": eligible,
            "filled_values": filled,
            "share_of_tensor": filled / float(result.size),
        }
        LOG.info("%s preencheu %d valores (prioridade %d)", source_name, filled, source_code)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    output_arrays = dict(
        weather=result,
        channels=np.asarray(channels),
        lat=lats,
        lon=lons,
        dates=np.asarray(date_strings),
        source_code=provenance,
        source_names=np.asarray(["ERA5", *source_names]),
    )
    for audit_key in (
        "era5_source_pixel_ids",
        "era5_nearest_fallback_mask",
        "era5_nearest_source_pixel_id",
        "interpolation_method",
        "resolution_warning",
        "target_lat",
        "target_lon",
    ):
        if audit_key in era5:
            output_arrays[audit_key] = era5[audit_key]
    np.savez_compressed(out, **output_arrays)
    report = {
        "policy": "first valid optional source by declared priority; ERA5 fallback",
        "era5": str(Path(args.era5)),
        "output": str(out),
        "data_key_read": data_key,
        "dates_key_read": dates_key,
        "priority": source_names,
        "configured_priority": priority,
        "enabled_channels": channels,
        "disabled_channels": [name for name in all_channels if name not in channels],
        "source_codes": {"ERA5": 0, **{name: i + 1 for i, name in enumerate(source_names)}},
        "coverage": coverage,
        "spatial_influence": influence,
        "shape": list(result.shape),
        "alpha": args.alpha,
        "cutoff_km": args.cutoff_km,
    }
    report_path = out.with_suffix(".provenance.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5", required=True, help="NPZ de cobertura meteorológica ERA5")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NOME=CSV",
        help="Fonte opcional; repetir na ordem de prioridade",
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--cutoff-km", type=float, default=150.0)
    parser.add_argument(
        "--priority", default=",".join(DEFAULT_PRIORITY),
        help="Prioridade separada por vírgulas; ERA5 é sempre o fallback final.",
    )
    parser.add_argument(
        "--enabled-channel", action="append", default=[],
        help="Canal meteorológico a manter; repetir. Vazio mantém todos.",
    )
    parser.add_argument(
        "--disable-channel", action="append", default=[],
        help="Canal meteorológico a desligar; repetir.",
    )
    parser.add_argument(
        "--allow-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignora fontes opcionais ausentes e mantém o fallback",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = fuse(parse_args())
    LOG.info("Fusão concluída: %s", report["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
