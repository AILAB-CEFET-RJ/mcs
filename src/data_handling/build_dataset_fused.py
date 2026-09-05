#!/usr/bin/env python3
"""Build canonical weekly DATA x CNES tables and tabular FULL_FUSED/CASEONLY datasets.

This is a fused-weather copy of the established tabular builder.  It never
opens ERA5 directly: all meteorological values and provenance come from the
single calibrated fusion artifact also used by the spatial branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from build_dataset import build_partition_with_history  # noqa: E402
from features.feature_config_parser import FeatureConfig  # noqa: E402
from features.feature_engineering import (  # noqa: E402
    CASE_LAG_CONTRACT_VERSION,
    create_new_features,
)


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_table(weather_path: Path, epidemiology_dir: Path, sinan_path: Path,
                    start: pd.Timestamp, end: pd.Timestamp, frequency: str,
                    purpose: str = "development", final_year: int = 2023) -> tuple[pd.DataFrame, dict]:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    mapping = pd.read_parquet(epidemiology_dir / "cnes_cell_mapping.parquet").copy()
    mapping["CNES"] = normalize_id(mapping["CNES"])
    mapping = mapping.sort_values("CNES").drop_duplicates("CNES")
    with np.load(weather_path, allow_pickle=True) as source:
        data_key = "weather" if "weather" in source else "era5_weekly"
        date_key = "dates" if "dates" in source else "week_dates"
        dates_all = pd.DatetimeIndex(source[date_key].astype(str))
        selected = (dates_all >= start) & (dates_all <= end)
        dates = dates_all[selected]
        channels = [str(v) for v in source["channels"]]
        units = [str(v) for v in source["channel_units"]]
        weather = np.asarray(source[data_key][selected], dtype=np.float32)
        provenance = np.asarray(source["source_code"][selected], dtype=np.uint8)
        source_names = [str(v) for v in source["source_names"]]
        target_ids = [str(v) for v in source["target_ids"]] if "target_ids" in source else None
    if channels != ["TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN", "RH_AVG", "RH_MIN", "RH_MAX"]:
        raise ValueError(f"Canais meteorológicos inesperados: {channels}")
    if units[channels.index("RAIN")].lower() != "mm":
        raise ValueError("Precipitação fusionada não está declarada em mm")
    if purpose == "development" and dates.max().year >= final_year:
        raise RuntimeError(f"Firewall violado: tabela de desenvolvimento contém {final_year}")
    if purpose == "final" and end.year != final_year:
        raise RuntimeError(f"Artefato final deve terminar em {final_year}")
    rows, cols = mapping.row.to_numpy(int), mapping.col.to_numpy(int)
    weather_rows, weather_cols = rows, cols
    if target_ids is not None:
        target_index = {value: index for index, value in enumerate(target_ids)}
        semantic_ids = mapping.CNES.astype(str)
        if not set(semantic_ids).issubset(target_index):
            semantic_ids = mapping.cell_id.astype(str)
        missing_targets = set(semantic_ids).difference(target_index)
        if missing_targets: raise RuntimeError(f"CNES sem alvo meteorológico: {len(missing_targets)}")
        weather_rows = np.zeros(len(mapping), dtype=int)
        weather_cols = semantic_ids.map(target_index).to_numpy(int)
    # Advanced indexing yields (T,C,U); transpose to date-major (T,U,C).
    met = weather[:, :, weather_rows, weather_cols].transpose(0, 2, 1)
    src = provenance[:, :, weather_rows, weather_cols].transpose(0, 2, 1)
    n_dates, n_units = len(dates), len(mapping)
    frame = pd.DataFrame({
        "DT_NOTIFIC": np.repeat(dates.to_numpy(), n_units),
        "ID_UNIDADE": np.tile(mapping.CNES.to_numpy(str), n_dates),
        "LAT": np.tile(mapping.LAT.to_numpy(float), n_dates),
        "LNG": np.tile(mapping.LNG.to_numpy(float), n_dates),
        "row": np.tile(rows, n_dates), "col": np.tile(cols, n_dates),
        "cell_id": np.tile(mapping.cell_id.to_numpy(int), n_dates),
    })
    for index, channel in enumerate(channels):
        frame[channel] = met[:, :, index].reshape(-1)
        frame[f"SOURCE_{channel}"] = src[:, :, index].reshape(-1)
    if not np.isfinite(frame[channels].to_numpy()).all():
        raise ValueError("Fallback meteorológico incompleto na tabela canônica")

    epi_manifest = json.loads((epidemiology_dir / "manifest.json").read_text(encoding="utf-8"))
    epidemiological_start = pd.Timestamp(epi_manifest["requested_interval"][0])
    cases = pd.read_parquet(sinan_path, columns=["ID_UNIDADE", "DT_NOTIFIC", "CASES"],
                            filters=[("CASES", ">", 0), ("DT_NOTIFIC", ">=", epidemiological_start),
                                     ("DT_NOTIFIC", "<=", end)])
    cases["ID_UNIDADE"] = normalize_id(cases["ID_UNIDADE"])
    cases["DT_NOTIFIC"] = pd.to_datetime(cases["DT_NOTIFIC"], errors="coerce")
    if cases["DT_NOTIFIC"].isna().any():
        raise RuntimeError("SINAN carregado contém DT_NOTIFIC inválida")
    raw_min_date = cases["DT_NOTIFIC"].min()
    raw_max_date = cases["DT_NOTIFIC"].max()
    rows_after_end = int((cases["DT_NOTIFIC"] > end).sum())
    year_2023_loaded = bool((cases["DT_NOTIFIC"].dt.year == 2023).any())
    if rows_after_end or (pd.notna(raw_max_date) and raw_max_date > end):
        raise RuntimeError(
            f"Firewall temporal violado: {rows_after_end} registros SINAN após {end.date()} "
            f"(máximo carregado={raw_max_date})"
        )
    if frequency == "weekly":
        cases["DT_NOTIFIC"] -= pd.to_timedelta(cases.DT_NOTIFIC.dt.weekday, unit="D")
    else:
        cases["DT_NOTIFIC"] = cases.DT_NOTIFIC.dt.normalize()
    cases = cases[cases.ID_UNIDADE.isin(set(mapping.CNES))]
    grouped = cases.groupby(["DT_NOTIFIC", "ID_UNIDADE"], as_index=False).CASES.sum()
    frame = frame.merge(grouped, on=["DT_NOTIFIC", "ID_UNIDADE"], how="left", validate="one_to_one")
    frame["CASES"] = frame.CASES.fillna(0).astype(np.float32)
    expected = n_dates * n_units
    if len(frame) != expected or frame[["DT_NOTIFIC", "ID_UNIDADE"]].duplicated().any():
        raise RuntimeError("Contrato DATA x CNES quebrado")
    audit = {"rows": len(frame), "dates": n_dates, "cnes_units": n_units,
             "grid_shape": list(weather.shape[-2:]), "channels": channels,
             "channel_units": dict(zip(channels, units)), "source_names": source_names,
             "cases": float(frame.CASES.sum()), "frequency": frequency,
             "sinan_loaded_interval": [raw_min_date.date().isoformat(), raw_max_date.date().isoformat()],
             "sinan_max_allowed_date": end.date().isoformat(),
             "sinan_rows_after_end": rows_after_end,
             "year_2023_loaded": year_2023_loaded, "final_year": final_year}
    return frame.sort_values(["ID_UNIDADE", "DT_NOTIFIC"], kind="stable"), audit


def save_model_dataset(frame: pd.DataFrame, config_path: Path, output: Path,
                       scenario: str, reference: Path | None = None,
                       train_start: str | None = None, validation_start: str | None = None,
                       test_start: str | None = None, test_end: str | None = None,
                       final_mode: bool = False, final_label: str = "final2023") -> dict:
    config = FeatureConfig(config_path)
    if train_start is not None: config.min_date = train_start
    if validation_start is not None: config.train_split = validation_start
    if test_start is not None: config.val_split = test_start
    if test_end is not None: config.max_date = test_end
    structural = ["LAT", "LNG", "row", "col", "cell_id"]
    provenance = [column for column in frame if column.startswith("SOURCE_")]
    model = frame.drop(columns=structural + provenance)
    if scenario == "CASEONLY":
        weather = ["TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN", "RH_AVG", "RH_MIN", "RH_MAX"]
        model = model.drop(columns=weather)
    train_date, val_date = pd.Timestamp(config.train_split), pd.Timestamp(config.val_split)
    # To make the first target after a split, its anchor (the immediately
    # preceding instant) needs lag_count earlier observations plus the anchor
    # itself. Rolling windows need exactly ``window`` rows including anchor.
    context = max(
        max(config.features.get("windows", [1])),
        int(config.features.get("lags", 0)) + 1,
    )
    if final_mode:
        if validation_start is None:
            raise RuntimeError("Modo final exige validation_start")
        partitions = {
            "train": (build_partition_with_history(model, config.min_date, train_date, context), config.min_date, train_date),
            final_label: (build_partition_with_history(model, train_date, None, context), train_date, None),
        }
    else:
        partitions = {
            "train": (build_partition_with_history(model, config.min_date, train_date, context), config.min_date, train_date),
            "val": (build_partition_with_history(model, train_date, val_date, context), train_date, val_date),
            "test": (build_partition_with_history(model, val_date, None, context), val_date, None),
        }
    output.mkdir(parents=True, exist_ok=True)
    arrays, ids = {}, {}
    for split, (part, target_start, target_end) in partitions.items():
        arrays[split] = create_new_features(part, split, config, str(output), target_start, target_end)
        ids[split] = arrays[split][2]
    if reference is not None:
        with (reference / "dataset_ids.pickle").open("rb") as handle:
            support = pickle.load(handle)
        for split in arrays:
            x, y, sidecar = arrays[split]
            current = pd.MultiIndex.from_arrays([pd.to_datetime(sidecar["DATE"]), sidecar["ID_UNIDADE"].astype(str)])
            wanted = pd.MultiIndex.from_arrays([pd.to_datetime(support[split]["DATE"]), support[split]["ID_UNIDADE"].astype(str)])
            positions = current.get_indexer(wanted)
            if np.any(positions < 0):
                raise RuntimeError(f"CASEONLY não reproduz o suporte FULL_FUSED em {split}")
            arrays[split] = (x[positions], y[positions], {k: np.asarray(v)[positions] for k, v in sidecar.items()})
            if not np.array_equal(arrays[split][1], np.asarray(support[split]["Y_TRUE"])):
                raise RuntimeError(f"Alvos CASEONLY/FULL_FUSED divergem em {split}")
            ids[split] = arrays[split][2]
    scaler = StandardScaler().fit(arrays["train"][0])
    scaled = {split: scaler.transform(value[0]).astype(np.float32) for split, value in arrays.items()}
    if not final_mode:
        with (output / "dataset.pickle").open("wb") as handle:
            pickle.dump((scaled["train"], arrays["train"][1], scaled["val"], arrays["val"][1],
                         scaled["test"], arrays["test"][1]), handle)
    split_dir = output / "splits"
    split_dir.mkdir(exist_ok=True)
    public_names = ({"train": "train", final_label: final_label} if final_mode else
                    {"train": "train", "val": "validation", "test": "confirmation"})
    for split, public_name in public_names.items():
        with (split_dir / f"{public_name}.pickle").open("wb") as handle:
            pickle.dump((scaled[split], arrays[split][1]), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (split_dir / f"{public_name}_ids.pickle").open("wb") as handle:
            pickle.dump(ids[split], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (output / "dataset_ids.pickle").open("wb") as handle:
        pickle.dump(ids, handle)
    with (output / "scaler.pickle").open("wb") as handle:
        pickle.dump(scaler, handle)
    target_date_ranges = {}
    for split, sidecar in ids.items():
        split_dates = pd.DatetimeIndex(pd.to_datetime(sidecar["DATE"])).normalize()
        target_date_ranges[split] = [
            split_dates.min().date().isoformat(), split_dates.max().date().isoformat()
        ] if len(split_dates) else [None, None]
    (output / "dataset_meta.json").write_text(
        json.dumps({
            "scenario": scenario,
            "case_lag_contract": CASE_LAG_CONTRACT_VERSION,
            "case_lag_definition": "CASES_LAG_k at anchor t equals CASES[t-k+1]",
            "first_target_definition": "t+1",
            "samples": {split: int(len(arrays[split][1])) for split in arrays},
            "target_date_ranges": target_date_ranges,
            "selection_contract": {
                "selection_reads": ["splits/train.pickle", "splits/validation.pickle"],
                "confirmation_artifact": "splits/confirmation.pickle",
                "confirmation_must_not_be_loaded_during_selection": True,
            } if not final_mode else {
                "refit_reads": ["splits/train.pickle"],
                "final_artifact": f"splits/{final_label}.pickle",
                "final_requires_frozen_protocol": True,
            },
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {split: int(len(arrays[split][1])) for split in arrays}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weather", required=True)
    p.add_argument("--epidemiology-dir", required=True)
    p.add_argument("--sinan", required=True)
    p.add_argument("--full-config", required=True)
    p.add_argument("--caseonly-config", required=True)
    p.add_argument("--frequency", choices=["daily", "weekly"], default="weekly")
    p.add_argument("--scenarios", choices=["CANONICAL", "FULL", "BOTH"], default="BOTH")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", default="2013-12-30", help="Inclui a semana parcial de contexto de 2014")
    p.add_argument("--end", default="2022-12-31")
    p.add_argument("--train-start", default="")
    p.add_argument("--validation-start", default="")
    p.add_argument("--test-start", default="")
    p.add_argument("--test-end", default="")
    p.add_argument("--expected-cases", type=float, default=None,
                   help="Falha se o total canônico não for exatamente o valor informado")
    p.add_argument("--purpose", choices=["development", "final"], default="development")
    p.add_argument("--frozen-protocol", default="",
                   help="Manifesto status=FROZEN obrigatório para carregar 2023")
    p.add_argument("--location-prefix", default="RJ")
    p.add_argument("--final-year", type=int, default=2023)
    p.add_argument("--final-label", default="final2023")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    weather, epi, sinan = Path(args.weather), Path(args.epidemiology_dir), Path(args.sinan)
    if args.purpose == "final":
        if not args.frozen_protocol:
            raise RuntimeError("Build final bloqueado: informe --frozen-protocol")
        frozen = json.loads(Path(args.frozen_protocol).read_text(encoding="utf-8"))
        if frozen.get("status") != "FROZEN":
            raise RuntimeError("Build final bloqueado: protocolo ainda não está FROZEN")
    frame, audit = canonical_table(weather, epi, sinan, pd.Timestamp(args.start), pd.Timestamp(args.end),
                                   args.frequency, args.purpose, args.final_year)
    if args.expected_cases is not None and not np.isclose(audit["cases"], args.expected_cases, rtol=0, atol=1e-6):
        raise RuntimeError(
            f"Total canônico inválido: obtido={audit['cases']}, esperado={args.expected_cases}"
        )
    label = args.frequency.upper()
    canonical = out / f"{args.location_prefix}_{label}_CANONICAL_DATA_CNES.parquet"
    frame.to_parquet(canonical, index=False)
    if args.scenarios == "CANONICAL":
        audit.update({"status": "READY", "canonical_table": str(canonical),
                      "purpose": args.purpose, "weather_artifact": str(weather),
                      "weather_sha256": sha256(weather), "case_lag_contract": CASE_LAG_CONTRACT_VERSION})
        (out / "manifest.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=False))
        return
    full = out / f"{args.location_prefix}_{label}_FULL_FUSED"
    caseonly = out / f"{args.location_prefix}_{label}_CASEONLY"
    overrides = dict(train_start=args.train_start or None, validation_start=args.validation_start or None,
                     test_start=args.test_start or None, test_end=args.test_end or None,
                     final_mode=args.purpose == "final", final_label=args.final_label)
    audit["samples"] = {"FULL_FUSED": save_model_dataset(frame, Path(args.full_config), full, "FULL_FUSED", **overrides)}
    if args.scenarios == "BOTH":
        audit["samples"]["CASEONLY"] = save_model_dataset(frame, Path(args.caseonly_config), caseonly, "CASEONLY", full, **overrides)
    full_meta = json.loads((full / "dataset_meta.json").read_text(encoding="utf-8"))
    if args.purpose == "final":
        audit["temporal_splits"] = {
            "refit": full_meta["target_date_ranges"]["train"],
            args.final_label: full_meta["target_date_ranges"][args.final_label],
            "interval_policy": f"refit before {args.final_year}; {args.final_label} locked until explicit evaluation",
        }
    else:
        audit["temporal_splits"] = {
            "train": full_meta["target_date_ranges"]["train"],
            "validation": full_meta["target_date_ranges"]["val"],
            "confirmation": full_meta["target_date_ranges"]["test"],
            "interval_policy": "train/validation right-open; confirmation end inclusive",
        }
    audit.update({"status": "READY", "canonical_table": str(canonical),
                  "purpose": args.purpose,
                  "weather_artifact": str(weather), "weather_sha256": sha256(weather),
                  "scaler_fit": "train only",
                  "caseonly_support": "identical to FULL_FUSED",
                  "case_lag_contract": CASE_LAG_CONTRACT_VERSION})
    (out / "manifest.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
