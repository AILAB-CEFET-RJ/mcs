"""Dataset builder.

Compatibilidade:
- Mantém o modo antigo (FULL/unidade específica + ERA5) como padrão.
- Adiciona modo de prova de conceito: clusters por proximidade em torno de uma
  estação do INMET, consumindo o aggregated.parquet gerado pelo legacy.

Modos (via YAML):

mode:
  aggregation: unit | cluster
  meteo: era5 | inmet_legacy
"""

import os
import argparse
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import xarray as xr
import yaml
import json
from pathlib import Path

from features.feature_config_parser import FeatureConfig
from features.feature_engineering import create_new_features

# === NOVO: providers/recipes (PoC INMET + clusters) ===
try:
    from providers.inmet_legacy_aggregated_provider import InmetLegacyAggregatedProvider
    from recipes.cluster_recipe import ClusterRecipe
except Exception:
    InmetLegacyAggregatedProvider = None
    ClusterRecipe = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Leitura ERA5
def extract_era5_data(ds, lat, lon, date, config):
    if "time" not in ds.coords and "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
        
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    if config.weekly:
        week_start = pd.Timestamp(date)
        week_end = week_start + pd.Timedelta(days=6)
        day_data = ds_point.sel(time=slice(f"{week_start} 00:00:00", f"{week_end} 23:00:00"))
    else:
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        day_data = ds_point.sel(time=slice(f"{date_str} 00:00:00", f"{date_str} 23:00:00"))

    if day_data.time.size == 0:
        raise ValueError(f"No data for {date}")

    result = {}

    raw_flags = config.features["enable"].get("raw_features", {})

    if raw_flags.get("tem_avg", True):
        result["TEM_AVG"] = day_data["t2m"].mean().item() - 273.15
    if raw_flags.get("tem_min", True):
        result["TEM_MIN"] = day_data["t2m"].min().item() - 273.15
    if raw_flags.get("tem_max", True):
        result["TEM_MAX"] = day_data["t2m"].max().item() - 273.15
    if raw_flags.get("rain", True):
        result["RAIN"] = day_data["tp"].sum().item()

    if any(raw_flags.get(k, False) for k in ["rh_avg", "rh_min", "rh_max"]):
        t2m = day_data["t2m"].values
        d2m = day_data["d2m"].values

        def sat_vapor_pressure(temp):
            return 6.112 * np.exp((17.67 * (temp - 273.15)) / (temp - 29.65))

        rh = 100 * (sat_vapor_pressure(d2m) / sat_vapor_pressure(t2m))

        if raw_flags.get("rh_avg", True):
            result["RH_AVG"] = np.mean(rh)
        if raw_flags.get("rh_min", False):
            result["RH_MIN"] = np.min(rh)
        if raw_flags.get("rh_max", False):
            result["RH_MAX"] = np.max(rh)

    return result


def _resample_era5_variable(data, rule, operation, weekly):
    kwargs = {"closed": "left", "label": "left"} if weekly else {}
    resampled = data.resample(time=rule, **kwargs)
    reduce_dims = [dim for dim in ("time", "expver") if dim in data.dims]
    return getattr(resampled, operation)(dim=reduce_dims, skipna=True)


def aggregate_era5_grid(ds, start_date, end_date, weekly=False):
    """Agrega toda a grade necessária de uma vez, sem alterar as estatísticas.

    A implementação anterior selecionava a mesma célula do NetCDF novamente
    para cada data e unidade. Aqui cada variável é agregada vetorialmente pelo
    xarray e depois convertida em uma tabela usada pelo merge.
    """
    if "time" not in ds.coords and "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    slice_end = end_date + pd.Timedelta(days=6 if weekly else 0, hours=23)
    selected = ds.sel(time=slice(start_date, slice_end))
    rule = "W-MON" if weekly else "1D"

    t2m = selected["t2m"]
    d2m = selected["d2m"]
    tp = selected["tp"]

    def sat_vapor_pressure(temp):
        return 6.112 * np.exp((17.67 * (temp - 273.15)) / (temp - 29.65))

    rh = 100 * (sat_vapor_pressure(d2m) / sat_vapor_pressure(t2m))
    aggregated = xr.Dataset({
        "TEM_AVG": _resample_era5_variable(t2m, rule, "mean", weekly) - 273.15,
        "TEM_MIN": _resample_era5_variable(t2m, rule, "min", weekly) - 273.15,
        "TEM_MAX": _resample_era5_variable(t2m, rule, "max", weekly) - 273.15,
        "RAIN": _resample_era5_variable(tp, rule, "sum", weekly),
        "RH_AVG": _resample_era5_variable(rh, rule, "mean", weekly),
        "RH_MIN": _resample_era5_variable(rh, rule, "min", weekly),
        "RH_MAX": _resample_era5_variable(rh, rule, "max", weekly),
    })
    result = aggregated.to_dataframe().reset_index()
    result = result.rename(columns={"time": "DT_NOTIFIC", "latitude": "LAT_ERA5", "longitude": "LNG_ERA5"})
    result["DT_NOTIFIC"] = pd.to_datetime(result["DT_NOTIFIC"])
    return result[
        ["DT_NOTIFIC", "LAT_ERA5", "LNG_ERA5", "TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN", "RH_AVG", "RH_MIN", "RH_MAX"]
    ]


def load_or_build_era5_grid(era5_path, ds, start_date, end_date, weekly=False):
    """Reutiliza a agregação se o NetCDF e o intervalo não mudaram."""
    source = Path(era5_path)
    frequency = "weekly" if weekly else "daily"
    start_key = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_key = pd.Timestamp(end_date).strftime("%Y%m%d")
    cache_path = source.with_name(f"{source.stem}.arboseer_{frequency}_{start_key}_{end_key}.parquet")

    if cache_path.exists() and cache_path.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        logging.info("⚡ Reutilizando cache ERA5: %s", cache_path)
        return pd.read_parquet(cache_path)

    logging.info("⚡ Agregando grade ERA5 de forma vetorizada...")
    result = aggregate_era5_grid(ds, start_date, end_date, weekly=weekly)
    result.to_parquet(cache_path, index=False)
    logging.info("💾 Cache ERA5 salvo em: %s", cache_path)
    return result


def build_partition_with_history(df, target_start, target_end, context_rows):
    """Mantém o período do alvo e acrescenta apenas contexto anterior ao corte."""
    dates = pd.to_datetime(df["DT_NOTIFIC"])
    current = df[dates >= pd.Timestamp(target_start)].copy()
    if target_end is not None:
        current = current[pd.to_datetime(current["DT_NOTIFIC"]) < pd.Timestamp(target_end)]

    history = df[dates < pd.Timestamp(target_start)].copy()
    if "ID_UNIDADE" in history.columns:
        history = (
            history.sort_values(["ID_UNIDADE", "DT_NOTIFIC"], kind="stable")
            .groupby("ID_UNIDADE", sort=False, group_keys=False)
            .tail(context_rows)
        )
    else:
        history = history.sort_values("DT_NOTIFIC", kind="stable").tail(context_rows)

    return pd.concat([history, current], ignore_index=True)


def filter_to_reference_support(X, y, sidecar, reference_dir, split):
    """Restringe uma partição às mesmas datas/unidades do dataset FULL."""
    reference_path = Path(reference_dir) / "dataset_ids.pickle"
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Dataset de suporte não encontrado: {reference_path}. "
            "Construa o dataset FULL antes do CASEONLY."
        )

    with reference_path.open("rb") as file:
        reference = pickle.load(file)[split]

    current_keys = pd.MultiIndex.from_arrays([
        pd.to_datetime(sidecar["DATE"]),
        np.asarray(sidecar["ID_UNIDADE"]).astype(str),
    ])
    reference_keys = pd.MultiIndex.from_arrays([
        pd.to_datetime(reference["DATE"]),
        np.asarray(reference["ID_UNIDADE"]).astype(str),
    ])
    keep = current_keys.isin(reference_keys)

    filtered_sidecar = {
        key: np.asarray(values)[keep]
        for key, values in sidecar.items()
    }
    X_filtered = X[keep]
    y_filtered = y[keep]

    if len(y_filtered) != len(reference_keys):
        raise RuntimeError(
            f"Suporte incompatível em {split}: CASEONLY={len(y_filtered)} "
            f"e FULL={len(reference_keys)}."
        )
    if not np.array_equal(y_filtered, np.asarray(reference["Y_TRUE"])):
        raise RuntimeError(f"Alvos divergentes entre CASEONLY e FULL em {split}.")

    return X_filtered, y_filtered, filtered_sidecar

# Processamento total
def build_dataset_era5(config_path, sinan_path, cnes_path, era5_path, output_path, id_unidade):
    """Modo antigo: por unidade + ERA5."""
    config = FeatureConfig(config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)
    reference_support = raw_config.get("paths", {}).get("support_dataset")

    logging.info("🔧 Lendo dados...")
    sinan_df = pd.read_parquet(sinan_path)
    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'])
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].astype(str)

    if id_unidade != "FULL":
        unidade_df = sinan_df[sinan_df["ID_UNIDADE"] == id_unidade].copy()
        sinan_df = unidade_df

    cnes_df = pd.read_parquet(cnes_path)
    cnes_df['CNES'] = cnes_df['CNES'].astype(str)
    sinan_df = pd.merge(sinan_df, cnes_df[['CNES', 'LAT', 'LNG']].rename(columns={'CNES': 'ID_UNIDADE'}), on='ID_UNIDADE', how='left')
    sinan_df.dropna(subset=['LAT', 'LNG'], inplace=True)

    if config.weekly:
        sinan_df['DT_SEMANA'] = sinan_df['DT_NOTIFIC'].dt.to_period('W').apply(lambda r: r.start_time)
        sinan_df = sinan_df.groupby(['ID_UNIDADE', 'DT_SEMANA']).agg({'CASES': 'sum', 'LAT': 'first', 'LNG': 'first'}).reset_index()

    dt_col = 'DT_SEMANA' if config.weekly else 'DT_NOTIFIC'
    raw_flags = config.features["enable"].get("raw_features", {})
    raw_columns = {
        "tem_avg": "TEM_AVG", "tem_min": "TEM_MIN", "tem_max": "TEM_MAX",
        "rain": "RAIN", "rh_avg": "RH_AVG", "rh_min": "RH_MIN", "rh_max": "RH_MAX",
    }
    requested_columns = [column for flag, column in raw_columns.items() if raw_flags.get(flag, False)]

    if requested_columns:
        era5_ds = xr.open_dataset(era5_path)
        grid_coords = np.array([
            (lat, lon) for lat in era5_ds.latitude.values for lon in era5_ds.longitude.values
        ])
        era5_tree = cKDTree(grid_coords)

        # O pareamento espacial depende apenas da unidade, não de cada registro.
        unit_coords = sinan_df[["ID_UNIDADE", "LAT", "LNG"]].drop_duplicates("ID_UNIDADE").copy()
        _, nearest_indices = era5_tree.query(unit_coords[["LAT", "LNG"]].to_numpy(), k=1)
        nearest_coords = grid_coords[nearest_indices]
        unit_coords["LAT_ERA5"] = nearest_coords[:, 0]
        unit_coords["LNG_ERA5"] = nearest_coords[:, 1]
        sinan_df = sinan_df.merge(
            unit_coords[["ID_UNIDADE", "LAT_ERA5", "LNG_ERA5"]],
            on="ID_UNIDADE",
            how="left",
        )

        era5_df = load_or_build_era5_grid(
            era5_path,
            era5_ds,
            sinan_df[dt_col].min(),
            sinan_df[dt_col].max(),
            weekly=config.weekly,
        ).rename(columns={"DT_NOTIFIC": dt_col})
        era5_df = era5_df[[dt_col, "LAT_ERA5", "LNG_ERA5"] + requested_columns]
        sinan_df = sinan_df.merge(
            era5_df,
            on=[dt_col, "LAT_ERA5", "LNG_ERA5"],
            how="left",
        )
        sinan_df.drop(columns=["LAT_ERA5", "LNG_ERA5"], inplace=True)
        era5_ds.close()
    else:
        logging.info("⚡ Dataset sem clima: leitura e processamento do ERA5 ignorados.")

    sinan_df.drop(columns=['LAT', 'LNG'], inplace=True)

    if config.weekly:
        sinan_df.rename(columns={'DT_SEMANA': 'DT_NOTIFIC'}, inplace=True)
        
    train_date = pd.to_datetime(config.train_split)
    val_date = pd.to_datetime(config.val_split)

    context_rows = max(
        max(config.features.get("windows", [1])),
        int(config.features.get("lags", 0)),
    )
    train = build_partition_with_history(
        sinan_df, config.min_date, train_date, context_rows
    )
    val = build_partition_with_history(
        sinan_df, train_date, val_date, context_rows
    )
    test = build_partition_with_history(
        sinan_df, val_date, None, context_rows
    )

    logging.info("🧪 Feature engineering...")
    X_train, y_train, ids_train = create_new_features(
        train, "train", config, output_path,
        target_start=config.min_date, target_end=train_date,
    )
    X_val, y_val, ids_val = create_new_features(
        val, "val", config, output_path,
        target_start=train_date, target_end=val_date,
    )
    X_test, y_test, ids_test = create_new_features(
        test, "test", config, output_path,
        target_start=val_date,
    )

    if reference_support:
        X_train, y_train, ids_train = filter_to_reference_support(
            X_train, y_train, ids_train, reference_support, "train"
        )
        X_val, y_val, ids_val = filter_to_reference_support(
            X_val, y_val, ids_val, reference_support, "val"
        )
        X_test, y_test, ids_test = filter_to_reference_support(
            X_test, y_test, ids_test, reference_support, "test"
        )

    logging.info("⚖️ Normalizando...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logging.info("💾 Salvando pickle...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path+"/dataset.pickle", "wb") as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)

    # 💾 Sidecar com identificadores alinhados (não entram no dataset.pickle)
    ids_payload = {
        "train": ids_train,   # dict com arrays: DATE, ID_UNIDADE
        "val":   ids_val,
        "test":  ids_test,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(os.path.join(output_path, "dataset_ids.pickle"), "wb") as f:
        pickle.dump(ids_payload, f)
    with open(os.path.join(output_path, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"sidecars": ["dataset_ids.pickle", "dataset_meta.json"]}, f, ensure_ascii=False, indent=2)      

    logging.info(f"✅ Dataset final salvo em: {output_path}")


def build_dataset_clusters_inmet_legacy(
    config_path,
    sinan_path,
    cnes_path,
    inmet_aggregated_parquet,
    inmet_station_id,
    station_lat,
    station_lon,
    level_edges_km,
    clusters,
    output_base,
):
    """PoC: cria datasets agregando unidades ao redor de uma estação INMET.

    Entrada meteorológica: aggregated.parquet produzido pelo legacy.
    """
    if InmetLegacyAggregatedProvider is None or ClusterRecipe is None:
        raise RuntimeError(
            "Módulos providers/ e recipes/ não encontrados. "
            "Aplique o patch completo ou verifique o PYTHONPATH."
        )

    config = FeatureConfig(config_path)

    logging.info("🔧 [cluster+inmet_legacy] Lendo SINAN + CNES...")
    sinan_df = pd.read_parquet(sinan_path)
    sinan_df["DT_NOTIFIC"] = pd.to_datetime(sinan_df["DT_NOTIFIC"])
    sinan_df["ID_UNIDADE"] = sinan_df["ID_UNIDADE"].astype(str)

    cnes_df = pd.read_parquet(cnes_path)
    cnes_df["CNES"] = cnes_df["CNES"].astype(str)

    recipe = ClusterRecipe(
        station_lat=station_lat,
        station_lon=station_lon,
        level_edges_km=level_edges_km,
        clusters=clusters,
    )

    cluster_cases = recipe.make_clusters(sinan_df, cnes_df, weekly=config.weekly)

    met_provider = InmetLegacyAggregatedProvider(
        aggregated_parquet_path=inmet_aggregated_parquet,
        station_id=str(inmet_station_id),
    )
    met_df = met_provider.get_series(weekly=config.weekly)

    train_date = pd.to_datetime(config.train_split)
    val_date = pd.to_datetime(config.val_split)

    for cluster_name, df_cases in cluster_cases.items():
        logging.info(f"🧩 [cluster] Montando dataset: {cluster_name}")

        df = df_cases.merge(met_df, on="DT_NOTIFIC", how="left")

        train = df[df["DT_NOTIFIC"] < train_date]
        val = df[(df["DT_NOTIFIC"] >= train_date) & (df["DT_NOTIFIC"] < val_date)]
        test = df[df["DT_NOTIFIC"] >= val_date]

        out_dir = os.path.join(output_base, cluster_name)
        os.makedirs(out_dir, exist_ok=True)

        logging.info("🧪 Feature engineering...")
        X_train, y_train, ids_train = create_new_features(train, "train", config, out_dir)
        X_val, y_val, ids_val = create_new_features(val, "val", config, out_dir)
        X_test, y_test, ids_test = create_new_features(test, "test", config, out_dir)

        logging.info("⚖️ Normalizando...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        logging.info("💾 Salvando pickle...")
        with open(os.path.join(out_dir, "dataset.pickle"), "wb") as f:
            pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)

        ids_payload = {"train": ids_train, "val": ids_val, "test": ids_test}
        with open(os.path.join(out_dir, "dataset_ids.pickle"), "wb") as f:
            pickle.dump(ids_payload, f)

        meta = {
            "mode": {"aggregation": "cluster", "meteo": "inmet_legacy"},
            "cluster_name": cluster_name,
            "station": {
                "id": str(inmet_station_id),
                "lat": station_lat,
                "lon": station_lon,
            },
            "level_edges_km": level_edges_km,
            "clusters": clusters,
            "sidecars": ["dataset_ids.pickle", "dataset_meta.json"],
        }
        with open(os.path.join(out_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logging.info(f"✅ Cluster dataset salvo em: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arboseer Dataset Builder v3")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)

    paths = full_config.get("paths", {})
    sinan_path = paths.get("sinan")
    cnes_path = paths.get("cnes")
    era5_path = paths.get("era5")
    output_path = paths.get("output")
    id_unidade = paths.get("unidade", "FULL")

    mode = full_config.get("mode", {})
    aggregation = (mode.get("aggregation") or "unit").lower()
    meteo = (mode.get("meteo") or "era5").lower()

    if aggregation == "unit" and meteo == "era5":
        build_dataset_era5(
            config_path=args.config,
            sinan_path=sinan_path,
            cnes_path=cnes_path,
            era5_path=era5_path,
            output_path=output_path,
            id_unidade=id_unidade,
        )
    elif aggregation == "cluster" and meteo in ("inmet_legacy", "inmet-legacy"):
        cluster_cfg = full_config.get("cluster", {})
        st = cluster_cfg.get("station", {})
        station_lat = float(st.get("lat"))
        station_lon = float(st.get("lon"))
        level_edges_km = cluster_cfg.get("level_edges_km", [5, 10, 20])
        clusters = cluster_cfg.get("clusters", [[1], [1, 2], [1, 2, 3]])

        inmet_aggregated_parquet = paths.get("inmet_aggregated_parquet")
        inmet_station_id = paths.get("inmet_station_id")
        if not inmet_aggregated_parquet or not inmet_station_id:
            raise ValueError(
                "Para mode.cluster + inmet_legacy, informe paths.inmet_aggregated_parquet "
                "e paths.inmet_station_id no YAML."
            )

        build_dataset_clusters_inmet_legacy(
            config_path=args.config,
            sinan_path=sinan_path,
            cnes_path=cnes_path,
            inmet_aggregated_parquet=inmet_aggregated_parquet,
            inmet_station_id=inmet_station_id,
            station_lat=station_lat,
            station_lon=station_lon,
            level_edges_km=level_edges_km,
            clusters=clusters,
            output_base=output_path,
        )
    else:
        raise ValueError(f"Modo não suportado: aggregation={aggregation}, meteo={meteo}")
