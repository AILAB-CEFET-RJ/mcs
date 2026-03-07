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
from tqdm import tqdm
import json

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


# Função auxiliar de matching espacial
def find_nearest(lat, lon, tree, coords):
    dist, idx = tree.query([[lat, lon]], k=1)
    return coords[idx[0]]


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

# Processamento total
def build_dataset_era5(config_path, sinan_path, cnes_path, era5_path, output_path, id_unidade):
    """Modo antigo: por unidade + ERA5."""
    config = FeatureConfig(config_path)

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

    era5_ds = xr.open_dataset(era5_path)
    era5_lat = era5_ds.latitude.values
    era5_lon = era5_ds.longitude.values
    grid_coords = np.array([(lat, lon) for lat in era5_lat for lon in era5_lon])
    era5_tree = cKDTree(grid_coords)

    sinan_coords = sinan_df[['LAT', 'LNG']].values
    nearest_era5_coords = np.apply_along_axis(lambda x: find_nearest(x[0], x[1], era5_tree, pd.DataFrame(grid_coords, columns=['LAT', 'LNG']).values), 1, sinan_coords)
    sinan_df['LAT_ERA5'], sinan_df['LNG_ERA5'] = nearest_era5_coords[:, 0], nearest_era5_coords[:, 1]

    era5_records = []
    dt_col = 'DT_SEMANA' if config.weekly else 'DT_NOTIFIC'
    unique_dates = sinan_df[[dt_col, 'LAT_ERA5', 'LNG_ERA5']].drop_duplicates()

    for _, row in tqdm(unique_dates.iterrows(), total=len(unique_dates), desc="ERA5 extraction"):
        date = row[dt_col]
        lat, lon = row['LAT_ERA5'], row['LNG_ERA5']
        try:
            vals = extract_era5_data(era5_ds, lat, lon, date, config)
            era5_records.append([date, lat, lon, *vals.values()])
        except ValueError:
            continue

    era5_df = pd.DataFrame(era5_records, columns=[dt_col, 'LAT', 'LNG'] + list(vals.keys()))    
    era5_df[dt_col] = pd.to_datetime(era5_df[dt_col])
    sinan_df = sinan_df.merge(
        era5_df.rename(columns={'LAT': 'LAT_ERA5', 'LNG': 'LNG_ERA5'}),
        on=[dt_col, 'LAT_ERA5', 'LNG_ERA5'],
        how='left'
    )
    sinan_df.drop(columns=['LAT_ERA5', 'LNG_ERA5', 'LAT', 'LNG'], inplace=True)

    if config.weekly:
        sinan_df.rename(columns={'DT_SEMANA': 'DT_NOTIFIC'}, inplace=True)
        
    sinan_df.to_csv('teste.csv')

    train_date = pd.to_datetime(config.train_split)
    val_date = pd.to_datetime(config.val_split)

    train = sinan_df[sinan_df['DT_NOTIFIC'] < train_date]
    val = sinan_df[(sinan_df['DT_NOTIFIC'] >= train_date) & (sinan_df['DT_NOTIFIC'] < val_date)]
    test = sinan_df[sinan_df['DT_NOTIFIC'] >= val_date]

    logging.info("🧪 Feature engineering...")
    X_train, y_train, ids_train = create_new_features(train, "train", config, output_path)
    X_val,   y_val,   ids_val   = create_new_features(val,   "val",   config, output_path)
    X_test,  y_test,  ids_test  = create_new_features(test,  "test",  config, output_path)

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
