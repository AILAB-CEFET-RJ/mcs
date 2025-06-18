# src/data/dataset_builder_v3.py

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

from features.feature_config_parser import FeatureConfig
from features.feature_engineering import create_new_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Função auxiliar de matching espacial
def find_nearest(lat, lon, tree, coords):
    dist, idx = tree.query([[lat, lon]], k=1)
    return coords[idx[0]]


# Leitura ERA5
def extract_era5_data(ds, lat, lon, date, config):
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
def build_dataset(config_path, sinan_path, cnes_path, era5_path, output_path, id_unidade):
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
    X_train, y_train = create_new_features(train, "train", config)
    X_val, y_val = create_new_features(val, "val", config)
    X_test, y_test = create_new_features(test, "test", config)

    logging.info("⚖️ Normalizando...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logging.info("💾 Salvando pickle...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)

    logging.info(f"✅ Dataset final salvo em: {output_path}")


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

    build_dataset(
        config_path=args.config,
        sinan_path=sinan_path,
        cnes_path=cnes_path,
        era5_path=era5_path,
        output_path=output_path,
        id_unidade=id_unidade
    )
