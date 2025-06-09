import pandas as pd
import numpy as np
import logging
import argparse
import xarray as xr
import pickle
import yaml
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree

INMET = "INMET"
ERA5 = "ERA5"
FULL = "FULL"

def find_nearest(lat, lon, tree, coords):
    dist, idx = tree.query([[lat, lon]], k=1)
    return coords[idx[0]]

def extract_era5_data(ds, lat, lon, date, isweekly):
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    if isweekly:
        week_start = pd.Timestamp(date)
        week_end = week_start + pd.Timedelta(days=6)
        day_data = ds_point.sel(time=slice(f"{week_start.strftime('%Y-%m-%d')}T00:00:00", f"{week_end.strftime('%Y-%m-%d')}T23:00:00"))
    else:
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        day_data = ds_point.sel(time=slice(f"{date_str}T00:00:00", f"{date_str}T23:00:00"))

    if day_data.time.size == 0:
        raise ValueError(f"No data available for {date} at lat={lat}, lon={lon}")

    tem_avg = day_data['t2m'].mean().item() - 273.15
    tem_min = day_data['t2m'].min().item() - 273.15
    tem_max = day_data['t2m'].max().item() - 273.15
    rain = day_data['tp'].sum().item()

    t2m = day_data['t2m'].values
    d2m = day_data['d2m'].values

    def saturation_vapor_pressure(temp):
        return 6.112 * np.exp((17.67 * (temp - 273.15)) / (temp - 29.65))

    rh = 100 * (saturation_vapor_pressure(d2m) / saturation_vapor_pressure(t2m))

    rh_avg = np.mean(rh)
    rh_min = np.min(rh)
    rh_max = np.max(rh)

    return tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max

def create_era5_dataset(era5_df, sinan_df, era5_tree, era5_coords, isweekly):
    sinan_coords = sinan_df[['LAT', 'LNG']].values
    nearest_era5_coords = np.apply_along_axis(
        lambda x: find_nearest(x[0], x[1], era5_tree, era5_coords.values), 
        1, 
        sinan_coords
    )

    sinan_df['closest_LAT_ERA5'] = nearest_era5_coords[:, 0]
    sinan_df['closest_LNG_ERA5'] = nearest_era5_coords[:, 1]

    era5_data = []

    if isweekly:
        unique_combinations = sinan_df[['DT_SEMANA', 'closest_LAT_ERA5', 'closest_LNG_ERA5']].drop_duplicates()
    else:
        unique_combinations = sinan_df[['DT_NOTIFIC', 'closest_LAT_ERA5', 'closest_LNG_ERA5']].drop_duplicates()

    for _, row in tqdm(unique_combinations.iterrows(), total=len(unique_combinations), desc="Extracting ERA5 data"):
        date = row['DT_SEMANA'] if isweekly else row['DT_NOTIFIC']
        lat, lon = row['closest_LAT_ERA5'], row['closest_LNG_ERA5']
        try:
            tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max = extract_era5_data(era5_df, lat, lon, date, isweekly)
            era5_data.append([date, lat, lon, tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max])
        except ValueError as e:
            logging.warning(f"Skipping data extraction for {date} at lat={lat}, lon={lon}: {e}")
            continue

    era5_data_df = pd.DataFrame(
        era5_data, 
        columns=['DT_NOTIFIC', 'LAT', 'LNG', 'TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN', 'RH_AVG', 'RH_MIN', 'RH_MAX']
    )
    era5_data_df['DT_NOTIFIC'] = pd.to_datetime(era5_data_df['DT_NOTIFIC'])
    return era5_data_df

def apply_sliding_window(X, y, window_size):
    X_windows = np.lib.stride_tricks.sliding_window_view(X, (window_size, X.shape[1]))
    y_aligned = y[window_size - 1:]
    return X_windows.squeeze(), y_aligned

def create_new_features(df: pd.DataFrame, subset: str = "", casesonly: bool = False):
    windows = [7, 14, 21, 28]
    if casesonly:
        for window in windows:
            df[f'CASES_MM_{window}'] = df['CASES'].rolling(window=window).mean()
            df[f'CASES_ACC_{window}'] = df['CASES'].rolling(window=window).sum()

        for lag in range(1, 7):
            df[f'CASES_LAG_{lag}'] = df['CASES'].shift(lag)    
    else:
        df['IDEAL_TEMP'] = df['TEM_AVG'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
        df['EXTREME_TEMP'] = df['TEM_AVG'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)
        df['SIGNIFICANT_RAIN'] = df['RAIN'].apply(lambda x: 1 if 0.010 <= x < 0.150 else 0)
        df['EXTREME_RAIN'] = df['RAIN'].apply(lambda x: 1 if x >= 0.150 else 0)
        df['TEMP_RANGE'] = df['TEM_MAX'] - df['TEM_MIN']
        df['WEEK_OF_YEAR'] = df['DT_NOTIFIC'].dt.isocalendar().week

        for window in windows:
            df[f'TEM_AVG_MM_{window}'] = df['TEM_AVG'].rolling(window=window).mean()
            df[f'CASES_MM_{window}'] = df['CASES'].rolling(window=window).mean()
            df[f'CASES_ACC_{window}'] = df['CASES'].rolling(window=window).sum()
            df[f'RAIN_ACC_{window}'] = df['RAIN'].rolling(window=window).sum()
            df[f'RAIN_MM_{window}'] = df['RAIN'].rolling(window=window).mean()
            df[f'RH_MM_{window}'] = df['RH_AVG'].rolling(window=window).mean()
            df[f'TEMP_RANGE_MM_{window}'] = df['TEMP_RANGE'].rolling(window=window).mean()

        for lag in range(1, 7):
            df[f'CASES_LAG_{lag}'] = df['CASES'].shift(lag)

    df = df.drop(columns=[col for col in ['DT_NOTIFIC', 'LAT', 'LNG', 'ID_UNIDADE'] if col in df.columns])
    df.dropna(inplace=True)

    X = df.drop(columns=['CASES']).to_numpy()
    y = df['CASES'].to_numpy()

    feature_names = df.drop(columns=['CASES']).columns.tolist()
    feature_dict = pd.DataFrame({"Index": range(len(feature_names)), "Feature": feature_names})
    feature_dict_name = "feature_dictionary"
    if(casesonly):
        feature_dict_name = feature_dict_name + "_cases"
    feature_dict.to_csv(f"{feature_dict_name}.csv", index=False)

    logging.info(f"{subset} — Tamanho total: {len(y)}, Zeros: {(y == 0).sum()}, Não-Zeros: {(y > 0).sum()}")

    return X, y

def build_dataset(id_unidade, sinan_path, cnes_path, era5_path, output_path, config_path, isweekly, casesonly):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    window_size = config['preproc']['SLIDING_WINDOW_SIZE']
    train_split = config['preproc']['TRAIN_SPLIT']
    val_split = config['preproc']['VAL_SPLIT']

    logging.info("Loading datasets...")
    sinan_df = pd.read_parquet(sinan_path)
    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'])
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].astype(str)

    if id_unidade != FULL:
        val_split_date = pd.to_datetime(val_split)
        unidade_df = sinan_df[(sinan_df['DT_NOTIFIC'] >= val_split_date) & (sinan_df["ID_UNIDADE"] == id_unidade)].copy()
        temp_df = sinan_df[(sinan_df['DT_NOTIFIC'] < val_split_date) & (sinan_df["ID_UNIDADE"] != id_unidade)].copy()
        sinan_df = pd.concat([temp_df, unidade_df], ignore_index=True)

    cnes_df = pd.read_parquet(cnes_path)
    cnes_df['CNES'] = cnes_df['CNES'].astype(str)
    sinan_df = pd.merge(sinan_df, cnes_df[['CNES', 'LAT', 'LNG']].rename(columns={'CNES': 'ID_UNIDADE'}), on='ID_UNIDADE', how='left')
    sinan_df.dropna(subset=['LAT', 'LNG'], inplace=True)

    if isweekly:
        sinan_df['DT_SEMANA'] = sinan_df['DT_NOTIFIC'].dt.to_period('W').apply(lambda r: r.start_time)
        sinan_df = sinan_df.groupby(['ID_UNIDADE', 'DT_SEMANA']).agg({'CASES': 'sum','LAT': 'first','LNG': 'first'}).reset_index()

    if not casesonly:
        logging.info("Merging datasets...")
        era5_df = xr.open_dataset(era5_path)
        logging.info("Extracting ERA5 data...")

        # CORRIGIDO: coordenadas reais da grade ERA5
        era5_lat = era5_df.latitude.values
        era5_lon = era5_df.longitude.values
        grid_coords = np.array([(lat, lon) for lat in era5_lat for lon in era5_lon])
        era5_tree = cKDTree(grid_coords)

        era5_data_df = create_era5_dataset(era5_df, sinan_df, era5_tree, pd.DataFrame(grid_coords, columns=['LAT', 'LNG']), isweekly)

        sinan_df.drop(columns=[col for col in ['LAT', 'LNG'] if col in sinan_df.columns], inplace=True)
        logging.info("Merging ERA5 data with SINAN data...")
        if isweekly:
            sinan_df = pd.merge(sinan_df, era5_data_df, left_on=['DT_SEMANA', 'closest_LAT_ERA5', 'closest_LNG_ERA5'], right_on=['DT_NOTIFIC', 'LAT', 'LNG'], how='left')
            sinan_df = sinan_df.rename(columns={'DT_SEMANA': 'DT_NOTIFIC'})
            sinan_df = sinan_df.loc[:, ~sinan_df.columns.duplicated()]
        else:
            sinan_df = pd.merge(sinan_df, era5_data_df, left_on=['DT_NOTIFIC', 'closest_LAT_ERA5', 'closest_LNG_ERA5'], right_on=['DT_NOTIFIC', 'LAT', 'LNG'], how='left')

        sinan_df.drop(columns=['closest_LAT_ERA5', 'closest_LNG_ERA5'], inplace=True)
    else:
        sinan_df = sinan_df.rename(columns={'DT_SEMANA': 'DT_NOTIFIC'})
        sinan_df.drop(columns=[col for col in ['LAT', 'LNG'] if col in sinan_df.columns], inplace=True)

    logging.info("Splitting data into train/val/test...")
    train_split_date = pd.to_datetime(train_split)
    val_split_date = pd.to_datetime(val_split)
    train_df = sinan_df[sinan_df['DT_NOTIFIC'] < train_split_date].copy()
    remaining_df = sinan_df[sinan_df['DT_NOTIFIC'] >= train_split_date].copy()
    val_df = remaining_df[remaining_df['DT_NOTIFIC'] < val_split_date].copy()
    test_df = remaining_df[remaining_df['DT_NOTIFIC'] >= val_split_date].copy()

    logging.info("Creating additional features for train...")
    X_train, y_train = create_new_features(train_df, "Train", casesonly)
    logging.info("Creating additional features for val...")
    X_val, y_val = create_new_features(val_df, "Val", casesonly)
    logging.info("Creating additional features for test...")
    X_test, y_test = create_new_features(test_df, "Test", casesonly)

    logging.info("Scaling columns")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train, y_train = apply_sliding_window(X_train, y_train, window_size)
    X_val, y_val = apply_sliding_window(X_val, y_val, window_size)
    X_test, y_test = apply_sliding_window(X_test, y_test, window_size)

    logging.info("Saving processed datasets...")
    with open(output_path, 'wb') as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)

    logging.info(f"Dataset saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Build time-series dataset with sliding window")
    parser.add_argument("id_unidade", help="ID UNIDADE/CNES")
    parser.add_argument("sinan_path", help="Path to SINAN data")
    parser.add_argument("cnes_path", help="Path to CNES data")
    parser.add_argument("era5_path", help="Path to ERA5 dataset")
    parser.add_argument("output_path", help="Output path for processed dataset")
    parser.add_argument("config_path", help="Path to configuration YAML file")
    parser.add_argument("isweekly", help="Aggregate weekly values")
    parser.add_argument("casesonly", help="Use only case data")
    args = parser.parse_args()
    
    isweekly = args.isweekly.lower() == 'true'
    casesonly = args.casesonly.lower() == 'true'

    build_dataset(
        id_unidade=args.id_unidade,
        sinan_path=args.sinan_path,
        cnes_path=args.cnes_path,
        era5_path=args.era5_path,
        output_path=args.output_path,
        config_path=args.config_path,
        isweekly=isweekly,
        casesonly=casesonly
    )

if __name__ == "__main__":
    main()
