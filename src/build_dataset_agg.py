import pandas as pd
import numpy as np
import logging
import argparse
import xarray as xr
import pickle
import yaml
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def extract_era5_data(ds, lat, lon, date):
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    day_data = ds_point.sel(time=slice(f"{date_str}T00:00:00", f"{date_str}T23:00:00"))

    if day_data.time.size == 0:
        raise ValueError(f"No data available for {date_str} at lat={lat}, lon={lon}")

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

def create_era5_dataset(era5_df, sinan_df):
    era5_data = []

    for _, row in tqdm(sinan_df.iterrows(), total=len(sinan_df), desc="Extracting ERA5 data"):
        date, lat, lon = row['DT_NOTIFIC'], row['LAT'], row['LNG']
        try:
            tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max = extract_era5_data(era5_df, lat, lon, date)
            era5_data.append([date, tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max])
        except ValueError as e:
            logging.warning(f"Skipping {date}: {e}")
            continue

    return pd.DataFrame(era5_data, columns=['DT_NOTIFIC', 'TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN', 'RH_AVG', 'RH_MIN', 'RH_MAX'])

def apply_sliding_window(X, y, window_size):
    X_windows = np.lib.stride_tricks.sliding_window_view(X, (window_size, X.shape[1]))
    y_aligned = y[window_size - 1:]  
    return X_windows.squeeze(), y_aligned

def create_new_features(df: pd.DataFrame):
    df = df.copy()
    df.loc[:, 'IDEAL_TEMP'] = df['TEM_AVG'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
    df.loc[:, 'EXTREME_TEMP'] = df['TEM_AVG'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)
    df.loc[:, 'SIGNIFICANT_RAIN'] = df['RAIN'].apply(lambda x: 1 if 0.010 <= x < 0.150 else 0)
    df.loc[:, 'EXTREME_RAIN'] = df['RAIN'].apply(lambda x: 1 if x >= 0.150 else 0)
    df.loc[:, 'TEMP_RANGE'] = df['TEM_MAX'] - df['TEM_MIN']
    df.loc[:, 'WEEK_OF_YEAR'] = df['DT_NOTIFIC'].dt.isocalendar().week

    windows = [7, 14, 21, 28]
    for window in windows:
        df.loc[:, f'TEM_AVG_MM_{window}'] = df['TEM_AVG'].rolling(window=window).mean()
        df.loc[:, f'CASES_MM_{window}'] = df['CASES'].rolling(window=window).mean()
        df.loc[:, f'CASES_ACC_{window}'] = df['CASES'].rolling(window=window).sum()
        df.loc[:, f'RAIN_ACC_{window}'] = df['RAIN'].rolling(window=window).sum()
        df.loc[:, f'RAIN_MM_{window}'] = df['RAIN'].rolling(window=window).mean()
        df.loc[:, f'RH_MM_{window}'] = df['RH_AVG'].rolling(window=window).mean()
        df.loc[:, f'TEMP_RANGE_MM_{window}'] = df['TEMP_RANGE'].rolling(window=window).mean()

    for lag in range(1, 7):
        df.loc[:, f'CASES_LAG_{lag}'] = df['CASES'].shift(lag)

    df = df.drop(columns=['DT_NOTIFIC'])
    df.dropna(inplace=True)

    return df.drop(columns=['CASES']).to_numpy(), df['CASES'].to_numpy()

def build_dataset(id_unidade, sinan_path, cnes_path, meteo_origin, dataset_path, output_path, config_path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    window_size = config['preproc']['SLIDING_WINDOW_SIZE']
    train_split = config['preproc']['TRAIN_SPLIT']
    val_split = config['preproc']['VAL_SPLIT']
    target_col = config['target']

    logging.info("Loading datasets...")
    sinan_df = pd.read_parquet(sinan_path)
    if id_unidade != "FULL":
        sinan_df = sinan_df[sinan_df["ID_UNIDADE"] == id_unidade]
    cnes_df = pd.read_parquet(cnes_path)

    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'])
    sinan_df = pd.merge(
        sinan_df,
        cnes_df[['CNES']].rename(columns={'CNES': 'ID_UNIDADE'}),
        on='ID_UNIDADE',
        how='left'
    )

    fixed_lat ,fixed_lon = -22.861389, -43.411389

    sinan_df = sinan_df.groupby('DT_NOTIFIC', as_index=False)['CASES'].sum()
    sinan_df['LAT'] = fixed_lat
    sinan_df['LNG'] = fixed_lon

    if meteo_origin == "ERA5":
        era5_df = xr.open_dataset(dataset_path)
        logging.info("Extracting ERA5 data...")
        era5_data_df = create_era5_dataset(era5_df, sinan_df)
        sinan_df = pd.merge(sinan_df, era5_data_df, on='DT_NOTIFIC', how='left')

    logging.info("Splitting data into train/val/test...")
    train_split_date = pd.to_datetime(train_split)
    train_df = sinan_df[sinan_df['DT_NOTIFIC'] < train_split_date]
    remaining_df = sinan_df[sinan_df['DT_NOTIFIC'] >= train_split_date]
    val_split_date = pd.to_datetime(val_split)
    val_df = remaining_df[remaining_df['DT_NOTIFIC'] < val_split_date]
    test_df = remaining_df[remaining_df['DT_NOTIFIC'] >= val_split_date]

    dump_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    dump_df.to_csv('teste.csv')

    logging.info("Creating additional features for train...")
    X_train, y_train = create_new_features(train_df)
    logging.info("Creating additional features for val...")
    X_val, y_val = create_new_features(val_df)
    logging.info("Creating additional features for test...")
    X_test, y_test = create_new_features(test_df)

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
    parser.add_argument("id_unidade")
    parser.add_argument("sinan_path")
    parser.add_argument("cnes_path")
    parser.add_argument("meteo_origin", choices=["INMET", "ERA5"], default="INMET")
    parser.add_argument("dataset_path")
    parser.add_argument("output_path")
    parser.add_argument("config_path")
    args = parser.parse_args()

    build_dataset(
        id_unidade=args.id_unidade,
        sinan_path=args.sinan_path,
        cnes_path=args.cnes_path,
        meteo_origin=args.meteo_origin,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        config_path=args.config_path
    )

if __name__ == "__main__":
    main()
