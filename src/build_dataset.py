import pandas as pd
import numpy as np
import logging
import argparse
import xarray as xr
import pickle
import yaml

from scipy.spatial import cKDTree

def find_nearest(lat, lon, tree, coords):
    dist, idx = tree.query([[lat, lon]], k=1)
    return coords[idx[0]]

def extract_era5_data(ds, lat, lon, date):
    """Extract ERA5 data for a specific location and date."""

    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    day_data = ds_point.sel(time=slice(f"{date_str}T00:00:00", f"{date_str}T23:00:00"))

    if day_data.time.size == 0:
        raise ValueError(f"No data available for {date_str} at lat={lat}, lon={lon}")

    tem_avg = day_data['t2m'].mean().item() - 273.15
    tem_min = day_data['t2m'].min().item() - 273.15
    tem_max = day_data['t2m'].max().item() - 273.15
    rain = day_data['tp'].sum() # In meters
    

    return tem_avg, tem_min, tem_max, rain

def apply_sliding_window(df, target_col, window_size):
    """Generate sliding window sequences for time-series data."""
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i+window_size].values)
        y.append(df.iloc[i+window_size][target_col])
    return np.array(X), np.array(y)

def build_dataset(id_unidade, sinan_path, cnes_path, era5_path, output_path, config_path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    window_size = config['preproc']['SLIDING_WINDOW_SIZE']
    train_split = config['preproc']['TRAIN_SPLIT']
    val_split = config['preproc']['VAL_SPLIT']

    logging.info("Loading datasets...")
    sinan_df = pd.read_parquet(sinan_path)
    if id_unidade != "FULL":
        sinan_df = sinan_df[sinan_df["ID_UNIDADE"] == id_unidade]
    cnes_df = pd.read_parquet(cnes_path)
    era5_df = xr.open_dataset(era5_path)

    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'])
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].astype(str)
    cnes_df['CNES'] = cnes_df['CNES'].astype(str)

    sinan_df = pd.merge(sinan_df, cnes_df[['CNES', 'LAT', 'LNG']].rename(columns={'CNES': 'ID_UNIDADE'}), on='ID_UNIDADE', how='left')
    sinan_df.dropna(subset=['LAT', 'LNG'], inplace=True)

    sinan_df['TEM_AVG_ERA5'] = np.nan
    sinan_df['TEM_MIN_ERA5'] = np.nan
    sinan_df['TEM_MAX_ERA5'] = np.nan
    sinan_df['RAIN_ERA5'] = 0.0

    logging.info("Extracting ERA5 data...")
    era5_dates = sinan_df['DT_NOTIFIC'].unique()
    era5_coords = sinan_df[['LAT', 'LNG']].drop_duplicates()
    era5_tree = cKDTree(era5_coords[['LAT', 'LNG']].values)  

    for date in era5_dates:
        for lat, lon in era5_coords.values:
            nearest_coords = find_nearest(lat, lon, era5_tree, era5_coords.values)

            logging.debug(f"Original LAT: {lat}, LON: {lon} | Nearest LAT: {nearest_coords[0]}, LON: {nearest_coords[1]}")
            
            era5_avg, era5_min, era5_max, era5_rain = extract_era5_data(era5_df, nearest_coords[0], nearest_coords[1], date)

            logging.debug(f"Extracted ERA5 Data - Date: {date}, Nearest LAT: {nearest_coords[0]}, "
                        f"LON: {nearest_coords[1]}, T_Avg: {era5_avg:.2f}, T_Min: {era5_min:.2f}, T_Max: {era5_max:.2f}")

            # Directly update the DataFrame where DT_NOTIFIC == date
            sinan_df.loc[(sinan_df['DT_NOTIFIC'] == date), 
                        ['TEM_AVG_ERA5', 'TEM_MIN_ERA5', 'TEM_MAX_ERA5', 'RAIN_ERA5']] = era5_avg, era5_min, era5_max, era5_rain


    logging.info("Creating additional ERA5 features...")
    sinan_df['IDEAL_TEMP_ERA5'] = sinan_df['TEM_AVG_ERA5'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
    sinan_df['EXTREME_TEMP_ERA5'] = sinan_df['TEM_AVG_ERA5'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)
    sinan_df['SIGNIFICANT_RAIN_ERA5'] = sinan_df['RAIN_ERA5'].apply(lambda x: 1 if 0.010 <= x < 0.150 else 0)
    sinan_df['EXTREME_RAIN_ERA5'] = sinan_df['RAIN_ERA5'].apply(lambda x: 1 if x >= 0.150 else 0)

    logging.info("Creating rolling metrics and accumulations...")
    windows = [7, 14, 21, 28]
    for window in windows:
        sinan_df[f'TEM_AVG_ERA5_MM_{window}'] = sinan_df['TEM_AVG_ERA5'].rolling(window=window).mean()

        sinan_df[f'CASES_MM_{window}'] = sinan_df['CASES'].rolling(window=window).mean()
        sinan_df[f'CASES_ACC_{window}'] = sinan_df['CASES'].rolling(window=window).sum()
        
        sinan_df[f'RAIN_ERA5_ACC_{window}'] = sinan_df['RAIN_ERA5'].rolling(window=window).sum()
        sinan_df[f'RAIN_ERA5_MM_{window}'] = sinan_df['RAIN_ERA5'].rolling(window=window).mean()

    logging.info("Normalizing features...")
    # sinan_df[['TEM_AVG_ERA5', 'TEM_MIN_ERA5', 'TEM_MAX_ERA5', 'RAIN_ERA5']] = (sinan_df[['TEM_AVG_ERA5', 'TEM_MIN_ERA5', 'TEM_MAX_ERA5', 'RAIN_ERA5']] - 
    #                                                                            sinan_df[['TEM_AVG_ERA5', 'TEM_MIN_ERA5', 'TEM_MAX_ERA5', 'RAIN_ERA5']].min()) / \
    #                                                                            (sinan_df[['TEM_AVG_ERA5', 'TEM_MIN_ERA5', 'TEM_MAX_ERA5', 'RAIN_ERA5']].max() - 
    #                                                                             sinan_df[['TEM_AVG_ERA5', 'TEM_MIN_ERA5', 'TEM_MAX_ERA5', 'RAIN_ERA5']].min())
    
    sinan_df['TEMP_RANGE_ERA5'] = sinan_df['TEM_MAX_ERA5'] - sinan_df['TEM_MIN_ERA5']
    for window in windows:
        sinan_df[f'TEMP_RANGE_ERA5_MM_{window}'] = sinan_df['TEMP_RANGE_ERA5'].rolling(window=window).mean()
   
    sinan_df = sinan_df.drop(columns=['ID_UNIDADE'])
    sinan_df.dropna(inplace=True)
    logging.info("Splitting data into train/val/test...")
    if isinstance(train_split, str):
        train_split_date = pd.to_datetime(train_split)
        train_df = sinan_df[sinan_df['DT_NOTIFIC'] < train_split_date]
        remaining_df = sinan_df[sinan_df['DT_NOTIFIC'] >= train_split_date]
    else:
        split_idx_train = int(len(sinan_df) * train_split)
        train_df = sinan_df.iloc[:split_idx_train]
        remaining_df = sinan_df.iloc[split_idx_train:]

    if isinstance(val_split, str):
        val_split_date = pd.to_datetime(val_split)
        val_df = remaining_df[remaining_df['DT_NOTIFIC'] < val_split_date]
        test_df = remaining_df[remaining_df['DT_NOTIFIC'] >= val_split_date]
    else:
        split_idx_val = int(len(remaining_df) * val_split)
        val_df = remaining_df.iloc[:split_idx_val]
        test_df = remaining_df.iloc[split_idx_val:]

    train_df = train_df.drop(columns=['DT_NOTIFIC', 'LAT', 'LNG'])
    val_df = val_df.drop(columns=['DT_NOTIFIC', 'LAT', 'LNG'])
    test_df = test_df.drop(columns=['DT_NOTIFIC', 'LAT', 'LNG'])
    train_df.to_csv('train.csv')

    logging.info("Applying sliding window...")
    X_train, y_train = apply_sliding_window(train_df, target_col=config['target'], window_size=window_size)
    X_val, y_val = apply_sliding_window(val_df, target_col=config['target'], window_size=window_size)
    X_test, y_test = apply_sliding_window(test_df, target_col=config['target'], window_size=window_size)

    logging.info("Saving processed datasets...")
    with open(output_path, 'wb') as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)

    logging.info(f"Dataset saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Build time-series dataset with sliding window")
    parser.add_argument("id_unidade", help="ID UNIDADE/CNES")
    parser.add_argument("sinan_path", help="Path to SINAN data")
    parser.add_argument("cnes_path", help="Path to CNES data")
    parser.add_argument("era5_path", help="Path to ERA5 data")
    parser.add_argument("output_path", help="Output path for processed dataset")
    parser.add_argument("config_path", help="Path to configuration YAML file")
    args = parser.parse_args()

    build_dataset(
        id_unidade=args.id_unidade,
        sinan_path=args.sinan_path,
        cnes_path=args.cnes_path,
        era5_path=args.era5_path,
        output_path=args.output_path,
        config_path=args.config_path
    )

if __name__ == "__main__":
    #main()
    SINAN_PATH="data\processed\sinan\DENG.parquet"
    CNES_PATH="data\processed\cnes\STRJ2401.parquet"
    ERA5_PATH=fr"data\raw\era5\RJ_1997_2024.nc"
    CONFIG_PATH="config\config.yaml"
    OUTPUT_PATH=fr"data\datasets\2268922.pickle"
    ID_UNIDADE="2268922"
    build_dataset(id_unidade=ID_UNIDADE,
                  sinan_path=SINAN_PATH,
                  cnes_path=CNES_PATH,
                  era5_path=ERA5_PATH,
                  output_path=OUTPUT_PATH,
                  config_path=CONFIG_PATH)