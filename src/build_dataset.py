import pandas as pd
import numpy as np
import logging
import argparse
import xarray as xr
import pickle
import yaml

from scipy.spatial import cKDTree

INMET = "INMET"
ERA5 = "ERA5"
FULL = "FULL"

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

def build_dataset(id_unidade, sinan_path, cnes_path, meteo_origin, dataset_path, output_path, config_path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    window_size = config['preproc']['SLIDING_WINDOW_SIZE']
    train_split = config['preproc']['TRAIN_SPLIT']
    val_split = config['preproc']['VAL_SPLIT']

    logging.info("Loading datasets...")
    sinan_df = pd.read_parquet(sinan_path)
    if id_unidade != FULL:
        sinan_df = sinan_df[sinan_df["ID_UNIDADE"] == id_unidade]
    cnes_df = pd.read_parquet(cnes_path)

    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'])
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].astype(str)
    cnes_df['CNES'] = cnes_df['CNES'].astype(str)

    sinan_df = pd.merge(sinan_df, cnes_df[['CNES', 'LAT', 'LNG']].rename(columns={'CNES': 'ID_UNIDADE'}), on='ID_UNIDADE', how='left')
    sinan_df.dropna(subset=['LAT', 'LNG'], inplace=True)

    if meteo_origin == ERA5:
        sinan_df['TEM_AVG'] = np.nan
        sinan_df['TEM_MIN'] = np.nan
        sinan_df['TEM_MAX'] = np.nan
        sinan_df['RAIN'] = 0.0
        era5_df = xr.open_dataset(dataset_path)
        logging.info("Extracting ERA5 data...")
        era5_dates = sinan_df['DT_NOTIFIC'].unique()
        era5_coords = sinan_df[['LAT', 'LNG']].drop_duplicates()
        era5_tree = cKDTree(era5_coords[['LAT', 'LNG']].values)  

        for date in era5_dates:
            for lat, lon in era5_coords.values:
                nearest_coords = find_nearest(lat, lon, era5_tree, era5_coords.values)

                logging.debug(f"Original LAT: {lat}, LON: {lon} | Nearest LAT: {nearest_coords[0]}, LON: {nearest_coords[1]}")
                
                if meteo_origin == ERA5:
                    era5_avg, era5_min, era5_max, era5_rain = extract_era5_data(era5_df, nearest_coords[0], nearest_coords[1], date)

                logging.debug(f"Extracted ERA5 Data - Date: {date}, Nearest LAT: {nearest_coords[0]}, "
                            f"LON: {nearest_coords[1]}, T_Avg: {era5_avg:.2f}, T_Min: {era5_min:.2f}, T_Max: {era5_max:.2f}")

                # Directly update the DataFrame where DT_NOTIFIC == date
                sinan_df.loc[(sinan_df['DT_NOTIFIC'] == date), 
                            ['TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN']] = era5_avg, era5_min, era5_max, era5_rain        
    elif meteo_origin == INMET:
        logging.info("Extracting INMET data...")
        inmet_df = pd.read_parquet(dataset_path)
        inmet_df['DT_MEDICAO'] = pd.to_datetime(inmet_df['DT_MEDICAO'], format='%Y-%m-%d')
        inmet_coords = inmet_df[['VL_LATITUDE', 'VL_LONGITUDE']].values
        
        tree_inmet = cKDTree(inmet_coords)
        
        nearest_inmet = np.apply_along_axis(lambda x: find_nearest(x[0], x[1], tree_inmet, inmet_coords), 1, sinan_df[['LAT', 'LNG']].values)        
        inmet_df.rename(columns={'CHUVA': 'RAIN'}, inplace=True)        
        
        sinan_df['closest_LAT_INMET'] = nearest_inmet[:, 0]
        sinan_df['closest_LNG_INMET'] = nearest_inmet[:, 1]

        logging.info("Merging INMET data...")
        sinan_df = pd.merge(sinan_df, inmet_df, left_on=['closest_LAT_INMET', 'closest_LNG_INMET', 'DT_NOTIFIC'], right_on=['VL_LATITUDE', 'VL_LONGITUDE', 'DT_MEDICAO'], how='left')
        sinan_df = sinan_df.drop(columns=['closest_LAT_INMET', 'closest_LNG_INMET', 'DT_MEDICAO', 'CD_ESTACAO'])

    logging.info("Creating additional features...")
    sinan_df['IDEAL_TEMP'] = sinan_df['TEM_AVG'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
    sinan_df['EXTREME_TEMP'] = sinan_df['TEM_AVG'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)
    sinan_df['SIGNIFICANT_RAIN'] = sinan_df['RAIN'].apply(lambda x: 1 if 0.010 <= x < 0.150 else 0)
    sinan_df['EXTREME_RAIN'] = sinan_df['RAIN'].apply(lambda x: 1 if x >= 0.150 else 0)

    logging.info("Creating rolling metrics and accumulations...")
    windows = [7, 14, 21, 28]
    for window in windows:
        sinan_df[f'TEM_AVG_MM_{window}'] = sinan_df['TEM_AVG'].rolling(window=window).mean()

        sinan_df[f'CASES_MM_{window}'] = sinan_df['CASES'].rolling(window=window).mean()
        sinan_df[f'CASES_ACC_{window}'] = sinan_df['CASES'].rolling(window=window).sum()
        
        sinan_df[f'RAIN_ACC_{window}'] = sinan_df['RAIN'].rolling(window=window).sum()
        sinan_df[f'RAIN_MM_{window}'] = sinan_df['RAIN'].rolling(window=window).mean()

    logging.info("Normalizing features...")
    # sinan_df[['TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN']] = (sinan_df[['TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN']] - 
    #                                                                            sinan_df[['TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN']].min()) / \
    #                                                                            (sinan_df[['TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN']].max() - 
    #                                                                             sinan_df[['TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN']].min())
    
    sinan_df['TEMP_RANGE'] = sinan_df['TEM_MAX'] - sinan_df['TEM_MIN']
    for window in windows:
        sinan_df[f'TEMP_RANGE_MM_{window}'] = sinan_df['TEMP_RANGE'].rolling(window=window).mean()
   
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
    parser.add_argument("meteo_origin", help="Origin of temperature/rain data", choices=[INMET, ERA5], default=INMET)
    parser.add_argument("dataset_path", help="Path to temperature/rain dataset")
    parser.add_argument("output_path", help="Output path for processed dataset")
    parser.add_argument("config_path", help="Path to configuration YAML file")
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
    # SINAN_PATH="data\processed\sinan\DENG.parquet"
    # CNES_PATH="data\processed\cnes\STRJ2401.parquet"
    # #DATASET_PATH=fr"data\raw\era5\RJ_1997_2024.nc"
    # DATASET_PATH=fr"data/processed/inmet/aggregated.parquet"
    # METEO_ORIGIN=INMET
    # CONFIG_PATH="config\config.yaml"
    # OUTPUT_PATH=fr"data\datasets\2268922.pickle"
    # ID_UNIDADE="2268922"
    # build_dataset(id_unidade=ID_UNIDADE,
    #               sinan_path=SINAN_PATH,
    #               cnes_path=CNES_PATH,
    #               meteo_origin=METEO_ORIGIN,
    #               dataset_path=DATASET_PATH,
    #               output_path=OUTPUT_PATH,
    #               config_path=CONFIG_PATH)