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

def extract_era5_data(ds, lat, lon, date):
    """Extract ERA5 data for a specific location and date, including relative humidity calculations."""

    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    day_data = ds_point.sel(time=slice(f"{date_str}T00:00:00", f"{date_str}T23:00:00"))

    if day_data.time.size == 0:
        raise ValueError(f"No data available for {date_str} at lat={lat}, lon={lon}")

    # Extract temperature and precipitation
    tem_avg = day_data['t2m'].mean().item() - 273.15  # Convert from Kelvin to Celsius
    tem_min = day_data['t2m'].min().item() - 273.15
    tem_max = day_data['t2m'].max().item() - 273.15
    rain = day_data['tp'].sum().item()  # Total precipitation in meters

    # Extract temperature and dew point (needed for RH calculation)
    t2m = day_data['t2m'].values  # Air temperature in Kelvin
    d2m = day_data['d2m'].values  # Dew point temperature in Kelvin

    # Function to compute saturation vapor pressure
    def saturation_vapor_pressure(temp):
        return 6.112 * np.exp((17.67 * (temp - 273.15)) / (temp - 29.65))

    # Compute relative humidity
    rh = 100 * (saturation_vapor_pressure(d2m) / saturation_vapor_pressure(t2m))

    # Compute min, max, and avg relative humidity
    rh_avg = np.mean(rh)
    rh_min = np.min(rh)
    rh_max = np.max(rh)

    return tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max

def create_era5_dataset(era5_df, sinan_df, era5_tree, era5_coords):
    """Create a dataset with ERA5 data mapped to the nearest coordinates for the sinan_df points."""
    
    # Calculate nearest ERA5 coordinates for all LAT/LNG pairs in sinan_df
    sinan_coords = sinan_df[['LAT', 'LNG']].values
    nearest_era5_coords = np.apply_along_axis(
        lambda x: find_nearest(x[0], x[1], era5_tree, era5_coords.values), 
        1, 
        sinan_coords
    )
    
    # Add the nearest coordinates to the sinan_df
    sinan_df['closest_LAT_ERA5'] = nearest_era5_coords[:, 0]
    sinan_df['closest_LNG_ERA5'] = nearest_era5_coords[:, 1]
    
    # Prepare a list to store ERA5 data
    era5_data = []
    
    # Iterate over all unique date and coordinate combinations
    unique_combinations = sinan_df[['DT_NOTIFIC', 'closest_LAT_ERA5', 'closest_LNG_ERA5']].drop_duplicates()

    for _, row in tqdm(unique_combinations.iterrows(), total=len(unique_combinations), desc="Extracting ERA5 data"):
        date, lat, lon = row['DT_NOTIFIC'], row['closest_LAT_ERA5'], row['closest_LNG_ERA5']
        
        try:
            # Extract temperature, rainfall, and humidity data
            tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max = extract_era5_data(era5_df, lat, lon, date)

            # Append results to list
            era5_data.append([date, lat, lon, tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max])
        
        except ValueError as e:
            logging.warning(f"Skipping data extraction for {date} at lat={lat}, lon={lon}: {e}")
            continue

    # Create a DataFrame from the ERA5 data
    era5_data_df = pd.DataFrame(
        era5_data, 
        columns=['DT_NOTIFIC', 'LAT', 'LNG', 'TEM_AVG', 'TEM_MIN', 'TEM_MAX', 'RAIN', 'RH_AVG', 'RH_MIN', 'RH_MAX']
    )
    
    era5_data_df['DT_NOTIFIC'] = pd.to_datetime(era5_data_df['DT_NOTIFIC'])
    
    return era5_data_df

def apply_sliding_window(X, y, window_size):
    """Generate sliding window sequences for time-series data using NumPy arrays."""
    X_windows = np.lib.stride_tricks.sliding_window_view(X, (window_size, X.shape[1]))
    y_aligned = y[window_size - 1:]  
    return X_windows.squeeze(), y_aligned

def create_new_features(df: pd.DataFrame):
    df['IDEAL_TEMP'] = df['TEM_AVG'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
    df['EXTREME_TEMP'] = df['TEM_AVG'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)
    df['SIGNIFICANT_RAIN'] = df['RAIN'].apply(lambda x: 1 if 0.010 <= x < 0.150 else 0)
    df['EXTREME_RAIN'] = df['RAIN'].apply(lambda x: 1 if x >= 0.150 else 0)
    df['TEMP_RANGE'] = df['TEM_MAX'] - df['TEM_MIN']
    df['WEEK_OF_YEAR'] = df['DT_NOTIFIC'].dt.isocalendar().week

    windows = [7, 14, 21, 28]
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

    # Removendo colunas em branco
    df = df.drop(columns=['DT_NOTIFIC', 'LAT', 'LNG', 'ID_UNIDADE'])
    df.dropna(inplace=True)

    return df.drop(columns=['CASES']).to_numpy(), df['CASES'].to_numpy()

def build_dataset(id_unidade, sinan_path, cnes_path, meteo_origin, dataset_path, output_path, config_path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    window_size = config['preproc']['SLIDING_WINDOW_SIZE']
    train_split = config['preproc']['TRAIN_SPLIT']
    val_split = config['preproc']['VAL_SPLIT']
    target_col = config['target']

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

    logging.info("Merging datasets...")
    if meteo_origin == ERA5:
        era5_df = xr.open_dataset(dataset_path)
        logging.info("Extracting ERA5 data...")
        sinan_df['LAT'] = -22.861389
        sinan_df['LNG'] = -43.411389
        era5_coords = sinan_df[['LAT', 'LNG']].drop_duplicates()
        era5_tree = cKDTree(era5_coords)  
        era5_data_df = create_era5_dataset(era5_df, sinan_df, era5_tree, era5_coords)
        sinan_df = sinan_df.drop(columns=['LAT', 'LNG'])
        logging.info("Merging ERA5 data with SINAN data...")
        sinan_df = pd.merge(sinan_df, era5_data_df, left_on=['DT_NOTIFIC', 'closest_LAT_ERA5', 'closest_LNG_ERA5'], right_on=['DT_NOTIFIC', 'LAT', 'LNG'], how='left')
        sinan_df = sinan_df.drop(columns=['closest_LAT_ERA5', 'closest_LNG_ERA5'])

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

    logging.info("Splitting data into train/val/test...")
    #Train
    train_split_date = pd.to_datetime(train_split)
    train_df = sinan_df[sinan_df['DT_NOTIFIC'] < train_split_date]
    remaining_df = sinan_df[sinan_df['DT_NOTIFIC'] >= train_split_date]
    
    #Val
    val_split_date = pd.to_datetime(val_split)
    val_df = remaining_df[remaining_df['DT_NOTIFIC'] < val_split_date]

    #Test
    test_df = remaining_df[remaining_df['DT_NOTIFIC'] >= val_split_date]

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
    # METEO_ORIGIN=ERA5
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