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
    """Extract ERA5 data for a specific location and date, including relative humidity calculations."""

    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    if isweekly:
        week_start = pd.Timestamp(date)
        week_end = week_start + pd.Timedelta(days=6)
        day_data = ds_point.sel(time=slice(f"{week_start.strftime('%Y-%m-%d')}T00:00:00", f"{week_end.strftime('%Y-%m-%d')}T23:00:00"))
    else:
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        day_data = ds_point.sel(time=slice(f"{date_str}T00:00:00", f"{date_str}T23:00:00"))

    if day_data.time.size == 0:
        if isweekly:
            raise ValueError(f"No data available for week starting {week_start} at lat={lat}, lon={lon}")
        else:
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

def create_era5_dataset(era5_df, sinan_df, era5_tree, era5_coords, isweekly):
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
    if isweekly:
        unique_combinations = sinan_df[['DT_SEMANA', 'closest_LAT_ERA5', 'closest_LNG_ERA5']].drop_duplicates()
    else:
        unique_combinations = sinan_df[['DT_NOTIFIC', 'closest_LAT_ERA5', 'closest_LNG_ERA5']].drop_duplicates()

    for _, row in tqdm(unique_combinations.iterrows(), total=len(unique_combinations), desc="Extracting ERA5 data"):
        if isweekly:
            date, lat, lon = row['DT_SEMANA'], row['closest_LAT_ERA5'], row['closest_LNG_ERA5']
        else:
            date, lat, lon = row['DT_NOTIFIC'], row['closest_LAT_ERA5'], row['closest_LNG_ERA5']
        
        try:
            # Extract temperature, rainfall, and humidity data
            tem_avg, tem_min, tem_max, rain, rh_avg, rh_min, rh_max = extract_era5_data(era5_df, lat, lon, date, isweekly)

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

def create_new_features(df: pd.DataFrame, zero_fraction: float = 1.0, subset: str = ""):
    df['IDEAL_TEMP'] = df['TEM_AVG'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
    df['EXTREME_TEMP'] = df['TEM_AVG'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)
    df['SIGNIFICANT_RAIN'] = df['RAIN'].apply(lambda x: 1 if 0.010 <= x < 0.150 else 0)
    df['EXTREME_RAIN'] = df['RAIN'].apply(lambda x: 1 if x >= 0.150 else 0)
    df['TEMP_RANGE'] = df['TEM_MAX'] - df['TEM_MIN']
    df['WEEK_OF_YEAR'] = df['DT_NOTIFIC'].dt.isocalendar().week

    windows = [7, 14, 21, 28]
    for window in windows:
        df[f'TEM_AVG_MM_{window}'] = df['TEM_AVG'].rolling(window=window).mean()

        # df[f'CASES_MM_{window}'] = df['CASES'].rolling(window=window).mean()
        # df[f'CASES_ACC_{window}'] = df['CASES'].rolling(window=window).sum()
        
        df[f'RAIN_ACC_{window}'] = df['RAIN'].rolling(window=window).sum()
        df[f'RAIN_MM_{window}'] = df['RAIN'].rolling(window=window).mean()

        df[f'RH_MM_{window}'] = df['RH_AVG'].rolling(window=window).mean()
        df[f'TEMP_RANGE_MM_{window}'] = df['TEMP_RANGE'].rolling(window=window).mean()
    
    # for lag in range(1, 7):
    #     df[f'CASES_LAG_{lag}'] = df['CASES'].shift(lag)

    # Removendo colunas em branco
    df = df.drop(columns=['DT_NOTIFIC', 'LAT', 'LNG', 'ID_UNIDADE'])
    df.dropna(inplace=True)

    if 0 <= zero_fraction < 1.0:
        zeros = df[df['CASES'] == 0]
        non_zeros = df[df['CASES'] > 0]
        keep_n = int(len(zeros) * zero_fraction)
        sampled_zeros = zeros.sample(n=keep_n, random_state=42)
        df = pd.concat([non_zeros, sampled_zeros]).sort_index()

    X = df.drop(columns=['CASES']).to_numpy()
    y = df['CASES'].to_numpy()

    feature_names = df.drop(columns=['CASES']).columns.tolist()
    feature_dict = pd.DataFrame({
        "Index": range(len(feature_names)),
        "Feature": feature_names
    })
    feature_dict.to_csv("feature_dictionary.csv", index=False)

    logging.info(f"{subset} — Tamanho total: {len(y)}, Zeros: {(y == 0).sum()}, Não-Zeros: {(y > 0).sum()}")

    return X, y

def build_dataset(id_unidade, sinan_path, cnes_path, meteo_origin, meteo_path, output_path, config_path, lat, lon, isweekly, zero_fraction):
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
    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'])
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].astype(str)
    val_split_date = pd.to_datetime(val_split)

    if id_unidade != FULL:
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

    logging.info("Merging datasets...")
    if meteo_origin == ERA5:
        era5_df = xr.open_dataset(meteo_path)
        logging.info("Extracting ERA5 data...")
        if lat is not None and lon is not None:
            sinan_df['LAT'] = lat
            sinan_df['LNG'] = lon
        era5_coords = sinan_df[['LAT', 'LNG']].drop_duplicates()
        era5_tree = cKDTree(era5_coords[['LAT', 'LNG']].values)  
        era5_data_df = create_era5_dataset(era5_df, sinan_df, era5_tree, era5_coords, isweekly)
        sinan_df = sinan_df.drop(columns=['LAT', 'LNG'])
        logging.info("Merging ERA5 data with SINAN data...")
        if isweekly:
            sinan_df = pd.merge(sinan_df, era5_data_df, left_on=['DT_SEMANA', 'closest_LAT_ERA5', 'closest_LNG_ERA5'], right_on=['DT_NOTIFIC', 'LAT', 'LNG'], how='left')
            sinan_df = sinan_df.rename(columns={'DT_SEMANA': 'DT_NOTIFIC'})
            sinan_df = sinan_df.loc[:, ~sinan_df.columns.duplicated()]
        else:
            sinan_df = pd.merge(sinan_df, era5_data_df, left_on=['DT_NOTIFIC', 'closest_LAT_ERA5', 'closest_LNG_ERA5'], right_on=['DT_NOTIFIC', 'LAT', 'LNG'], how='left')
        
        sinan_df = sinan_df.drop(columns=['closest_LAT_ERA5', 'closest_LNG_ERA5'])

    elif meteo_origin == INMET:
        logging.info("Extracting INMET data...")
        inmet_df = pd.read_parquet(meteo_path)
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
    train_df = sinan_df[sinan_df['DT_NOTIFIC'] < train_split_date].copy()
    remaining_df = sinan_df[sinan_df['DT_NOTIFIC'] >= train_split_date].copy()
    
    #Val
    val_split_date = pd.to_datetime(val_split)
    val_df = remaining_df[remaining_df['DT_NOTIFIC'] < val_split_date].copy()

    #Test
    test_df = remaining_df[remaining_df['DT_NOTIFIC'] >= val_split_date].copy()

    logging.info("Creating additional features for train...")
    X_train, y_train = create_new_features(train_df, zero_fraction)

    logging.info("Creating additional features for val...")
    X_val, y_val = create_new_features(val_df, zero_fraction)

    logging.info("Creating additional features for test...")
    X_test, y_test = create_new_features(test_df, 1.0)

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
    parser.add_argument("meteo_path", help="Path to temperature/rain dataset")
    parser.add_argument("output_path", help="Output path for processed dataset")
    parser.add_argument("config_path", help="Path to configuration YAML file")
    parser.add_argument("lat", help="Override the LAT for the whole dataset")
    parser.add_argument("lon", help="Override the LON for the whole dataset")
    parser.add_argument("isweekly", help="Aggregate weekly values")
    parser.add_argument("zero_fraction", help="How many percent of 0 cases in the final dataset")
    args = parser.parse_args()

    isweekly = args.isweekly.lower() == 'true'
    lat = float(args.lat) if args.lat.lower() != 'none' else None
    lon = float(args.lon) if args.lon.lower() != 'none' else None
    zero_fraction = float(args.zero_fraction)

    build_dataset(
        id_unidade=args.id_unidade,
        sinan_path=args.sinan_path,
        cnes_path=args.cnes_path,
        meteo_origin=args.meteo_origin,
        meteo_path=args.meteo_path,
        output_path=args.output_path,
        config_path=args.config_path,
        lat=lat,
        lon=lon,
        isweekly=isweekly,
        zero_fraction=zero_fraction
    )

if __name__ == "__main__":
    main()
    # SINAN_PATH="data\processed\sinan\DENG.parquet"
    # CNES_PATH="data\processed\cnes\STRJ2401.parquet"
    # #DATASET_PATH=fr"data\raw\era5\RJ_1997_2024.nc"
    # METEO_PATH=fr"data/processed/inmet/aggregated.parquet"
    # METEO_ORIGIN=ERA5
    # CONFIG_PATH="config\config.yaml"
    # OUTPUT_PATH=fr"data\datasets\2268922.pickle"
    # ID_UNIDADE="2268922"
    # build_dataset(id_unidade=ID_UNIDADE,
    #               sinan_path=SINAN_PATH,
    #               cnes_path=CNES_PATH,
    #               meteo_origin=METEO_ORIGIN,
    #               dataset_path=METEO_PATH,
    #               output_path=OUTPUT_PATH,
    #               config_path=CONFIG_PATH)