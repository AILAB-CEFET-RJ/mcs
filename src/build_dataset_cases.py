import pandas as pd
import numpy as np
import logging
import argparse
import pickle
import yaml
from sklearn.preprocessing import StandardScaler

FULL = "FULL"

def apply_sliding_window(X, y, window_size):
    """Generate sliding window sequences for time-series data using NumPy arrays."""
    X_windows = np.lib.stride_tricks.sliding_window_view(X, (window_size, X.shape[1]))
    y_aligned = y[window_size - 1:]  
    return X_windows.squeeze(), y_aligned

def create_new_features(df: pd.DataFrame, subset: str = ""):
    windows = [7, 14, 21, 28]
    for window in windows:
        df[f'CASES_MM_{window}'] = df['CASES'].rolling(window=window).mean()
        df[f'CASES_ACC_{window}'] = df['CASES'].rolling(window=window).sum()
    
    for lag in range(1, 7):
        df[f'CASES_LAG_{lag}'] = df['CASES'].shift(lag)

    # Removendo colunas em branco
    df = df.drop(columns=['DT_NOTIFIC', 'LAT', 'LNG', 'ID_UNIDADE'])
    df.dropna(inplace=True)

    X = df.drop(columns=['CASES']).to_numpy()
    y = df['CASES'].to_numpy()

    feature_names = df.drop(columns=['CASES']).columns.tolist()
    feature_dict = pd.DataFrame({"Index": range(len(feature_names)),"Feature": feature_names})
    feature_dict.to_csv("feature_dictionary.csv", index=False)

    logging.info(f"{subset} — Tamanho total: {len(y)}, Zeros: {(y == 0).sum()}, Não-Zeros: {(y > 0).sum()}")

    return X, y

def build_dataset(id_unidade, sinan_path, cnes_path, output_path, config_path, isweekly):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    window_size = config['preproc']['SLIDING_WINDOW_SIZE']
    train_split = config['preproc']['TRAIN_SPLIT']
    val_split = config['preproc']['VAL_SPLIT']

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
    parser.add_argument("output_path", help="Output path for processed dataset")
    parser.add_argument("config_path", help="Path to configuration YAML file")
    parser.add_argument("isweekly", help="Aggregate weekly values")
    args = parser.parse_args()

    isweekly = args.isweekly.lower() == 'true'

    build_dataset(
        id_unidade=args.id_unidade,
        sinan_path=args.sinan_path,
        cnes_path=args.cnes_path,
        output_path=args.output_path,
        config_path=args.config_path,
        isweekly=isweekly
    )

if __name__ == "__main__":
    main()