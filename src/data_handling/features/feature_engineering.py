import pandas as pd
import numpy as np
import logging

def create_new_features(df: pd.DataFrame, subset: str, config):
    enabled = config.features["enable"]
    windows = config.features.get("windows", [7, 14, 21, 28])
    lags = config.features.get("lags", 6)

    if enabled.get("cases_windows", False):
        for window in windows:
            df[f"CASES_MM_{window}"] = df["CASES"].rolling(window=window).mean()

    if enabled.get("cases_accumulators", False):
        for window in windows:
            df[f"CASES_ACC_{window}"] = df["CASES"].rolling(window=window).sum()

    if enabled.get("cases_lags", False):
        for lag in range(1, lags + 1):
            df[f"CASES_LAG_{lag}"] = df["CASES"].shift(lag)

    if enabled.get("ideal_temp", False):
        low, high = config.features["temp"]["ideal_temp_range"]
        df["IDEAL_TEMP"] = df["TEM_AVG"].apply(lambda x: 1 if low <= x <= high else 0)

    if enabled.get("extreme_temp", False):
        low, high = config.features["temp"]["extreme_temp_range"]
        df["EXTREME_TEMP"] = df["TEM_AVG"].apply(lambda x: 1 if x <= low or x >= high else 0)

    if enabled.get("significant_rain", False):
        low, high = config.features["rain"]["significant_range"]
        df["SIGNIFICANT_RAIN"] = df["RAIN"].apply(lambda x: 1 if low <= x < high else 0)

    if enabled.get("extreme_rain", False):
        threshold = config.features["rain"]["extreme_threshold"]
        df["EXTREME_RAIN"] = df["RAIN"].apply(lambda x: 1 if x >= threshold else 0)

    if enabled.get("temp_range", False):
        df["TEMP_RANGE"] = df["TEM_MAX"] - df["TEM_MIN"]

    if enabled.get("week_of_year", False):
        df["WEEK_OF_YEAR"] = df["DT_NOTIFIC"].dt.isocalendar().week

    if enabled.get("windows", False):
        for window in windows:
            df[f"TEM_AVG_MM_{window}"] = df["TEM_AVG"].rolling(window=window).mean()
            df[f"RAIN_ACC_{window}"] = df["RAIN"].rolling(window=window).sum()
            df[f"RAIN_MM_{window}"] = df["RAIN"].rolling(window=window).mean()
            df[f"RH_MM_{window}"] = df["RH_AVG"].rolling(window=window).mean()
            df[f"TEMP_RANGE_MM_{window}"] = df["TEMP_RANGE"].rolling(window=window).mean()

    if enabled.get("lags", False):
        for lag in range(1, lags + 1):
            df[f"CASES_LAG_{lag}"] = df["CASES"].shift(lag)

    df.drop(columns=[c for c in ["DT_NOTIFIC", "LAT", "LNG", "ID_UNIDADE"] if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=["CASES"]).to_numpy()
    y = df["CASES"].to_numpy()

    feature_names = df.drop(columns=["CASES"]).columns.tolist()
    dict_name = "feature_dictionary.csv"
    pd.DataFrame({"Index": range(len(feature_names)), "Feature": feature_names}).to_csv(dict_name, index=False)

    logging.info(f"{subset} - samples: {len(y)} | zeros: {(y==0).sum()} | non-zeros: {(y>0).sum()}")
    return X, y
