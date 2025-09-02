import os
import pandas as pd
import numpy as np
import logging

def create_new_features(df: pd.DataFrame, subset: str, config, output_path):
    df = df.copy()
    enabled = config.features["enable"] 
    windows = config.features.get("windows", [7, 14, 21, 28])
    lags = config.features.get("lags", 6)
    min_date = pd.to_datetime(config.min_date)
    max_date = pd.to_datetime(config.max_date)
    print(min_date)
    print(max_date)

    def is_available(colname):
        return colname in df.columns

    # --- Casos
    if enabled.get("cases_windows", False):
        for window in windows:
            df[f"CASES_MM_{window}"] = df["CASES"].rolling(window=window).mean()

    if enabled.get("cases_accumulators", False):
        for window in windows:
            df[f"CASES_ACC_{window}"] = df["CASES"].rolling(window=window).sum()

    if enabled.get("cases_lags", False):
        for lag in range(1, lags + 1):
            df[f"CASES_LAG_{lag}"] = df["CASES"].shift(lag)

    # --- Meteorologia simples
    if enabled.get("ideal_temp", False) and is_available("TEM_AVG"):
        low, high = config.features["temp"]["ideal_temp_range"]
        df["IDEAL_TEMP"] = df["TEM_AVG"].apply(lambda x: 1 if low <= x <= high else 0)

    if enabled.get("extreme_temp", False) and is_available("TEM_AVG"):
        low, high = config.features["temp"]["extreme_temp_range"]
        df["EXTREME_TEMP"] = df["TEM_AVG"].apply(lambda x: 1 if x <= low or x >= high else 0)

    if enabled.get("significant_rain", False) and is_available("RAIN"):
        low, high = config.features["rain"]["significant_range"]
        df["SIGNIFICANT_RAIN"] = df["RAIN"].apply(lambda x: 1 if low <= x < high else 0)

    if enabled.get("extreme_rain", False) and is_available("RAIN"):
        threshold = config.features["rain"]["extreme_threshold"]
        df["EXTREME_RAIN"] = df["RAIN"].apply(lambda x: 1 if x >= threshold else 0)

    if enabled.get("temp_range", False) and is_available("TEM_MAX") and is_available("TEM_MIN"):
        df["TEMP_RANGE"] = df["TEM_MAX"] - df["TEM_MIN"]

    if enabled.get("week_of_year", False):
        df["WEEK_OF_YEAR"] = df["DT_NOTIFIC"].dt.isocalendar().week

    # --- Rolling features meteorológicas
    if enabled.get("windows", False):
        for window in windows:
            if is_available("TEM_AVG"):
                df[f"TEM_AVG_MM_{window}"] = df["TEM_AVG"].rolling(window=window).mean()
            if is_available("RAIN"):
                df[f"RAIN_ACC_{window}"] = df["RAIN"].rolling(window=window).sum()
                df[f"RAIN_MM_{window}"] = df["RAIN"].rolling(window=window).mean()
            if is_available("RH_AVG"):
                df[f"RH_MM_{window}"] = df["RH_AVG"].rolling(window=window).mean()
            if is_available("TEMP_RANGE"):
                df[f"TEMP_RANGE_MM_{window}"] = df["TEMP_RANGE"].rolling(window=window).mean()

    # --- Limpeza final
    if subset == "train":
        df = df[df['DT_NOTIFIC'] >= min_date]

    # 1) preparar colunas do sidecar ANTES de descartar nada
    date_col = "DATE" if "DATE" in df.columns else "DT_NOTIFIC"
    sidecar_cols = [c for c in [date_col, "ID_UNIDADE"] if c in df.columns]
    sidecar_df = df[sidecar_cols].copy()

    # 2) definir colunas de features (excluir alvo e sidecar/coords)
    cols_to_exclude = set(["CASES", "LAT", "LNG"] + sidecar_cols)
    feat_cols = [c for c in df.columns if c not in cols_to_exclude]

    # 3) máscara de linhas válidas (sem NaN em y nem em features)
    valid_mask = df[["CASES"] + feat_cols].notna().all(axis=1)
    df_valid = df.loc[valid_mask].copy()

    # 4) montar sidecar ALINHADO 1-a-1 com y
    sidecar = {
        "DATE": pd.to_datetime(sidecar_df.loc[valid_mask, date_col]).to_numpy() if date_col in sidecar_df else None,
        "ID_UNIDADE": sidecar_df.loc[valid_mask, "ID_UNIDADE"].to_numpy() if "ID_UNIDADE" in sidecar_df else None,
    }
    df.drop(columns=[c for c in ["DT_NOTIFIC", "LAT", "LNG", "ID_UNIDADE"] if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    # --- Extração final
    X = df.drop(columns=["CASES"]).to_numpy()
    y = df["CASES"].to_numpy()

    # --- Exporta feature dictionary
    feature_names = df.drop(columns=["CASES"]).columns.tolist()
    print(output_path)
    os.makedirs(os.path.dirname(output_path+"/"), exist_ok=True)
    pd.DataFrame({"Index": range(len(feature_names)), "Feature": feature_names}).to_csv(f"{output_path}/feature_dictionary.csv", index=False)

    logging.info(f"{subset} - samples: {len(y)} | zeros: {(y==0).sum()} | non-zeros: {(y>0).sum()}")
    return X, y, sidecar
