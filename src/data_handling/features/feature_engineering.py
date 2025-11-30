# src/features/feature_engineering.py

import os
import logging
import pandas as pd
import numpy as np


def create_new_features(df: pd.DataFrame, subset: str, config, output_path):
    """
    Gera features, constrói janelas deslizantes e prepara:
      - X: matriz de preditores
      - y: alvo (CASES no dia/semana seguinte -> h = 1)
      - sidecar: dicionário com metadados alinhados a y
        * DATE: data associada ao alvo (TARGET_DATE)
        * ID_UNIDADE: unidade de notificação
        * Y_TRUE: valor real do alvo (TARGET)
    """
    df = df.copy()
    enabled = config.features["enable"]
    windows = config.features.get("windows", [7, 14, 21, 28])
    lags = config.features.get("lags", 6)
    min_date = pd.to_datetime(config.min_date)
    max_date = pd.to_datetime(config.max_date)
    print(min_date)
    print(max_date)

    def is_available(colname: str) -> bool:
        return colname in df.columns

    # -------------------------------------------------------------------------
    # 1) FEATURES DE CASOS (baseadas em CASES_t)
    # -------------------------------------------------------------------------
    if enabled.get("cases_windows", False) and is_available("CASES"):
        for window in windows:
            df[f"CASES_MM_{window}"] = df["CASES"].rolling(window=window).mean()

    if enabled.get("cases_accumulators", False) and is_available("CASES"):
        for window in windows:
            df[f"CASES_ACC_{window}"] = df["CASES"].rolling(window=window).sum()

    if enabled.get("cases_lags", False) and is_available("CASES"):
        for lag in range(1, lags + 1):
            df[f"CASES_LAG_{lag}"] = df["CASES"].shift(lag)

    # -------------------------------------------------------------------------
    # 2) FEATURES METEOROLÓGICAS SIMPLES
    # -------------------------------------------------------------------------
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
        # semana epidemiológica baseada em DT_NOTIFIC
        df["WEEK_OF_YEAR"] = df["DT_NOTIFIC"].dt.isocalendar().week

    # -------------------------------------------------------------------------
    # 3) FEATURES METEOROLÓGICAS COM JANELAS (ROLLING)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 4) CORTE TEMPORAL
    # -------------------------------------------------------------------------
    if subset == "train":
        df = df[df["DT_NOTIFIC"] >= min_date]

    # -------------------------------------------------------------------------
    # 5) DEFINIÇÃO DO ALVO PARA h = 1
    #    TARGET_t = CASES_{t+1}
    #    TARGET_DATE_t = data de notificação do dia/semana seguinte
    # -------------------------------------------------------------------------
    if "ID_UNIDADE" in df.columns:
        df["TARGET"] = df.groupby("ID_UNIDADE")["CASES"].shift(-1)
        df["TARGET_DATE"] = df.groupby("ID_UNIDADE")["DT_NOTIFIC"].shift(-1)
    else:
        df["TARGET"] = df["CASES"].shift(-1)
        df["TARGET_DATE"] = df["DT_NOTIFIC"].shift(-1)

    # -------------------------------------------------------------------------
    # 6) PREPARO DO SIDECAR (metadados alinhados ao y)
    # -------------------------------------------------------------------------
    sidecar_cols = []
    if "TARGET_DATE" in df.columns:
        sidecar_cols.append("TARGET_DATE")
    if "ID_UNIDADE" in df.columns:
        sidecar_cols.append("ID_UNIDADE")

    sidecar_df = df[sidecar_cols].copy()

    # -------------------------------------------------------------------------
    # 7) DEFINIÇÃO DAS COLUNAS DE FEATURES
    #    Excluímos: alvo, casos brutos, coordenadas e colunas de sidecar
    # -------------------------------------------------------------------------
    cols_to_exclude = set(["CASES", "TARGET", "LAT", "LNG", "DT_NOTIFIC"] + sidecar_cols)
    feat_cols = [c for c in df.columns if c not in cols_to_exclude]

    # -------------------------------------------------------------------------
    # 8) MÁSCARA DE LINHAS VÁLIDAS (sem NaN em TARGET nem em features)
    # -------------------------------------------------------------------------
    valid_mask = df[["TARGET"] + feat_cols].notna().all(axis=1)
    df_valid = df.loc[valid_mask].copy()

    # -------------------------------------------------------------------------
    # 9) SIDECAR ALINHADO 1-a-1 COM y
    # -------------------------------------------------------------------------
    sidecar = {}

    if "TARGET_DATE" in sidecar_df.columns:
        sidecar["DATE"] = pd.to_datetime(
            sidecar_df.loc[valid_mask, "TARGET_DATE"]
        ).to_numpy()

    if "ID_UNIDADE" in sidecar_df.columns:
        sidecar["ID_UNIDADE"] = sidecar_df.loc[valid_mask, "ID_UNIDADE"].to_numpy()

    # valor real do alvo (y_true) para facilitar comparação posterior
    sidecar["Y_TRUE"] = df_valid["TARGET"].to_numpy()

    # -------------------------------------------------------------------------
    # 10) EXTRAÇÃO FINAL: X (features) e y (TARGET)
    # -------------------------------------------------------------------------
    X = df_valid[feat_cols].to_numpy()
    y = df_valid["TARGET"].to_numpy()

    # -------------------------------------------------------------------------
    # 11) EXPORTA DICIONÁRIO DE FEATURES (index -> nome)
    # -------------------------------------------------------------------------
    feature_names = feat_cols
    print(output_path)
    os.makedirs(os.path.dirname(output_path + "/"), exist_ok=True)
    pd.DataFrame({"Index": range(len(feature_names)), "Feature": feature_names}).to_csv(
        f"{output_path}/feature_dictionary.csv", index=False
    )

    logging.info(
        f"{subset} - samples: {len(y)} | zeros: {(y == 0).sum()} | non-zeros: {(y > 0).sum()}"
    )
    return X, y, sidecar
