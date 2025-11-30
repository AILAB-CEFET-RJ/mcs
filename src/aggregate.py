#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ensemble_with_sidecars_comparative.py

Fluxo:

1) Para cada DATASET (ex.: RJ_DAILY_CASEONLY) e cada MODEL (rf, xgb_poisson, xgb_zip):
   - Procura pastas models/<DATASET>_<SEED>_<MODEL>/predictions.csv
   - Faz ensemble entre seeds -> y_pred_mean, y_pred_std
   - Liga com sidecar data/datasets/<DATASET>/dataset_ids.pickle
   - Agrega por DATE (somando unidades) e propaga incerteza
   - Gera gráficos individuais por modelo:
       * série temporal (real vs média ± 1 desvio-padrão)
       * resíduos no tempo
       * histograma de resíduos
       * scatter real vs predito (média)

2) Para cada DATASET, usando os 3 modelos já agregados:
   - Gera gráficos comparativos sobrepostos:
       * série temporal comparativa (real + 3 modelos)
       * scatter comparativo (3 nuvens de pontos)

Estrutura esperada:

- data/datasets/<DATASET_NAME>/dataset_ids.pickle
- models/<DATASET_NAME>_<SEED>_<MODEL>/predictions.csv

Exemplos de pastas de modelo:
    RJ_DAILY_CASEONLY_75_rf
    RJ_DAILY_CASEONLY_75_xgb_poisson
    RJ_DAILY_CASEONLY_75_xgb_zip
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SIDECAR_NAME = "dataset_ids.pickle"


# ============================================================
# Utils básicos
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Parse de nomes de pastas e procura de models/datasets
# ============================================================

def parse_model_folder_name(folder_name: str):
    """
    Espera algo como:
      RJ_DAILY_CASEONLY_75_rf
      RJ_DAILY_CASEONLY_75_xgb_poisson

    Retorna:
      dataset_name = 'RJ_DAILY_CASEONLY'
      seed         = '75'
      model_type   = 'rf' / 'xgb_poisson' / 'xgb_zip'
    """
    parts = folder_name.split("_")

    if len(parts) < 4:
        raise ValueError(f"Nome de pasta inesperado: {folder_name}")

    dataset_name = "_".join(parts[0:3])  # RJ_DAILY_CASEONLY
    seed = parts[3]

    model_type = parts[4]
    if len(parts) >= 6:
        model_type = parts[4] + "_" + parts[5]

    return dataset_name, seed, model_type


def find_model_folders(models_dir: str, dataset_name: str, model_type: str):
    """
    Encontra pastas do tipo:
      models/<DATASET_NAME>_*_<model_type>
    Ex: models/RJ_DAILY_CASEONLY_*_rf
    """
    pattern = os.path.join(models_dir, f"{dataset_name}_*_{model_type}")
    folders = sorted(glob.glob(pattern))
    return folders


def find_dataset_dir(datasets_root: str, dataset_name: str):
    """
    Procura data/datasets/<dataset_name>.
    Se não encontrar exato, tenta substring match.
    """
    direct = os.path.join(datasets_root, dataset_name)
    if os.path.isdir(direct):
        return direct

    # fallback: substring
    candidates = [
        d for d in os.listdir(datasets_root)
        if os.path.isdir(os.path.join(datasets_root, d))
    ]
    scored = []
    for d in candidates:
        score = 0
        if dataset_name and dataset_name in d:
            score += 2
        if dataset_name and d in dataset_name:
            score += 1
        scored.append((score, d))
    scored.sort(reverse=True)

    if scored and scored[0][0] > 0:
        return os.path.join(datasets_root, scored[0][1])

    return None


# ============================================================
# Sidecars: dataset_ids.pickle
# ============================================================

def choose_split_by_len(ids_payload: dict, n_rows: int):
    """
    Determina automaticamente o split cujo comprimento de DATE bate com n_rows.
    Se houver várias possibilidades, prefere 'test'.
    """
    matches = []
    for split in ["test", "val", "train"]:
        sid = ids_payload.get(split, {})
        arr = sid.get("DATE", None)
        if arr is not None and len(arr) == n_rows:
            matches.append(split)

    if len(matches) == 1:
        return matches[0]
    if "test" in ids_payload and ids_payload["test"].get("DATE", None) is not None:
        return "test"
    return matches[0] if matches else None


def attach_sidecar(df: pd.DataFrame, dataset_dir: str, split: str = None):
    """
    Anexa DATE, ID_UNIDADE e t_index usando dataset_ids.pickle, por posição.
    Retorna (df_enriquecido, split_usado, sidecar_path).
    """
    sidecar_path = os.path.join(dataset_dir, SIDECAR_NAME)
    if not os.path.isfile(sidecar_path):
        print(f"[SIDEcar] Não encontrei {SIDECAR_NAME} em {dataset_dir}")
        return df, None, None

    with open(sidecar_path, "rb") as f:
        ids_payload = pickle.load(f)

    if split is None or split == "auto":
        split = choose_split_by_len(ids_payload, len(df))

    if split not in ids_payload:
        print(f"[SIDEcar] split '{split}' ausente em {sidecar_path}")
        return df, split, sidecar_path

    sid = ids_payload[split]

    m = min(len(df), len(sid.get("DATE", [])))
    if m == 0:
        print(f"[SIDEcar] sidecar para '{split}' vazio ou incompatível.")
        return df, split, sidecar_path

    df = df.iloc[:m].copy()

    if "DATE" in sid and sid["DATE"] is not None:
        df["DATE"] = pd.to_datetime(sid["DATE"][:m])
    if "ID_UNIDADE" in sid and sid["ID_UNIDADE"] is not None:
        df["ID_UNIDADE"] = sid["ID_UNIDADE"][:m]
    if "t_index" in sid and sid["t_index"] is not None:
        df["t_index"] = sid["t_index"][:m]

    print(f"[SIDEcar] OK: {sidecar_path} | split={split} | linhas={m}")
    return df, split, sidecar_path


# ============================================================
# Carregar predictions e fazer ensemble entre seeds
# ============================================================

def load_predictions(pred_path: str, seed: str) -> pd.DataFrame:
    """
    Carrega um predictions.csv e renomeia y_pred -> y_pred_seed_<seed>.
    Espera colunas: t_index, y_true, y_pred (pelo menos).
    """
    df = pd.read_csv(pred_path)
    if "y_pred" not in df.columns:
        raise ValueError(f"Arquivo {pred_path} não contém coluna 'y_pred'.")

    df = df.rename(columns={"y_pred": f"y_pred_seed_{seed}"})
    return df


def aggregate_predictions_across_seeds(models_dir: str, dataset_name: str, model_type: str):
    """
    Lê todos os predictions.csv de todas as seeds para um modelo específico
    e produz um DataFrame com:

        t_index, y_true,
        y_pred_seed_<seed1>, y_pred_seed_<seed2>, ...,
        y_pred_mean, y_pred_std
    """
    model_folders = find_model_folders(models_dir, dataset_name, model_type)

    if not model_folders:
        raise RuntimeError(
            f"Nenhuma pasta de modelo encontrada para {dataset_name}_*_{model_type} em {models_dir}"
        )

    merged = None

    for folder in model_folders:
        folder_name = os.path.basename(folder)
        ds_name, seed, mt = parse_model_folder_name(folder_name)

        if ds_name != dataset_name or mt != model_type:
            continue  # segurança

        pred_path = os.path.join(folder, "predictions.csv")
        if not os.path.exists(pred_path):
            print(f"[AVISO] Sem predictions.csv em {folder}")
            continue

        df_pred = load_predictions(pred_path, seed)

        if merged is None:
            merged = df_pred.copy()
        else:
            # merge por t_index e y_true (garantindo alinhamento)
            merged = merged.merge(
                df_pred[["t_index", "y_true", f"y_pred_seed_{seed}"]],
                on=["t_index", "y_true"],
                how="inner"
            )

    if merged is None:
        raise RuntimeError("Nenhum predictions.csv válido encontrado para o ensemble.")

    # colunas de predição das seeds
    pred_cols = [c for c in merged.columns if c.startswith("y_pred_seed_")]
    merged["y_pred_mean"] = merged[pred_cols].mean(axis=1)
    merged["y_pred_std"] = merged[pred_cols].std(axis=1)

    return merged


# ============================================================
# Agregar por DATE (somando unidades) e propagar incerteza
# ============================================================

def aggregate_by_date(df: pd.DataFrame, resample_rule: str = None):
    """
    Recebe DataFrame com colunas:
        DATE, y_true, y_pred_mean, y_pred_std
        (e possivelmente ID_UNIDADE, t_index, etc.)

    Agrega por DATE somando unidades:

        y_true_agg(date)       = sum(y_true)
        y_pred_mean_agg(date)  = sum(y_pred_mean)
        var_agg(date)          = sum( (y_pred_std)^2 )
        y_pred_std_agg(date)   = sqrt(var_agg)

    Se resample_rule não for None (ex. 'W'), faz resample no tempo
    somando períodos e propagando variância.
    """
    if "DATE" not in df.columns:
        raise ValueError("DataFrame não possui coluna 'DATE' para agregar por data.")

    tmp = df.copy()
    tmp = tmp.dropna(subset=["y_true", "y_pred_mean"])

    # variância = std^2
    tmp["var_pred"] = (tmp["y_pred_std"] ** 2).fillna(0.0)

    # agrega por DATE (soma unidades naturalmente)
    grouped = tmp.groupby("DATE").agg({
        "y_true": "sum",
        "y_pred_mean": "sum",
        "var_pred": "sum",
    }).sort_index()

    grouped["y_pred_std"] = np.sqrt(grouped["var_pred"])
    grouped = grouped.drop(columns=["var_pred"])

    # resample opcional (ex.: semanal)
    if resample_rule:
        grouped = grouped.resample(resample_rule).agg({
            "y_true": "sum",
            "y_pred_mean": "sum",
            "y_pred_std": lambda x: np.sqrt((x ** 2).sum())
        }).dropna(how="all")

    grouped = grouped.reset_index()  # DATE vira coluna

    return grouped  # colunas: DATE, y_true, y_pred_mean, y_pred_std


# ============================================================
# Funções de plot individuais (por modelo)
# ============================================================

def plot_ensemble_time(df: pd.DataFrame, model_name: str, outdir: str, split_name: str = "Teste"):
    """
    Série temporal: y_true vs y_pred_mean, com faixa ± 1 desvio-padrão.
    df deve ter colunas: DATE, y_true, y_pred_mean, y_pred_std.
    """
    ensure_dir(outdir)

    x = df["DATE"]
    y_true = df["y_true"].values
    y_mean = df["y_pred_mean"].values
    y_std = df["y_pred_std"].values

    plt.figure(figsize=(12, 4))
    plt.plot(x, y_true, label="Real", linewidth=1.5, alpha=0.9)
    plt.plot(x, y_mean, label="Predito (média entre seeds)", linewidth=1.5)

    upper = y_mean + y_std
    lower = y_mean - y_std
    plt.fill_between(x, lower, upper, alpha=0.2, label="± 1 desvio-padrão")

    plt.title(f"Série real vs ensemble - {model_name} ({split_name})")
    plt.xlabel("Data")
    plt.ylabel("Casos (agregado por data)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"{model_name.lower()}_ensemble_real_vs_pred.png".replace(" ", "_")
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=220)
    plt.close()
    print(f"📈 Série temporal (ensemble) salva em: {fpath}")


def plot_residuals_time(df: pd.DataFrame, model_name: str, outdir: str, split_name: str = "Teste"):
    """
    Resíduos ao longo do tempo: y_true - y_pred_mean (agregados por data).
    """
    ensure_dir(outdir)

    x = df["DATE"]
    residuals = df["y_true"].values - df["y_pred_mean"].values

    plt.figure(figsize=(12, 4))
    plt.plot(x, residuals, label="Resíduo", linewidth=1.2)
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    plt.title(f"Resíduos ao longo do tempo - {model_name} ({split_name})")
    plt.xlabel("Data")
    plt.ylabel("Resíduo (real - predito)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"{model_name.lower()}_residuos_tempo.png".replace(" ", "_")
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=220)
    plt.close()
    print(f"📉 Resíduos (tempo) salvos em: {fpath}")


def plot_residual_hist(df: pd.DataFrame, model_name: str, outdir: str, bins: int = 30, split_name: str = "Teste"):
    """
    Histograma dos resíduos (real - y_pred_mean), já agregados por data.
    """
    ensure_dir(outdir)
    residuals = df["y_true"].values - df["y_pred_mean"].values

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=bins, alpha=0.8)
    plt.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    plt.title(f"Histograma dos resíduos - {model_name} ({split_name})")
    plt.xlabel("Resíduo (real - predito)")
    plt.ylabel("Frequência")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"{model_name.lower()}_residuos_hist.png".replace(" ", "_")
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=220)
    plt.close()
    print(f"📊 Histograma de resíduos salvo em: {fpath}")


def plot_scatter_real_vs_pred(df: pd.DataFrame, model_name: str, outdir: str, split_name: str = "Teste"):
    """
    Scatter plot: y_true vs y_pred_mean (agregados por data), com linha y = x.
    """
    ensure_dir(outdir)
    y_true = df["y_true"].values
    y_mean = df["y_pred_mean"].values

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_mean, alpha=0.4, s=12)

    min_val = min(np.min(y_true), np.min(y_mean))
    max_val = max(np.max(y_true), np.max(y_mean))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    plt.title(f"Real vs Predito (média) - {model_name} ({split_name})")
    plt.xlabel("Real (agregado por data)")
    plt.ylabel("Predito (média, agregado por data)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"{model_name.lower()}_scatter_real_vs_pred.png".replace(" ", "_")
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=220)
    plt.close()
    print(f"📌 Scatter real vs predito salvo em: {fpath}")


# ============================================================
# Funções de plot comparativos (entre modelos)
# ============================================================

def plot_comparative_prediction_time(df_dict, dataset_name, outdir, split_name="Teste"):
    """
    Série temporal comparativa com modelos sobrepostos.

    df_dict: dict com:
        {
            "rf": df_rf,
            "xgb_poisson": df_xp,
            "xgb_zip": df_xz
        }

    Cada df deve ter: DATE, y_true, y_pred_mean
    """
    ensure_dir(outdir)

    plt.figure(figsize=(12, 4))

    # y_real da primeira entrada
    first_key = list(df_dict.keys())[0]
    x = df_dict[first_key]["DATE"]
    y_true = df_dict[first_key]["y_true"].values

    # Série real
    plt.plot(x, y_true, label="Real", linewidth=1.8, color="black")

    COLORS = {
        "rf": "tab:blue",
        "xgb_poisson": "tab:orange",
        "xgb_zip": "tab:green",
    }

    LABELS = {
        "rf": "Random Forest",
        "xgb_poisson": "XGBoost Poisson",
        "xgb_zip": "XGBoost ZIP",
    }

    for model_key, df in df_dict.items():
        y_pred = df["y_pred_mean"].values
        plt.plot(
            x,
            y_pred,
            label=LABELS.get(model_key, model_key),
            linewidth=1.5,
            alpha=0.9,
            color=COLORS.get(model_key, None),
        )

    plt.title(f"Série temporal comparativa - {dataset_name} ({split_name})")
    plt.xlabel("Data")
    plt.ylabel("Casos (agregado por data)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fname = f"{dataset_name}_comparative_pred_vs_real.png".lower()
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=220)
    plt.close()

    print(f"📈 Gráfico comparativo 'pred vs real' salvo em: {fpath}")


def plot_comparative_scatter(df_dict, dataset_name, outdir, split_name="Teste"):
    """
    Scatter comparativo: real vs predito para vários modelos no mesmo gráfico.
    """
    ensure_dir(outdir)

    plt.figure(figsize=(6, 6))

    COLORS = {
        "rf": "tab:blue",
        "xgb_poisson": "tab:orange",
        "xgb_zip": "tab:green",
    }

    LABELS = {
        "rf": "Random Forest",
        "xgb_poisson": "XGBoost Poisson",
        "xgb_zip": "XGBoost ZIP",
    }

    # y_real do primeiro modelo
    first_key = list(df_dict.keys())[0]
    y_true = df_dict[first_key]["y_true"].values

    # faixa para linha y=x
    mins = [df["y_pred_mean"].min() for df in df_dict.values()] + [y_true.min()]
    maxs = [df["y_pred_mean"].max() for df in df_dict.values()] + [y_true.max()]
    min_val = min(mins)
    max_val = max(maxs)

    for model_key, df in df_dict.items():
        y_pred = df["y_pred_mean"].values
        plt.scatter(
            y_true,
            y_pred,
            s=12,
            alpha=0.5,
            color=COLORS.get(model_key, None),
            label=LABELS.get(model_key, model_key),
        )

    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    plt.title(f"Scatter comparativo Real vs Predito - {dataset_name} ({split_name})")
    plt.xlabel("Real (agregado por data)")
    plt.ylabel("Predito (média, agregado por data)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fname = f"{dataset_name}_comparative_scatter_real_vs_pred.png".lower()
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=220)
    plt.close()

    print(f"📌 Scatter comparativo salvo em: {fpath}")


# ============================================================
# Orquestração: processar um modelo e retornar df agregado
# ============================================================

def process_model(models_dir: str,
                  datasets_root: str,
                  dataset_name: str,
                  model_type: str,
                  resample_rule: str = None,
                  split_name: str = "Teste") -> pd.DataFrame:
    """
    Para um dado dataset + modelo (rf / xgb_poisson / xgb_zip):

    1) Faz ensemble entre seeds (aggregate_predictions_across_seeds)
    2) Liga com sidecar (attach_sidecar) para obter DATE
    3) Agrega por data (aggregate_by_date)
    4) Gera gráficos individuais
    5) Retorna df agregado por data
    """
    print(f"\n🚀 Processando ensemble para: {dataset_name} | modelo={model_type}")

    # 1) Ensemble entre seeds (sem sidecar ainda)
    df_ensemble = aggregate_predictions_across_seeds(
        models_dir=models_dir,
        dataset_name=dataset_name,
        model_type=model_type
    )

    # 2) Encontra diretório do dataset e anexa sidecar
    dataset_dir = find_dataset_dir(datasets_root, dataset_name)
    if not dataset_dir:
        raise RuntimeError(f"Não encontrei dataset_dir para {dataset_name} em {datasets_root}.")

    df_ensemble_sc, eff_split, sidecar_path = attach_sidecar(
        df_ensemble,
        dataset_dir,
        split="auto"
    )
    if eff_split is None:
        eff_split = split_name

    # 3) Agrega por data (somando unidades) e resample opcional
    df_date = aggregate_by_date(df_ensemble_sc, resample_rule=resample_rule)

    # 4) Gera gráficos individuais
    model_name = f"{dataset_name}_{model_type}"
    outdir = models_dir  # pode trocar se quiser outra pasta

    plot_ensemble_time(df_date, model_name, outdir, split_name=eff_split)
    plot_residuals_time(df_date, model_name, outdir, split_name=eff_split)
    plot_residual_hist(df_date, model_name, outdir, split_name=eff_split)
    plot_scatter_real_vs_pred(df_date, model_name, outdir, split_name=eff_split)

    return df_date


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Ajuste aqui conforme seus datasets e modelos
    MODELS_DIR = "models"
    DATASETS_ROOT = "data/datasets"

    # Seus 4 datasets principais
    DATASETS = [
        "RJ_DAILY_FULL",
        "RJ_DAILY_CASEONLY",
        "RJ_WEEKLY_FULL",
        "RJ_WEEKLY_CASESONLY",
    ]

    # Tipos de modelo
    MODELOS = ["rf", "xgb_poisson", "xgb_zip"]

    # Regra de resample temporal:
    # - Para datasets DIÁRIOS, pode manter None (não reamostrar)
    # - Para WEEKLY, também pode deixar None (já está semanal)
    # - Se quiser reamostrar diário -> semanal, use "W"
    RESAMPLE_RULES = {
        "RJ_DAILY_FULL": None,
        "RJ_DAILY_CASEONLY": None,
        "RJ_WEEKLY_FULL": None,
        "RJ_WEEKLY_CASESONLY": None,
    }

    for ds in DATASETS:
        resample = RESAMPLE_RULES.get(ds, None)

        # Guardar dfs agregados por modelo para depois fazer os comparativos
        df_models = {}

        for m in MODELOS:
            try:
                df_m = process_model(
                    models_dir=MODELS_DIR,
                    datasets_root=DATASETS_ROOT,
                    dataset_name=ds,
                    model_type=m,
                    resample_rule=resample,
                    split_name="Teste",  # só para título
                )
                df_models[m] = df_m
            except Exception as e:
                print(f"\n[ERRO] Falha ao processar {ds} / {m}: {e}")

        # Gráficos comparativos, só se tiver pelo menos 2 modelos válidos
        if len(df_models) >= 2:
            try:
                plot_comparative_prediction_time(df_models, ds, MODELS_DIR, split_name="Teste")
                plot_comparative_scatter(df_models, ds, MODELS_DIR, split_name="Teste")
            except Exception as e:
                print(f"[ERRO] Falha ao gerar gráficos comparativos para {ds}: {e}")

    print("\n✅ Finalizado: ensemble + sidecars + gráficos individuais e comparativos.")
