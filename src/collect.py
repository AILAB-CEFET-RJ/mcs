#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coletar_metrics_otimizados.py

Varre a pasta 'models/', encontra subpastas no formato:
    RJ_DAILY_CASEONLY_75_rf
    RJ_WEEKLY_FULL_123_xgb_poisson
    RN_DAILY_CASESONLY_42_xgb_zip
etc.

Para cada pasta que contém um 'metrics.csv':
    - Lê o arquivo
    - Seleciona a linha onde Conjunto == 'Teste'
    - Extrai as métricas
    - Salva em um DataFrame com colunas:
        local, resolucao, features, seed, modelo, MSE, RMSE, ...

Ao final:
    - Salva um CSV com todas as seeds: metrics_test_by_seed.csv
    - Agrupa por (local, resolucao, features, modelo) e calcula a média
      das métricas numéricas, salvando em metrics_test_agg.csv
"""

import os
import pandas as pd
from typing import Tuple, Optional

MODELS_ROOT = "models"  # ajuste se necessário


def parse_model_dirname(dirname: str) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Espera algo como:
        RJ_DAILY_CASEONLY_75_rf
        RN_WEEKLY_FULL_123_xgb_poisson

    Retorna:
        (local, resolucao, features, seed, modelo)

    Se o nome não bater com o padrão mínimo, retorna None.
    """
    parts = dirname.split("_")
    if len(parts) < 5:
        return None

    local = parts[0]
    resolucao = parts[1]
    features = parts[2]
    seed = parts[3]
    modelo = "_".join(parts[4:])  # rf, xgb_poisson, xgb_zip, etc.

    return local, resolucao, features, seed, modelo


def main():
    registros = []

    for dirname in sorted(os.listdir(MODELS_ROOT)):
        full_dir = os.path.join(MODELS_ROOT, dirname)
        if not os.path.isdir(full_dir):
            continue

        parsed = parse_model_dirname(dirname)
        if parsed is None:
            # pula pastas que não seguem o padrão
            continue

        local, resolucao, features, seed, modelo = parsed

        metrics_path = os.path.join(full_dir, "metrics.csv")
        if not os.path.isfile(metrics_path):
            continue

        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"[AVISO] Erro ao ler {metrics_path}: {e}")
            continue

        if "Conjunto" not in df.columns:
            print(f"[AVISO] Arquivo {metrics_path} não possui coluna 'Conjunto'. Ignorando.")
            continue

        df_teste = df[df["Conjunto"] == "Teste"]
        if df_teste.empty:
            print(f"[AVISO] Nenhuma linha 'Teste' em {metrics_path}. Ignorando.")
            continue

        # assume que há apenas uma linha de teste
        row = df_teste.iloc[0]

        registro = {
            "local": local,
            "resolucao": resolucao,
            "features": features,
            "seed": seed,
            "modelo": modelo,
        }

        # adiciona todas as colunas numéricas de métricas
        for col in df.columns:
            if col == "Conjunto":
                continue
            registro[col] = row[col]

        registros.append(registro)

    if not registros:
        print("Nenhuma métrica encontrada. Verifique o caminho de 'models/'.")
        return

    df_all = pd.DataFrame(registros)

    # Salva métricas por seed
    out_by_seed = "metrics_test_by_seed.csv"
    df_all.to_csv(out_by_seed, index=False)
    print(f"✅ Métricas por seed salvas em: {out_by_seed}")

    # Agrupa por (local, resolucao, features, modelo) e tira média das métricas numéricas
    group_cols = ["local", "resolucao", "features", "modelo"]
    metric_cols = [c for c in df_all.columns if c not in group_cols + ["seed"]]

    df_agg = (
        df_all
        .groupby(group_cols, as_index=False)[metric_cols]
        .mean()
    )

    out_agg = "metrics_test_agg.csv"
    df_agg.to_csv(out_agg, index=False)
    print(f"✅ Métricas agregadas (média entre seeds) salvas em: {out_agg}")


if __name__ == "__main__":
    main()
