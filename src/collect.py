#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coleta resultados de métricas (linha Teste) em pastas do tipo:
RJ_DAILY_CASEONLY_75_rf

Campos codificados no nome da pasta:
- Local: RJ ou RN
- Resolução: DAILY ou WEEKLY
- Features: CASEONLY ou FULL
- Seed: número (ex: 75)
- Modelo: rf, poisson, zip, etc.

Saídas:
- detailed_results.csv: uma linha por pasta (seed)
- aggregated_results.csv: médias por (Local, Resolução, Features, Modelo)
"""

import os
import re
import argparse
import pandas as pd

# regex para pastas do tipo RJ_DAILY_CASEONLY_75_rf
DIR_PATTERN = re.compile(
    r'^(RJ|RN)_(DAILY|WEEKLY)_(CASEONLY|FULL)_(\d+)_([A-Za-z0-9]+)$'
)

def parse_dir_name(dirname: str):
    """
    Extrai local, resolução, features, seed e modelo do nome do diretório.
    Exemplo: RJ_DAILY_CASEONLY_75_rf
    """
    m = DIR_PATTERN.match(dirname)
    if not m:
        return None
    local, freq, features, seed, model = m.groups()
    return {
        "local": local,
        "frequencia": freq,
        "conjunto": features,
        "seed": int(seed),
        "modelo": model,
    }

def coletar_resultados(root_dir: str):
    registros = []

    for current_dir, subdirs, files in os.walk(root_dir):
        dirname = os.path.basename(current_dir)
        parsed = parse_dir_name(dirname)
        if parsed is None:
            continue

        if "metrics.csv" not in files:
            # pula diretórios sem metrics.csv
            continue

        metrics_path = os.path.join(current_dir, "metrics.csv")
        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"[AVISO] Não foi possível ler {metrics_path}: {e}")
            continue

        # Normaliza a coluna 'Conjunto' e filtra linha 'Teste'
        if "Conjunto" not in df.columns:
            print(f"[AVISO] Arquivo {metrics_path} não tem coluna 'Conjunto'.")
            continue

        df["Conjunto_normalizado"] = df["Conjunto"].astype(str).str.strip().str.lower()
        df_teste = df[df["Conjunto_normalizado"] == "teste"]

        if df_teste.empty:
            print(f"[AVISO] Não encontrei linha 'Teste' em {metrics_path}.")
            continue

        # assume apenas uma linha 'Teste'
        row = df_teste.iloc[0]

        registro = {
            "pasta": dirname,
            "local": parsed["local"],
            "frequencia": parsed["frequencia"],
            "conjunto": parsed["conjunto"],
            "seed": parsed["seed"],
            "modelo": parsed["modelo"],
        }

        # copia as métricas principais, se existirem
        for col in [
            "MSE", "RMSE", "MAE", "R2",
            "MAPE (ign. zeros)", "SMAPE",
            "Poisson_Deviance", "Pearson"
        ]:
            if col in row.index:
                registro[col] = row[col]
            else:
                registro[col] = None

        registros.append(registro)

    if not registros:
        print("Nenhum resultado encontrado. Verifique o diretório raiz e o padrão das pastas.")
        return None, None

    df_detalhado = pd.DataFrame(registros)

    # agrega por Local, Frequência, Conjunto e Modelo
    group_cols = ["local", "frequencia", "conjunto", "modelo"]
    metric_cols = [
        "MSE", "RMSE", "MAE", "R2",
        "MAPE (ign. zeros)", "SMAPE",
        "Poisson_Deviance", "Pearson"
    ]

    df_agregado = (
        df_detalhado
        .groupby(group_cols)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )

    return df_detalhado, df_agregado


def main():
    parser = argparse.ArgumentParser(
        description="Coleta métricas (linha Teste) de pastas com modelos otimizados."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Diretório raiz onde estão as pastas do tipo RJ_DAILY_CASEONLY_75_rf"
    )
    parser.add_argument(
        "--out-detailed",
        type=str,
        default="detailed_results.csv",
        help="Caminho de saída para o CSV com resultados por seed."
    )
    parser.add_argument(
        "--out-agg",
        type=str,
        default="aggregated_results.csv",
        help="Caminho de saída para o CSV com médias por local/frequência/conjunto/modelo."
    )

    args = parser.parse_args()

    df_detalhado, df_agregado = coletar_resultados(args.root)

    if df_detalhado is None:
        return

    df_detalhado.to_csv(args.out_detailed, index=False)
    df_agregado.to_csv(args.out_agg, index=False)

    print(f"✅ Resultados detalhados salvos em: {args.out_detailed}")
    print(f"✅ Resultados agregados salvos em: {args.out_agg}")


if __name__ == "__main__":
    main()
