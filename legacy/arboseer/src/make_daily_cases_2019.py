#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_daily_cases_2019.py

Lê um arquivo Parquet do SINAN, soma todos os CASOS por dia em 2019, e gera:
- daily_cases_2019.csv  (DATE, CASES_SUM)
- daily_cases_2019.png  (linha com a série diária)

Opcional: sobrepor uma segunda série vinda de CSV (ex.: y_true agregado) para comparação.

Exemplos:
  python make_daily_cases_2019.py data/processed/sinan/DENG240810.parquet

  # Se a coluna de data tiver outro nome:
  python make_daily_cases_2019.py data/processed/sinan/DENG240810.parquet --date-col DT_NOTIFIC --cases-col CASES

  # Sobrepor uma série de comparação (ex.: predictions_enriched.csv com y_true):
  python make_daily_cases_2019.py data/processed/sinan/DENG240810.parquet \
    --compare-csv models/RN_WEEKLY_CASESONLY_987_rf/predictions_enriched.csv \
    --compare-date-col DATE --compare-value-col y_true
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def to_datetime_col(df, col):
    if col not in df.columns:
        raise KeyError(f"Coluna de data '{col}' não encontrada no dataframe.")
    dt = pd.to_datetime(df[col], errors="coerce", utc=False)
    if dt.isna().all():
        raise ValueError(f"Falha ao converter '{col}' para datetime (todos NaN).")
    return dt

def main():
    p = argparse.ArgumentParser(description="Soma casos diários de 2019 e gera gráfico/CSV.")
    p.add_argument("parquet_path", help="Caminho para o parquet do SINAN (ex.: data/processed/sinan/DENG240810.parquet)")
    p.add_argument("--date-col", default="DT_NOTIFIC", help="Nome da coluna de data (padrão: DT_NOTIFIC)")
    p.add_argument("--cases-col", default="CASES", help="Nome da coluna de casos (padrão: CASES)")
    p.add_argument("--outdir", default=None, help="Diretório de saída (padrão: pasta do parquet)")
    # Overlay opcional para comparação
    p.add_argument("--compare-csv", default=None, help="CSV com série para comparação (opcional)")
    p.add_argument("--compare-date-col", default="DATE", help="Coluna de data no CSV de comparação (padrão: DATE)")
    p.add_argument("--compare-value-col", default="y_true", help="Coluna de valores no CSV de comparação (padrão: y_true)")
    args = p.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.parquet_path))
    os.makedirs(outdir, exist_ok=True)

    # 1) Ler parquet original
    df = pd.read_parquet(args.parquet_path)
    if args.cases_col not in df.columns:
        raise KeyError(f"Coluna de casos '{args.cases_col}' não encontrada. Colunas disponíveis: {list(df.columns)}")

    # 2) Preparar data e filtrar 2019
    dt = to_datetime_col(df, args.date_col)
    df = df.assign(DATE=dt.dt.floor("D"))
    start = pd.Timestamp("2019-01-01")
    end   = pd.Timestamp("2019-12-31")
    mask = (df["DATE"] >= start) & (df["DATE"] <= end)
    df_2019 = df.loc[mask, ["DATE", args.cases_col]].copy()

    # 3) Agregar por dia (soma entre todas as unidades)
    daily = (
        df_2019.groupby("DATE", as_index=True)[args.cases_col]
        .sum()
        .sort_index()
    )

    # 4) Garantir todos os dias do período (preencher zeros onde não houver)
    idx = pd.date_range(start=start, end=end, freq="D")
    daily = daily.reindex(idx, fill_value=0)
    daily.name = "CASES_SUM"
    daily_df = daily.reset_index().rename(columns={"index": "DATE"})

    # 5) Salvar CSV com a série diária
    csv_out = os.path.join(outdir, "daily_cases_2019.csv")
    daily_df.to_csv(csv_out, index=False)
    print(f"[CSV] {csv_out}  (linhas={len(daily_df)})")

    # 6) Plotar
    plt.figure(figsize=(12, 5))
    plt.plot(daily.index, daily.values, linewidth=1.6, label="CASES_SUM (SINAN)")

    # 6b) Sobrepor série de comparação (opcional)
    if args.compare_csv:
        try:
            comp = pd.read_csv(args.compare_csv)
            comp_dt = pd.to_datetime(comp[args.compare_date_col], errors="coerce", utc=False)
            comp = comp.assign(DATE=comp_dt.dt.floor("D"))
            comp_2019 = comp[(comp["DATE"] >= start) & (comp["DATE"] <= end)]
            comp_daily = (
                comp_2019.groupby("DATE", as_index=True)[args.compare_value_col]
                .sum()
                .sort_index()
                .reindex(idx, fill_value=0)
            )
            plt.plot(comp_daily.index, comp_daily.values, linewidth=1.2, alpha=0.9,
                     label=f"COMPARE ({args.compare_value_col})")
            print(f"[COMPARE] Série de comparação sobreposta a partir de {args.compare_csv}")
        except Exception as e:
            print(f"[AVISO] Não foi possível sobrepor comparação ({e}). Prosseguindo apenas com a série base.")

    plt.title("SINAN - Casos Diários Somados (2019)")
    plt.xlabel("Data")
    plt.ylabel("Casos (soma diária)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_out = os.path.join(outdir, "daily_cases_2019.png")
    plt.savefig(png_out, dpi=220)
    plt.close()
    print(f"[PNG] {png_out}")

    # 7) Estatísticas rápidas
    total_2019 = int(daily.sum())
    pico_data = daily.idxmax()
    pico_val  = int(daily.max())
    print(f"[INFO] Total 2019: {total_2019} | Pico: {pico_val} em {pico_data.date()}")

if __name__ == "__main__":
    main()
