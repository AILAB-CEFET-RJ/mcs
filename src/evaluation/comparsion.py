# === comparison.py ===

import os
import pandas as pd

def load_consolidated_metrics(path="consolidated_metrics.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    return df

def compare_models(df, focus="Teste", export_path="comparativo_pre_otimizacao.xlsx"):
    """
    Gera o leaderboard agrupado por modelo + variante.
    """

    # Filtrar o conjunto desejado (default: Teste)
    df_focus = df[df["Conjunto"].str.lower() == focus.lower()]

    # Selecionar métricas principais
    metricas = ["MSE", "RMSE", "MAE", "R2", "SMAPE", "Poisson_Deviance", "Pearson"]

    # Agrupar e calcular média por modelo e variante
    resultados = df_focus.groupby(["Model", "Variant"])[metricas].mean().reset_index()

    # Ordenar para visualização
    resultados = resultados.sort_values(by="MSE")

    print("📊 Leaderboard (média por modelo e variante):")
    print(resultados)

    # Exportar para Excel (opcional)
    resultados.to_excel(export_path, index=False)
    print(f"✅ Comparativo salvo em: {export_path}")

    return resultados

def main():
    df = load_consolidated_metrics("consolidated_metrics.csv")
    compare_models(df)

if __name__ == "__main__":
    main()
