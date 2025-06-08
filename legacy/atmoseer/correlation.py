import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def plot_feature_target_correlation(df, target_name='Target'):
    # Separa features e target
    X = df.drop(columns=[target_name])
    y = df[target_name]
    print('Test')

    # Correlação de Pearson entre as features e o target
    correlation_matrix = X.corrwith(y, method='pearson')

    # Informação Mútua (avaliação não linear)
    if y.nunique() < 10:  # Exemplo: se for um problema de classificação
        mi_scores = mutual_info_classif(X, y)
    else:  # Se for um problema de regressão
        mi_scores = mutual_info_regression(X, y)
    
    mi_scores = pd.Series(mi_scores, index=X.columns, name="Mutual Information")

    # Exibe a matriz de correlação usando um heatmap
    plt.figure(figsize=(8, 10))
    correlation_df = pd.DataFrame({'Pearson': correlation_matrix, 'Mutual Information': mi_scores})
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlação das Features com o Target ({target_name})')
    plt.savefig('correlation.png')
    
    return correlation_df


def main():
    parser = argparse.ArgumentParser(description="Análise de correlação entre features e target")
    parser.add_argument("parquet_path", help="Caminho do arquivo .parquet contendo o dataset")
    parser.add_argument("target_name", help="Nome da coluna target")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)

    correlation = plot_feature_target_correlation(df, target_name=args.target_name)
    print(correlation)

if __name__ == "__main__":
    main()
