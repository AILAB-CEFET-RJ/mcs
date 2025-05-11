import numpy as np
import pandas as pd
import argparse
import pickle
from scipy.stats import skew, kurtosis

def diagnostico_y(y, conjunto):
    y = np.asarray(y)
    total = len(y)

    faixa_0 = np.sum(y == 0)
    faixa_1_5 = np.sum((y >= 1) & (y <= 5))
    faixa_6_20 = np.sum((y > 5) & (y <= 20))
    faixa_maior_20 = np.sum(y > 20)

    media = np.mean(y)
    variancia = np.var(y)
    assimetria = skew(y)
    curtose_val = kurtosis(y)

    return {
        "Conjunto": conjunto,
        "Tamanho": total,
        "Zeros (%)": round(100 * faixa_0 / total, 2),
        "Média": round(media, 2),
        "Variância": round(variancia, 2),
        "Variância / Média": round(variancia / media, 2) if media > 0 else 0,
        "Assimetria": round(assimetria, 2),
        "Curtose": round(curtose_val, 2),
        "0": faixa_0,
        "1–5": faixa_1_5,
        "6–20": faixa_6_20,
        ">20": faixa_maior_20
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_path", help="Caminho para o arquivo .pickle com (X_train, y_train, X_val, y_val, X_test, y_test)")
    parser.add_argument("--saida_csv", default="diagnostico_pickle_saida.csv", help="Caminho do arquivo CSV de saída")

    args = parser.parse_args()

    with open(args.pickle_path, "rb") as f:
        X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

    registros = [
        diagnostico_y(y_train, "Treino"),
        diagnostico_y(y_val, "Validação"),
        diagnostico_y(y_test, "Teste")
    ]

    df = pd.DataFrame(registros)
    df.to_csv(args.saida_csv, index=False)
    print(f"✅ Diagnóstico salvo em {args.saida_csv}")

if __name__ == "__main__":
    main()
