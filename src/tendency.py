import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

import data_handling.utils.data_utils as du


def analyze_trend(
    y_train,
    y_val,
    y_test,
    season_period: int = 7,
    rolling_window: int = 30,
    series_name: str = "Casos de dengue - RJ (diário)"
):
    """
    Analisa tendência (trend) de uma série temporal de casos:
    - Concatena train/val/test
    - Plota série e média móvel
    - Aplica teste de Mann-Kendall
    - Aplica teste ADF (estacionariedade)
    - Faz decomposição STL (trend + residual)
    - Retorna um dicionário com resultados numéricos
    """

    # 1) Concatena toda a série
    y_full = np.concatenate([y_train, y_val, y_test])
    y_full = pd.Series(y_full)

    # 2) Plot série completa + média móvel
    plt.figure(figsize=(14, 4))
    plt.plot(y_full, label="Série original", alpha=0.7)

    if rolling_window is not None and rolling_window > 1:
        y_mm = y_full.rolling(rolling_window, center=True).mean()
        plt.plot(y_mm, label=f"Média móvel ({rolling_window} períodos)", linewidth=2)

    plt.title(f"Série completa de casos ({series_name})")
    plt.xlabel("Tempo (índice)")
    plt.ylabel("Casos")
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig1.png')

    # 3) Teste de Mann-Kendall (tendência monotônica)
    mk_result = mk.original_test(y_full)

    # 4) Teste ADF (estacionariedade)
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(y_full, autolag="AIC")

    # 5) Decomposição STL para extrair trend
    stl = STL(y_full, period=season_period, robust=True).fit()
    trend = stl.trend
    seasonal = stl.seasonal
    resid = stl.resid
    detrended = y_full - trend

    # 5.1) Plot trend e série detrendada
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(y_full, label="Original", alpha=0.7)
    axes[0].plot(trend, label="Trend (STL)", linewidth=2)
    axes[0].set_title("Série original + tendência (STL)")
    axes[0].legend()

    axes[1].plot(seasonal)
    axes[1].set_title("Componente sazonal (STL)")

    axes[2].plot(detrended)
    axes[2].set_title("Série detrendida (original - trend)")

    plt.tight_layout()
    plt.savefig('fig2.png')

    # 6) Resumo textual rápido
    print("\n===== RESULTADOS DE TENDÊNCIA =====")
    print(f"Man-Kendall trend     : {mk_result.trend}")
    print(f"Man-Kendall p-value   : {mk_result.p}")
    print(f"Man-Kendall slope     : {mk_result.sen_slope}")

    print("\nADF (estacionariedade)")
    print(f"ADF statistic         : {adf_stat:.4f}")
    print(f"ADF p-value           : {adf_p:.4f}")
    print("ADF critical values   :")
    for k, v in adf_crit.items():
        print(f"  {k}: {v:.4f}")

    if mk_result.p < 0.05:
        print("\nInterpretação MK: há uma tendência monotônica estatisticamente significativa (p < 0.05).")
    else:
        print("\nInterpretação MK: não rejeitamos H0 de ausência de tendência monotônica (p >= 0.05).")

    if adf_p < 0.05:
        print("Interpretação ADF: rejeitamos H0 de raiz unitária → série mais próxima de estacionária.")
    else:
        print("Interpretação ADF: não rejeitamos H0 de raiz unitária → série não estacionária (possível trend).")

    # 7) Retorno estruturado (para usar depois em tabelas/capítulo)
    results = {
        "mk_trend": mk_result.trend,
        "mk_pvalue": mk_result.p,
        "mk_sen_slope": mk_result.sen_slope,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_p,
        "adf_crit_values": adf_crit,
        "y_full": y_full,
        "trend": trend,
        "seasonal": seasonal,
        "residuals": resid,
        "detrended": detrended,
    }

    return results


if __name__ == "__main__":
    # Carrega os dados
    X_train, y_train, X_val, y_val, X_test, y_test = du.load_data(
        "data/datasets/RJ_DAILY/dataset.pickle"
    )

    # Para RJ_DAILY, provavelmente há sazonalidade semanal (7 dias).
    # Se você estiver analisando série SEMANAL, troque para season_period=52.
    trend_results = analyze_trend(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        season_period=7,          # ajuste se quiser anual: 365, ou semanal agregada: 52
        rolling_window=30,
        series_name="Casos de dengue - RJ (diário)"
    )
