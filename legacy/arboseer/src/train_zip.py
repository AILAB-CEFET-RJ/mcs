import argparse
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor, callback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss
from scipy.stats import pearsonr
from sklearn.metrics import mean_poisson_deviance

# Função para calcular o SMAPE - erro percentual absoluto simétrico
# Útil em previsões com valores muito baixos ou zeros
# Referência: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    )

# Gera e salva a curva de aprendizado de um modelo
# útil para visualizar o overfitting ou underfitting
# Referência: https://xgboost.readthedocs.io/en/stable/python/examples/plot_learning_curve.html
def plot_learning_curve(model, outdir, model_name):
    if hasattr(model, 'evals_result'):
        evals_result = model.evals_result()
        metric_name = list(evals_result['validation_0'].keys())[0]
        train_metric = evals_result['validation_0'][metric_name]
        val_metric = evals_result['validation_1'][metric_name]

        plt.figure(figsize=(8, 6))
        plt.plot(train_metric, label='Treino')
        plt.plot(val_metric, label='Validação')
        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Boosting Rounds")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        save_path = os.path.join(outdir, f"learning_curve_{model_name.lower()}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Learning curve salva em: {save_path}")

# Função principal de avaliação com modelo ZIP (Zero-Inflated Poisson)
# Referência sobre ZIP: https://en.wikipedia.org/wiki/Zero-inflated_model
# Referência teórica: Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. Technometrics, 34(1), 1–14.
# DOI: https://doi.org/10.1080/00401706.1992.10485228
def evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, outdir):
    log_lines = []

    # Etapa 1: classificação binária (0 casos vs. >0 casos)
    y_train_bin = (y_train > 0).astype(int)
    y_val_bin = (y_val > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    # Modelo classificador: prevê a presença ou ausência de casos
    # Referência: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=8,
        random_state=0,
        callbacks=[callback.EarlyStopping(rounds=20)]  # ⏱️ Early stopping também aplicado na classificação
    )
    clf.fit(X_train, y_train_bin, eval_set=[(X_val, y_val_bin)], verbose=True)

    # Etapa 2: regressão somente para os dados com casos > 0
    X_train_reg = X_train[y_train > 0]
    y_train_reg = y_train[y_train > 0]
    X_val_reg = X_val[y_val > 0]
    y_val_reg = y_val[y_val > 0]
    X_test_reg = X_test[y_test > 0]
    y_test_reg = y_test[y_test > 0]

    # Modelo regressivo Poisson: prevê a quantidade de casos, assumindo que ocorreram
    # Este regressor é aplicado apenas onde a classificação binária indicou que haverá casos
    # Referência: https://xgboost.readthedocs.io/en/stable/tutorials/poisson_regression.html
    # Este modelo é treinado apenas com os dados não-zero (parte de contagem do ZIP)
    reg = XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        n_jobs=8,
        random_state=0,
        # ⏱️ Early stopping para evitar overfitting na etapa de regressão
        callbacks=[callback.EarlyStopping(rounds=20)]
    )
    reg.fit(X_train_reg, y_train_reg, eval_set=[(X_val_reg, y_val_reg)], verbose=True)

    # Combina os modelos: previsão final é zero se classificador indicar ausência de casos
    prob_nonzero = clf.predict_proba(X_test)[:, 1]  # probabilidade de casos
    # 🔧 Otimização do threshold com base na validação
    # Aqui testamos múltiplos thresholds e escolhemos aquele com menor MSE no conjunto de validação
    best_thresh = 0.5
    best_score = float('inf')
    val_prob = clf.predict_proba(X_val)[:, 1]
    val_pred_reg = reg.predict(X_val)

    for t in np.linspace(0.01, 0.5, 25):
        val_zip = val_pred_reg * (val_prob > t)
        score = mean_squared_error(y_val, val_zip)
        if score < best_score:
            best_score = score
            best_thresh = t

    print(f"🔧 Threshold ótimo com base na validação: {best_thresh:.3f}")
    is_nonzero = prob_nonzero > best_thresh
    y_pred_reg = reg.predict(X_test)
    y_pred = y_pred_reg * is_nonzero

    # Métricas gerais de avaliação
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    smape_val = smape(y_test, y_pred)
    try:
        poisson_dev = mean_poisson_deviance(y_test, np.maximum(y_pred, 1e-6))
    except:
        poisson_dev = float('nan')

    # Log-verossimilhança negativa da parte binária
    # Referência: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    nloglik = log_loss(y_test_bin, prob_nonzero, eps=1e-15)

    # Correlação de Pearson entre predições e valores reais (mede alinhamento)
    rho, _ = pearsonr(y_test, y_pred)

    # Relatório textual
    log_lines.append("📊 Avaliação do modelo ZIP:")
    log_lines.append(f"MSE                : {mse:.4f}")
    log_lines.append(f"MAE                : {mae:.4f}")
    log_lines.append(f"R²                 : {r2:.4f}")
    log_lines.append(f"SMAPE              : {smape_val:.2f}%")
    log_lines.append(f"Poisson Deviance   : {poisson_dev:.4f}")
    log_lines.append(f"NLogLik (Binário)  : {nloglik:.4f}")
    log_lines.append(f"Correlação Pearson : {rho:.4f}")

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "result_xgb_zip.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Curvas de aprendizado
    plot_learning_curve(clf, outdir, "XGB_ZIP_Classifier")
    plot_learning_curve(reg, outdir, "XGB_ZIP_Regressor")

    # Salvamento dos modelos
    # Referência: https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html
    joblib.dump((clf, reg), os.path.join(outdir, "model_xgb_zip.pkl"))
    print(f"📂 Modelos salvos em: {outdir}")

# Ponto de entrada principal do script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    with open(args.dataset, "rb") as file:
        X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(file)

    # Redimensionamento para entrada do XGBoost
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, args.outdir)
