import argparse
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def get_model(model_name):
    model_name = model_name.lower()
    if model_name == "randomforest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=987,
            n_jobs=8,
            verbose=1
        )
    elif model_name == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=987,
            n_jobs=8,
            verbosity=1
        )
    else:
        raise ValueError(f"❌ Modelo não suportado: {model_name}")

def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test, outdir):
    model.fit(X_train, y_train)
    log_lines = []

    def metrics(label, y_true, y_pred):
        y_pred = np.maximum(y_pred, 0)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        non_zero_mask = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100 \
            if np.any(non_zero_mask) else float('nan')

        log_lines.append(f"\n📊 [{label}] - {name}")
        log_lines.append(f"  MSE : {mse:.4f}")
        log_lines.append(f"  RMSE: {rmse:.4f}")
        log_lines.append(f"  MAE : {mae:.4f}")
        log_lines.append(f"  R²  : {r2:.4f}")
        log_lines.append(f"  MAPE (ign. zeros): {mape:.2f}%")

    y_pred_train = np.round(model.predict(X_train))
    y_pred_val = np.round(model.predict(X_val))
    y_pred_test = np.round(model.predict(X_test))

    metrics("TREINO", y_train, y_pred_train)
    metrics("VALIDAÇÃO", y_val, y_pred_val)
    metrics("TESTE", y_test, y_pred_test)

    treino_teste = np.maximum(model.predict(X_test), 0)
    y_true = y_test
    erro_abs = np.abs(y_true - treino_teste)
    erro = y_true - treino_teste

    result_df = pd.concat([
        pd.DataFrame(y_true, columns=["Cases"]),
        pd.DataFrame(treino_teste, columns=["Predito"]),
        pd.DataFrame(erro_abs, columns=["Erro Absoluto"]),
        pd.DataFrame(erro, columns=["Erro"])
    ], axis=1)
    
    result_csv_path = os.path.join(outdir, f"predictions_{name.lower()}.csv")
    result_df.to_csv(result_csv_path, index=False)

    erro_medio = np.mean(erro_abs)
    rmse = np.sqrt(mean_squared_error(y_true, treino_teste))
    r2 = r2_score(y_true, treino_teste)
    corr, _ = pearsonr(y_true, treino_teste)

    log_lines.append(f"\n📈 Avaliação Final no Conjunto de Teste:")
    log_lines.append(f"Erro absoluto médio: {erro_medio:.3f}")
    log_lines.append(f"RMSE               : {rmse:.3f}")
    log_lines.append(f"R²                 : {r2:.3f}")
    log_lines.append(f"Correlação Pearson : {corr:.3f}")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        log_lines.append("\n🔎 Top 10 variáveis mais importantes:")
        for i in np.argsort(importances)[::-1][:10]:
            log_lines.append(f"Feature {i}: Importance = {importances[i]:.4f}")

    # 🔻 Salvar log
    os.makedirs(outdir, exist_ok=True)
    result_txt_path = os.path.join(outdir, f"result_{name.lower()}.txt")
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return model, y_pred_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de modelo para previsão de dengue")
    parser.add_argument("--model", type=str, required=True, choices=["randomforest", "xgboost"],
                        help="Modelo a ser usado: 'randomforest' ou 'xgboost'")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Caminho para o arquivo .pickle com os dados (X_train, y_train, etc)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Diretório para salvar resultados (modelo, CSV, avaliação)")

    args = parser.parse_args()

    print(f"🔍 Treinando modelo: {args.model}")
    model = get_model(args.model)

    with open(args.dataset, "rb") as file:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model, y_pred = evaluate_model(args.model.capitalize(), model,
                                   X_train, y_train, X_val, y_val, X_test, y_test,
                                   outdir=args.outdir)

    model_path = os.path.join(args.outdir, f"model_{args.model.lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Modelo salvo em: {model_path}")
