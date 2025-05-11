import argparse
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance, mean_tweedie_deviance
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from statsmodels.discrete.count_model import ZeroInflatedPoisson
import statsmodels.api as sm

def get_model(model_name, seed):
    model_name = model_name.lower()
    if model_name == "randomforest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=seed,
            n_jobs=8,
            verbose=1
        )
    elif model_name == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=seed,
            n_jobs=8,
            verbosity=1,
            objective="reg:squarederror"
        )
    elif model_name == "xgboost_poisson":
        return XGBRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=seed,
            n_jobs=8,
            verbosity=1,
            objective="count:poisson",
            eval_metric="poisson-nloglik"
        )
    elif model_name == "xgboost_tweedie":
        return XGBRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=seed,
            n_jobs=8,
            verbosity=1,
            objective="reg:tweedie",
            tweedie_variance_power=1.1
        )
    elif model_name == "zip":
        return "ZIP"
    else:
        raise ValueError(f"\u274c Modelo não suportado: {model_name}")

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    )

def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test, outdir):   
    log_lines = []

    def metrics(label, y_true, y_pred):
        y_pred = np.maximum(y_pred, 0)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        non_zero_mask = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100 if np.any(non_zero_mask) else float('nan')
        try:
            poisson_dev = mean_poisson_deviance(y_true, y_pred)
            tweedie_dev = mean_tweedie_deviance(y_true, y_pred, power=1.1)
        except:
            poisson_dev = tweedie_dev = float('nan')
        smape_val = smape(y_true, y_pred)

        log_lines.append(f"\n📊 [{label}] - {name}")
        log_lines.append(f"  MSE : {mse:.4f}")
        log_lines.append(f"  RMSE: {rmse:.4f}")
        log_lines.append(f"  MAE : {mae:.4f}")
        log_lines.append(f"  R²  : {r2:.4f}")
        log_lines.append(f"  MAPE (ign. zeros): {mape:.2f}%")
        log_lines.append(f"  SMAPE: {smape_val:.2f}%")
        log_lines.append(f"  Poisson Deviance : {poisson_dev:.4f}")
        log_lines.append(f"  Tweedie Deviance : {tweedie_dev:.4f}")

    if model == "ZIP":
        # Treinamento com statsmodels
        X_train_df = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
        X_train_df = sm.add_constant(X_train_df)

        zip_model = ZeroInflatedPoisson(endog=y_train, exog=X_train_df, exog_infl=X_train_df, inflation='logit')
        result = zip_model.fit(maxiter=100, method="bfgs")

        log_lines.append("\n📑 Resumo do Modelo ZIP:")
        log_lines.append(result.summary().as_text())

        # Predição
        def predict_zip(X):
            X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            X_df = sm.add_constant(X_df)
            return np.maximum(result.predict(X_df), 0)

        y_pred_train = predict_zip(X_train)
        y_pred_val = predict_zip(X_val)
        y_pred_test = predict_zip(X_test)
    else:        
        model.fit(X_train, y_train)
        y_pred_train = np.round(model.predict(X_train))
        y_pred_val = np.round(model.predict(X_val))
        y_pred_test = np.round(model.predict(X_test))

    metrics("TREINO", y_train, y_pred_train)
    metrics("VALIDAÇÃO", y_val, y_pred_val)
    metrics("TESTE", y_test, y_pred_test)

    treino_teste = np.round(np.maximum(model.predict(X_test), 0)).astype(int)
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
        feature_names = pd.read_csv("feature_dictionary.csv")["Feature"].tolist()
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        importance_df.to_csv(os.path.join(outdir, f"feature_importance_{name.lower()}.csv"), index=False)

        # Gerar histograma
        import matplotlib.pyplot as plt
        top_n = 20
        plt.figure(figsize=(10, 6))
        importance_df.head(top_n).plot(kind='barh', x='Feature', y='Importance', legend=False)
        plt.title(f"Top {top_n} Features Mais Importantes - {name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"feature_importance_{name.lower()}.png"))
        plt.close()

    os.makedirs(outdir, exist_ok=True)
    result_txt_path = os.path.join(outdir, f"result_{name.lower()}.txt")
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return model, y_pred_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de modelo para previsão de dengue")
    parser.add_argument("--model", type=str, required=True,
                        choices=["randomforest", "xgboost", "xgboost_poisson", "xgboost_tweedie"],
                        help="Modelo a ser usado")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Caminho para o arquivo .pickle com os dados")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Diretório para salvar resultados")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed")    

    args = parser.parse_args()

    print(f"🔍 Treinando modelo: {args.model}")
    seed = 987
    if args.seed:
        seed = args.seed
    model = get_model(args.model, seed)

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
    print(f"📂 Modelo salvo em: {model_path}")
