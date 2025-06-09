import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance, mean_tweedie_deviance
from scipy.stats import pearsonr
from utils import smape, plot_learning_curve

def evaluate_model_basic(model, X_train, y_train, X_val, y_val, X_test, y_test, outdir, model_name="modelo"):
    os.makedirs(outdir, exist_ok=True)
    log_lines = []

    def evaluate_split(name, y_true, y_pred):
        y_pred = np.maximum(y_pred, 0)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        non_zero_mask = y_true != 0
        mape = (np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])).mean() * 100 if np.any(non_zero_mask) else float("nan")
        try:
            poisson_dev = mean_poisson_deviance(y_true, y_pred)
            tweedie_dev = mean_tweedie_deviance(y_true, y_pred, power=1.1)
        except:
            poisson_dev = tweedie_dev = float("nan")
        smape_val = smape(y_true, y_pred)

        log_lines.append(f"\n📊 [{name}]")
        log_lines.append(f"  MSE               : {mse:.4f}")
        log_lines.append(f"  RMSE              : {rmse:.4f}")
        log_lines.append(f"  MAE               : {mae:.4f}")
        log_lines.append(f"  R²                : {r2:.4f}")
        log_lines.append(f"  MAPE (ign. zeros) : {mape:.2f}%")
        log_lines.append(f"  SMAPE             : {smape_val:.2f}%")
        log_lines.append(f"  Poisson Deviance  : {poisson_dev:.4f}")
        log_lines.append(f"  Tweedie Deviance  : {tweedie_dev:.4f}")

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    if hasattr(model, "best_iteration"):
        y_pred_train = model.predict(X_train, iteration_range=(0, model.best_iteration + 1))
        y_pred_val = model.predict(X_val, iteration_range=(0, model.best_iteration + 1))
        y_pred_test = model.predict(X_test, iteration_range=(0, model.best_iteration + 1))
    else:
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

    evaluate_split("TREINO", y_train, y_pred_train)
    evaluate_split("VALIDAÇÃO", y_val, y_pred_val)
    evaluate_split("TESTE", y_test, y_pred_test)

    erro_abs = np.abs(y_test - y_pred_test)
    corr, _ = pearsonr(y_test, y_pred_test)
    log_lines.append(f"\n📈 Avaliação Final (Teste)")
    log_lines.append(f"Erro Absoluto Médio : {np.mean(erro_abs):.4f}")
    log_lines.append(f"Correlação de Pearson: {corr:.4f}")

    result_df = pd.DataFrame({
        "Real": y_test,
        "Predito": y_pred_test,
        "Erro": y_test - y_pred_test,
        "Erro Absoluto": erro_abs
    })
    result_df.to_csv(os.path.join(outdir, f"predictions_{model_name.lower()}.csv"), index=False)

    with open(os.path.join(outdir, f"result_{model_name.lower()}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    plot_learning_curve(model, outdir, model_name)

    return model, y_pred_test