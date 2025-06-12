import os
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    )
         
def get_training_metrics(y_true, y_pred):
    # Arredondar para métricas de erro
    y_pred_rounded = np.round(np.maximum(y_pred, 0)).astype(int)
    y_true = y_true.astype(int)

    # Métricas comuns
    mse = mean_squared_error(y_true, y_pred_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_rounded)
    r2 = r2_score(y_true, y_pred_rounded)

    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred_rounded[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = None

    smape_val = smape(y_true, y_pred_rounded)

    # Pearson precisa de valores contínuos
    try:
        pearson_corr = pearsonr(y_true, y_pred_rounded)[0]
    except:
        pearson_corr = None

    # Poisson Deviance usa valores contínuos e positivos
    try:
        poisson_dev = mean_poisson_deviance(y_true, np.maximum(y_pred, 1e-10))
    except:
        poisson_dev = None

    # Conversão segura para tipos nativos
    def safe(val):
        return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else None

    return {
        "MSE": safe(mse),
        "RMSE": safe(rmse),
        "MAE": safe(mae),
        "R2": safe(r2),
        "MAPE (ign. zeros)": safe(mape),
        "SMAPE": safe(smape_val),
        "Poisson_Deviance": safe(poisson_dev),
        "Pearson": safe(pearson_corr)
    }
    
def save_all_metrics(metrics_dict, outdir):
    df = pd.DataFrame.from_dict(metrics_dict, orient="index").reset_index()
    df = df.rename(columns={"index": "Conjunto"})
    df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

def save_feature_importance(name, model, X_train, outdir, feature_dictionary):
    importances = model.feature_importances_

        # Carrega o nome original das features
    try:
        original_features = pd.read_csv(feature_dictionary)["Feature"].tolist()
    except Exception as e:
        print(f"⚠️ Erro ao carregar feature dictionary: {e}")
        original_features = [f"f{i}" for i in range(X_train.shape[1] // window_size)]

        # Gera nomes expandidos considerando sliding window
    def get_sliding_feature_names(original_features, window_size):
        return [
                f"{f}_t-{i}" for i in reversed(range(window_size)) for f in original_features
            ]

    window_size = X_train.shape[1] // len(original_features)
    all_features = get_sliding_feature_names(original_features, window_size)

    if len(all_features) != len(importances):
        print(f"⚠️ Tamanho dos nomes de features ({len(all_features)}) difere de importances ({len(importances)}). Gerando nomes genéricos.")
        all_features = [f"Feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({
            "Feature": all_features,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

    importance_df.to_csv(os.path.join(outdir, f"feature_importance_{name.lower()}.csv"), index=False)

    # Histograma das top 20
    import matplotlib.pyplot as plt
    top_n = 20
    plt.figure(figsize=(10, 6))
    importance_df.head(top_n).plot(kind='barh', x='Feature', y='Importance', legend=False)
    plt.title(f"Top {top_n} Features Mais Importantes - {name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"feature_importance_{name.lower()}.png"))
    plt.close()    

def optimize_threshold(prob_nonzero, y_pred_reg, y_true):
    best_thresh = 0.5
    best_score = float("inf")

    for t in np.linspace(0.01, 0.5, 25):
        y_zip = y_pred_reg * (prob_nonzero > t)
        score = mean_squared_error(y_true, y_zip)
        if score < best_score:
            best_score = score
            best_thresh = t
            
    return best_thresh

def save_threshold(thresh, outdir):
    with open(os.path.join(outdir, "best_threshold.txt"), "w") as f:
        f.write(f"{thresh:.4f}")