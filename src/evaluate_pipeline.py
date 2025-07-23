import argparse
import os
import time
import joblib
import numpy as np
import pandas as pd
import yaml

from data_handling.utils.data_utils import load_data
from evaluation.eval_utils import get_training_metrics, save_all_metrics, save_feature_importance
from evaluation.plots import plot_prediction_distribution
from models.models import get_xgb_poisson, get_xgb_clf

def evaluate_model_on_new_data(model_dir, dataset_path, outdir, feature_dict_path):
    print(f"\n📦 Avaliando modelo na pasta: {model_dir}")
    model_name = os.path.basename(model_dir)

    # Identificar modelo salvo
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not model_files:
        print(f"⚠️ Nenhum modelo .pkl encontrado em {model_dir}")
        return

    model_path = os.path.join(model_dir, model_files[0])
    clf, reg = None, None

    # Verificar se é ZIP
    if "zip" in model_name.lower():
        clf, reg = joblib.load(model_path)
        thresh_path = os.path.join(model_dir, "best_threshold.txt")
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                threshold = float(f.read().strip())
        else:
            print("⚠️ Threshold não encontrado. Usando 0.5.")
            threshold = 0.5
    else:
        reg = joblib.load(model_path)

    # Carregar dataset
    print("📥 Carregando dados...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_path)

    print("🔮 Gerando predições...")
    if clf is not None:
        prob_test = clf.predict_proba(X_test)[:, 1]
        y_pred_reg = reg.predict(X_test)
        y_pred = y_pred_reg * (prob_test > threshold)
        y_pred = np.round(np.maximum(y_pred, 0)).astype(int)
    else:
        y_pred = reg.predict(X_test)
        y_pred = np.round(np.maximum(y_pred, 0)).astype(int)

    print("📊 Calculando métricas...")
    metrics_dict = {
        "Transferência": get_training_metrics(y_test, y_pred)
    }
    save_all_metrics(metrics_dict, outdir)

    print("📉 Plotando distribuição das predições...")
    plot_prediction_distribution(y_pred, model_name + "_transfer", outdir)

    print("📌 Salvando importância das features...")
    if hasattr(reg, "feature_importances_"):
        save_feature_importance(model_name + "_transfer", reg, X_train, outdir, feature_dict_path)

    print(f"✅ Avaliação finalizada para {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de modelos em novo dataset")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o arquivo YAML de configuração")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_path = config["DATASET"]
    outdir = config["OUTDIR"]
    model_dirs = config["MODEL_DIRS"]

    os.makedirs(outdir, exist_ok=True)

    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            csv_files = [f for f in os.listdir(model_dir) if f.endswith(".csv") and "feature_importance" in f]
            if not csv_files:
                print(f"⚠️ Nenhum arquivo de feature_importance encontrado em {model_dir}")
                continue
            dict_path = os.path.join(model_dir, csv_files[0])
            evaluate_model_on_new_data(model_dir, dataset_path, outdir, dict_path)
        else:
            print(f"🚫 Diretório inválido: {model_dir}")
