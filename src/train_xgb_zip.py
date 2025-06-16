import argparse
import os
import time
import joblib
import numpy as np

from data_handling.utils.data_utils import load_data
from evaluation.eval_utils import get_training_metrics, optimize_threshold, save_all_metrics, save_feature_importance, save_threshold
from evaluation.plots import plot_prediction_distribution, plot_learning_curve
from models.models import get_xgb_clf, get_xgb_poisson

def train_and_evaluate_zip(name, clf, reg, X_train, y_train, X_val, y_val, X_test, y_test, outdir, feature_dict):

    print("🔎 Preparando dados para etapa binária...")
    y_train_bin = (y_train > 0).astype(int)
    y_val_bin = (y_val > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    print("🧠 Treinando classificador binário...")
    clf.fit(X_train, y_train_bin, eval_set=[(X_train, y_train_bin), (X_val, y_val_bin)], verbose=False)

    print("📊 Separando dados com y > 0 para regressão...")
    X_train_reg = X_train[y_train > 0]
    y_train_reg = y_train[y_train > 0]
    X_val_reg = X_val[y_val > 0]
    y_val_reg = y_val[y_val > 0]
    X_test_reg = X_test[y_test > 0]
    y_test_reg = y_test[y_test > 0]

    print("🔢 Treinando regressor Poisson...")
    reg.fit(X_train_reg, y_train_reg, eval_set=[(X_train_reg, y_train_reg), (X_val_reg, y_val_reg)], verbose=False)

    print("🎯 Otimizando threshold baseado em validação...")
    val_prob = clf.predict_proba(X_val)[:, 1]
    val_pred_reg = reg.predict(X_val)
    best_thresh = optimize_threshold(val_prob, val_pred_reg, y_val)
    save_threshold(best_thresh, outdir)

    print("📈 Gerando predições finais...")
    prob_test = clf.predict_proba(X_test)[:, 1]
    y_pred_reg = reg.predict(X_test)
    y_pred = y_pred_reg * (prob_test > best_thresh)
    y_pred = np.round(np.maximum(y_pred, 0)).astype(int)

    print("🧮 Calculando métricas...")
    train_prob = clf.predict_proba(X_train)[:, 1]
    train_pred_reg = reg.predict(X_train)
    train_pred = train_pred_reg * (train_prob > best_thresh)
    train_pred = np.round(np.maximum(train_pred, 0)).astype(int)

    metrics_dict = {
        "Treino": get_training_metrics(y_train, train_pred),
        "Validação": get_training_metrics(y_val, val_pred_reg * (val_prob > best_thresh)),
        "Teste": get_training_metrics(y_test, y_pred)
    }

    print("💾 Salvando métricas...")
    save_all_metrics(metrics_dict, outdir)

    print("📉 Plotando curva de aprendizado...")
    plot_learning_curve(clf, outdir, f"{name}_Classifier")
    plot_learning_curve(reg, outdir, f"{name}_Regressor")

    print("📊 Plotando distribuição de predições...")
    plot_prediction_distribution(y_pred, f"{name}_ZIP", outdir)

    print("📌 Salvando importância das features...")
    if hasattr(reg, "feature_importances_"):
        save_feature_importance(f"{name}_Regressor", reg, X_train, outdir, feature_dict)

    return clf, reg, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento do modelo ZIP (Zero-Inflated Poisson)")
    parser.add_argument("--dataset", type=str, required=True, help="Arquivo .pickle com os dados")
    parser.add_argument("--outdir", type=str, required=True, help="Diretório para salvar resultados")
    parser.add_argument("--seed", type=int, required=True, help="Semente de aleatoriedade")
    parser.add_argument("--dict", type=str, required=True, help="Caminho para o dicionário de features (.csv)")

    args = parser.parse_args()
    start_time = time.time()

    seed = args.seed
    clf = get_xgb_clf(seed)
    reg = get_xgb_poisson(seed)

    print("📥 Carregando dados...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    clf, reg, y_pred = train_and_evaluate_zip(
        "XGBOOST-ZIP",
        clf,
        reg,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        args.outdir,
        feature_dict=args.dict
    )

    model_path = os.path.join(args.outdir, "model_xgb_zip.pkl")
    joblib.dump((clf, reg), model_path)
    print(f"✅ Modelos salvos em: {model_path}")

    duration = (time.time() - start_time) / 60
    print(f"⏱️ Duração total do script: {duration:.2f} minutos")
