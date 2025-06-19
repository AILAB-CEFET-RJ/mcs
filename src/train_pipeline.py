import os
import yaml
import joblib
from data_handling.utils.data_utils import load_data
from models.models import get_rf, get_xgb_poisson, get_xgb_clf
from train_xgb_poisson import train_and_evaluate as train_xgb_poisson
from train_rf import train_and_evaluate as train_rf
from train_xgb_zip import train_and_evaluate_zip

def main(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_path = config["dataset"]
    dict_path = config["feature_dict"]
    outdir = config["output_dir"]
    seed = config.get("seed", 42)

    os.makedirs(outdir, exist_ok=True)
    print(f"📥 Carregando dados de: {dataset_path}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_path)

    if config["models"].get("random_forest", False):
        print("🌲 Treinando Random Forest...")
        rf_model = get_rf(seed)
        train_rf("Random Forest", rf_model,
                 X_train, y_train, X_val, y_val, X_test, y_test,
                 outdir, dict_path)

    if config["models"].get("xgb_poisson", False):
        print("📈 Treinando XGBoost Poisson...")
        xgb_model = get_xgb_poisson(seed)
        train_xgb_poisson("XGBOOST - Poisson", xgb_model,
                          X_train, y_train, X_val, y_val, X_test, y_test,
                          outdir, dict_path)

    if config["models"].get("xgb_zip", False):
        print("🎯 Treinando XGBoost ZIP...")
        clf = get_xgb_clf(seed)
        reg = get_xgb_poisson(seed)
        train_and_evaluate_zip("XGBOOST-ZIP", clf, reg,
                               X_train, y_train, X_val, y_val, X_test, y_test,
                               outdir, dict_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento de Modelos para Previsão de Dengue")
    parser.add_argument("--config", required=True, help="Caminho para arquivo YAML de configuração")
    args = parser.parse_args()

    main(args.config)