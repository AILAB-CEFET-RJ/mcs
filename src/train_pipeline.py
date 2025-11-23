# train_pipeline.py  (same file where your main() is)

import os
import yaml
import joblib
from data_handling.utils.data_utils import load_data
from models.models import get_rf, get_xgb_poisson, get_xgb_clf
from train_xgb_poisson import train_and_evaluate as train_xgb_poisson
from train_rf import train_and_evaluate as train_rf
from train_xgb_zip import train_and_evaluate_zip

def run_training(config_path: str, hyperparams_by_model=None, seeds=None):
    """
    Runs robust training using base YAML, but lets caller override:
      - hyperparams_by_model: dict like {"xgb_poisson": {...}, "random_forest": {...}, "xgb_zip": {"clf": {...}, "reg": {...}}}
      - seeds: list[int]
    If overrides are None, defaults + YAML behavior are used.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_path = config["dataset"]
    dict_path = config["feature_dict"]
    outdir = config["output_dir"]

    # base seed from YAML (still supported)
    base_seed = config.get("seed", 42)

    # if caller didn’t pass seeds, fall back to YAML single seed
    seeds = seeds if seeds is not None else [base_seed]

    print(f"📥 Carregando dados de: {dataset_path}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_path)

    hp = hyperparams_by_model or {}

    for seed in seeds:
        print(f"\n🎲 Rodando seed={seed}")

        if config["models"].get("random_forest", False):
            print("🌲 Treinando Random Forest...")
            rf_hp = hp.get("random_forest", {})
            rf_model = get_rf(seed, hyperparams=rf_hp)
            rf_dir = f"{outdir}_{seed}_rf"
            os.makedirs(rf_dir, exist_ok=True)

            model_rf, _ = train_rf(
                "Random Forest", rf_model,
                X_train, y_train, X_val, y_val, X_test, y_test,
                rf_dir, dict_path
            )
            joblib.dump(model_rf, os.path.join(rf_dir, "model.pkl"))

        if config["models"].get("xgb_poisson", False):
            print("📈 Treinando XGBoost Poisson...")
            xgb_hp = hp.get("xgb_poisson", {})
            xgb_model = get_xgb_poisson(seed, hyperparams=xgb_hp)
            xgb_dir = f"{outdir}_{seed}_xgb_poisson"
            os.makedirs(xgb_dir, exist_ok=True)

            model_xgb, _ = train_xgb_poisson(
                "XGBOOST - Poisson", xgb_model,
                X_train, y_train, X_val, y_val, X_test, y_test,
                xgb_dir, dict_path
            )
            joblib.dump(model_xgb, os.path.join(xgb_dir, "model.pkl"))

        if config["models"].get("xgb_zip", False):
            print("🎯 Treinando XGBoost ZIP...")
            zip_hp = hp.get("xgb_zip", {})
            clf_hp = zip_hp.get("clf", {})
            reg_hp = zip_hp.get("reg", {})

            clf = get_xgb_clf(seed, hyperparams=clf_hp)
            reg = get_xgb_poisson(seed, hyperparams=reg_hp)

            xgb_zip_dir = f"{outdir}_{seed}_xgb_zip"
            os.makedirs(xgb_zip_dir, exist_ok=True)

            clf_d, reg_d, _ = train_and_evaluate_zip(
                "XGBOOST-ZIP", clf, reg,
                X_train, y_train, X_val, y_val, X_test, y_test,
                xgb_zip_dir, dict_path
            )
            joblib.dump((clf_d, reg_d), os.path.join(xgb_zip_dir, "model.pkl"))


def main(config_path: str):
    # old behavior preserved
    run_training(config_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
