# train_pipeline.py  (same file where your main() is)

import os
import yaml
import joblib
from data_handling.utils.data_utils import load_selection_data
from models.models import get_rf, get_xgb_poisson, get_xgb_clf
from train_xgb_poisson import train_and_evaluate as train_xgb_poisson
from train_rf import train_and_evaluate as train_rf
from train_xgb_zip import train_and_evaluate_zip

def run_training(
    config_path: str,
    hyperparams_by_model=None,
    seeds=None,
    output_tag=None,
    selected_models=None,
    skip_completed=False,
    evaluation_policy="selection",
):
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
    if output_tag:
        outdir = f"{outdir}_{output_tag}"

    # base seed from YAML (still supported)
    base_seed = config.get("seed", 42)

    # if caller didn’t pass seeds, fall back to YAML single seed
    seeds = seeds if seeds is not None else [base_seed]

    print(f"Carregando dados de: {dataset_path}")
    if evaluation_policy == "selection":
        X_train, y_train, X_val, y_val = load_selection_data(dataset_path)
        X_test = y_test = None
        print("Politica selection: artefato de confirmacao nao foi aberto")
    else:
        raise ValueError("Este trainer é selection-only; use um avaliador bloqueado para confirmação/final")

    hp = hyperparams_by_model or {}
    selected_models = set(selected_models) if selected_models is not None else None

    def should_train(model_name):
        return config["models"].get(model_name, False) and (
            selected_models is None or model_name in selected_models
        )

    def output_complete(model_dir):
        required = ("model.pkl", "metrics.csv") if evaluation_policy == "selection" else ("model.pkl", "metrics.csv", "predictions.csv")
        return all(
            os.path.isfile(os.path.join(model_dir, filename))
            and os.path.getsize(os.path.join(model_dir, filename)) > 0
            for filename in required
        )

    for seed in seeds:
        print(f"\nRodando seed={seed}")

        if should_train("random_forest"):
            print("Treinando Random Forest...")
            rf_hp = hp.get("random_forest", {})
            rf_model = get_rf(seed, hyperparams=rf_hp)
            rf_dir = f"{outdir}_{seed}_rf"
            if skip_completed and output_complete(rf_dir):
                print(f"Random Forest ja concluido em {rf_dir}")
                continue
            os.makedirs(rf_dir, exist_ok=True)

            model_rf, _ = train_rf(
                "Random Forest", rf_model,
                X_train, y_train, X_val, y_val, X_test, y_test,
                rf_dir, dict_path,
                generate_learning_curve=config.get("generate_rf_learning_curve", False),
            )
            joblib.dump(model_rf, os.path.join(rf_dir, "model.pkl"))

        if should_train("xgb_poisson"):
            print("Treinando XGBoost Poisson...")
            xgb_hp = hp.get("xgb_poisson", {})
            xgb_model = get_xgb_poisson(seed, hyperparams=xgb_hp)
            xgb_dir = f"{outdir}_{seed}_xgb_poisson"
            if skip_completed and output_complete(xgb_dir):
                print(f"XGBoost Poisson ja concluido em {xgb_dir}")
                continue
            os.makedirs(xgb_dir, exist_ok=True)

            model_xgb, _ = train_xgb_poisson(
                "XGBOOST - Poisson", xgb_model,
                X_train, y_train, X_val, y_val, X_test, y_test,
                xgb_dir, dict_path
            )
            joblib.dump(model_xgb, os.path.join(xgb_dir, "model.pkl"))

        if should_train("xgb_zip"):
            print("Treinando Bernoulli-Poisson em dois estagios...")
            zip_hp = hp.get("xgb_zip", {})
            clf_hp = zip_hp.get("clf", {})
            reg_hp = zip_hp.get("reg", {})

            clf = get_xgb_clf(seed, hyperparams=clf_hp)
            reg = get_xgb_poisson(seed, hyperparams=reg_hp)

            xgb_zip_dir = f"{outdir}_{seed}_xgb_zip"
            if skip_completed and output_complete(xgb_zip_dir):
                print(f"Bernoulli-Poisson ja concluido em {xgb_zip_dir}")
                continue
            os.makedirs(xgb_zip_dir, exist_ok=True)

            clf_d, reg_d, _ = train_and_evaluate_zip(
                "XGBOOST-ZIP", clf, reg,
                X_train, y_train, X_val, y_val, X_test, y_test,
                xgb_zip_dir, dict_path
            )
            joblib.dump((clf_d, reg_d), os.path.join(xgb_zip_dir, "model.pkl"))


def main(config_path: str, output_tag=None):
    # old behavior preserved
    run_training(config_path, output_tag=output_tag)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--evaluation-policy", choices=["selection"], default="selection")
    args = parser.parse_args()
    run_training(args.config, output_tag=args.output_tag, evaluation_policy=args.evaluation_policy)
