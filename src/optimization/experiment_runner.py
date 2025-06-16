# src/optimization/experiment_runner.py

import argparse
import pickle
import optuna
import os
import numpy as np

from optimization import adaptive_spaces, objective_functions
from evaluation.eval_utils import get_training_metrics
from data_handling.utils.data_utils import load_data

def run_single(dataset_path, model_type, trials, directions):

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_path)

    space = getattr(adaptive_spaces, f"initial_{model_type}_space")
    suggest_fn = getattr(adaptive_spaces, f"suggest_{model_type}")
    objective_fn = getattr(objective_functions, f"objective_{model_type}")

    study = optuna.create_study(
        directions=directions,
        study_name=f"debug_{model_type}",
    )

    def optuna_objective(trial):
        params = suggest_fn(trial, space)
        metrics = objective_fn(trial, X_train, y_train, X_val, y_val, params)
        return list(metrics.values())

    study.optimize(optuna_objective, n_trials=trials)

    # Avalia no teste
    best_trial = study.best_trials[0]
    best_params = suggest_fn(best_trial, space)
    model_final = objective_fn(None, X_train, y_train, X_test, y_test, best_params, return_model=True)

    y_pred = model_final.predict(X_test)
    y_pred = np.round(np.maximum(y_pred, 0)).astype(int)
    test_metrics = get_training_metrics(y_test, y_pred)

    print("\n📊 Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["poisson", "zip", "rf"], required=True)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    # As direções agora são fixas e alinhadas com o experiments.yaml
    directions = [
        "minimize",  # MSE
        "minimize",  # RMSE
        "minimize",  # MAE
        "minimize",  # -R2
        "minimize",  # MAPE
        "minimize",  # SMAPE
        "minimize",  # -Rho
        "minimize",  # Poisson Deviance
    ]

    run_single(args.dataset, args.model, args.trials, directions)
