# src/optimization/adaptive_controller.py (ajustado para trials variáveis)

import sys
import os

from data.data_utils import load_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pickle
import numpy as np
import optuna
import json
from datetime import datetime

from models.objective_functions import objective_poisson, objective_zip, objective_rf
from optimization.adaptive_spaces import suggest_poisson, suggest_zip, suggest_rf
from optimization.adaptive_spaces import (
    initial_poisson_space, 
    initial_zip_space, 
    initial_rf_space
)
from optimization.space_refiner import refine_space
from utils.utils import setup_logging, seed_everything, save_study_results
from optimization.config import SEED, RUNS_DIR

def run_single_round(dataset_path, model_type, trials, round_id, search_space):
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{dataset_name}_{model_type}_R{round_id}_{timestamp}"

    output_dir = os.path.join(RUNS_DIR, study_name)
    os.makedirs(output_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(output_dir, 'study.db')}"

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_path)

    study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name)

    def objective(trial):
        if model_type == "poisson":
            params = suggest_poisson(trial, search_space)
            return objective_poisson(trial, X_train, y_train, X_val, y_val, params)
        elif model_type == "zip":
            params = suggest_zip(trial, search_space)
            return objective_zip(trial, X_train, y_train, X_val, y_val, params)
        elif model_type == "rf":
            params = suggest_rf(trial, search_space)
            return objective_rf(trial, X_train, y_train, X_val, y_val, params)

    study.optimize(objective, n_trials=trials)
    save_study_results(study, output_dir)
    return study

def run_adaptive_pipeline(dataset_path, model_type, trials_list):
    if model_type == "poisson":
        current_space = initial_poisson_space
    elif model_type == "zip":
        current_space = initial_zip_space
    elif model_type == "rf":
        current_space = initial_rf_space

    for round_id, trials_in_round in enumerate(trials_list, start=1):
        print(f"\n🔄 Dataset: {os.path.basename(dataset_path)} | Modelo: {model_type} | Rodada: {round_id}/{len(trials_list)}")
        study = run_single_round(dataset_path, model_type, trials_in_round, round_id, current_space)
        current_space = refine_space(study, current_space)
        print(f"🔧 Novo espaço de busca: {current_space}")

if __name__ == "__main__":
    setup_logging()
    seed_everything(SEED)

    parser = argparse.ArgumentParser(description="Controlador Adaptativo Completo")
    parser.add_argument("--datasets", nargs="+", required=True, help="Lista de datasets pickle")
    parser.add_argument("--models", nargs="+", choices=["poisson", "zip", "rf"], required=True)
    parser.add_argument("--trials", nargs="+", type=int, required=True, help="Lista de trials por rodada")

    args = parser.parse_args()

    for dataset_path in args.datasets:
        for model_type in args.models:
            run_adaptive_pipeline(dataset_path, model_type, args.trials)
