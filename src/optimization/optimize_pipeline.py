# === optimize_pipeline.py ===

import argparse
import os
import pickle
import optuna
import multiprocessing
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.objective_functions import objective_poisson, objective_rf, objective_zip
from optimization.config import N_TRIALS_DEFAULT, RUNS_DIR, SEED
from utils.utils import save_study_results, seed_everything, setup_logging

# Função principal de execução de um único experimento
def run_single_optimization(dataset_path, model_type, trials):
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{dataset_name}_{model_type}_{timestamp}"

    output_dir = os.path.join(RUNS_DIR, study_name)
    os.makedirs(output_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(output_dir, 'study.db')}"

    with open(dataset_path, "rb") as f:
        X_train, y_train, X_val, y_val, _, _ = pickle.load(f)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name)

    if model_type == "poisson":
        study.optimize(lambda trial: objective_poisson(trial, X_train, y_train, X_val, y_val), n_trials=trials)
    elif model_type == "zip":
        study.optimize(lambda trial: objective_zip(trial, X_train, y_train, X_val, y_val), n_trials=trials)
    elif model_type == "rf":
        study.optimize(lambda trial: objective_rf(trial, X_train, y_train, X_val, y_val), n_trials=trials)        

    save_study_results(study, output_dir)

# Suporte a execução paralela
def worker(job):
    run_single_optimization(*job)

if __name__ == "__main__":
    setup_logging()
    seed_everything(SEED)

    parser = argparse.ArgumentParser(description="Pipeline Optuna Profissional")
    parser.add_argument("--datasets", nargs="+", required=True, help="Lista de arquivos pickle")
    parser.add_argument("--models", nargs="+", choices=["poisson", "zip"], required=True)
    parser.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT, help="Número de trials")
    parser.add_argument("--parallel", action="store_true", help="Ativa execução paralela")
    args = parser.parse_args()

    jobs = []
    for dataset_path in args.datasets:
        for model_type in args.models:
            jobs.append((dataset_path, model_type, args.trials))

    if args.parallel:
        with multiprocessing.Pool() as pool:
            pool.map(worker, jobs)
    else:
        for job in jobs:
            worker(job)