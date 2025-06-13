# src/optimization/adaptive_optimize_pipeline.py

import sys
import os

from data.data_utils import load_data

# Ajuste de sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pickle
import optuna
from datetime import datetime

from models.objective_functions import objective_poisson, objective_rf, objective_zip
from optimization.adaptive_spaces import suggest_poisson, suggest_rf, suggest_zip
from optimization.space_refiner import refine_space
from utils.utils import setup_logging, seed_everything, save_study_results
from optimization.config import SEED, RUNS_DIR

# Rodada única de otimização
def run_single_round(dataset_path, model_type, trials, round_id, search_space):

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{dataset_name}_{model_type}_R{round_id}_{timestamp}"

    output_dir = os.path.join(RUNS_DIR, study_name)
    os.makedirs(output_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(output_dir, 'study.db')}"

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_path)

    study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name)

    # Dynamic objective com o espaço externo controlável
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

# Pipeline completo de várias rodadas adaptativas
def run_adaptive_pipeline(dataset_path, model_type, initial_space, trials_per_round, total_rounds):
    
    current_space = initial_space

    for round_id in range(1, total_rounds + 1):
        print(f"\n🔄 Iniciando rodada {round_id}/{total_rounds}")
        study = run_single_round(dataset_path, model_type, trials_per_round, round_id, current_space)

        # Refinar o espaço automaticamente:
        current_space = refine_space(study, current_space)
        print(f"🔧 Novo espaço de busca gerado para próxima rodada: {current_space}")

if __name__ == "__main__":
    setup_logging()
    seed_everything(SEED)

    parser = argparse.ArgumentParser(description="Pipeline Adaptativo com Refinamento Automático")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["poisson", "zip"], required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)

    args = parser.parse_args()

    # Definir espaços iniciais (poderia vir de arquivo de config depois)
    from optimization.adaptive_spaces import initial_poisson_space, initial_zip_space, initial_rf_space

    if args.model == "poisson":
        initial_space = initial_poisson_space
    elif args.model == "zip":
        initial_space = initial_zip_space
    elif args.model == "rf":
        initial_space = initial_rf_space        
        

    run_adaptive_pipeline(args.dataset, args.model, initial_space, args.trials, args.rounds)
