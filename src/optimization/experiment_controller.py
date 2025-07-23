# src/optimization/experiment_controller.py

import os
import pickle
import optuna
import numpy as np
from datetime import datetime

from optimization.experiment_config_parser import ExperimentConfig
from  optimization import adaptive_spaces, objective_functions, space_refiner
from evaluation.eval_utils import get_training_metrics

class ExperimentController:

    def __init__(self, config_path):
        self.config = ExperimentConfig(config_path)

    def load_data(self, dataset_path):
        with open(dataset_path, "rb") as f:
            X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

        # Reshape (caso sliding window)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def run(self):
        for dataset_path, model_type in self.config.list_experiments():
            print(f"\n🚀 Iniciando experimento: Dataset={dataset_path} | Modelo={model_type}")
            self.run_single_experiment(dataset_path, model_type)

    def run_single_experiment(self, dataset_path, model_type):
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data(dataset_path)

        # Espaço inicial
        space = getattr(adaptive_spaces, f"initial_{model_type}_space")
        suggest_fn = getattr(adaptive_spaces, f"suggest_{model_type}")
        objective_fn = getattr(objective_functions, f"objective_{model_type}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = os.path.join(self.config.runs_dir, f"{os.path.basename(dataset_path)}_{model_type}_{timestamp}")
        os.makedirs(study_dir, exist_ok=True)

        # Execução por rounds adaptativos
        for round_idx, trials in enumerate(self.config.trials_per_round, start=1):
            print(f"\n🔄 Round {round_idx} com {trials} trials")

            study = optuna.create_study(
                directions=self.config.directions,
                study_name=f"{model_type}_round{round_idx}",
                storage=f"sqlite:///{os.path.join(study_dir, f'study_round{round_idx}.db')}",
                load_if_exists=True
            )

            def optuna_objective(trial):
                params = suggest_fn(trial, space)
                metrics = objective_fn(params, X_train, y_train, X_val, y_val, self.config.n_jobs_xgb, self.config.early_stopping_rounds)
                return list(metrics.values())

            study.optimize(optuna_objective, n_trials=trials)

            # Salva study do round
            study.trials_dataframe().to_csv(os.path.join(study_dir, f"trials_round{round_idx}.csv"), index=False)

            # Refina espaço adaptativo para próxima rodada
            if round_idx != len(self.config.trials_per_round):
                space = space_refiner.refine_space(study, space)

        # Avaliação final no teste
        best_trial = study.best_trials[0]
        best_params = suggest_fn(best_trial, space)

        model_final = model_final = objective_fn(best_params, X_train, y_train, X_test, y_test, self.config.n_jobs_xgb, self.config.early_stopping_rounds, return_model=True)

        if model_type == "zip":
            clf_model, reg_model = model_final

            prob_test = clf_model.predict_proba(X_test)[:, 1]
            y_pred_reg = reg_model.predict(X_test)

            from evaluation.eval_utils import optimize_threshold
            threshold = optimize_threshold(prob_test, y_pred_reg, y_test)

            y_pred = y_pred_reg * (prob_test > threshold)
        else:
            y_pred = model_final.predict(X_test)        
        
        y_pred = np.round(np.maximum(y_pred, 0)).astype(int)

        test_metrics = get_training_metrics(y_test, y_pred)
        print("\n📊 Test Metrics:", test_metrics)

        # Salva métricas de teste
        with open(os.path.join(study_dir, "final_test_metrics.json"), "w") as f:
            import json
            json.dump(test_metrics, f, indent=4)
