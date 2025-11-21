# src/optimization/experiment_controller.py

import os
import pickle
import optuna
import numpy as np
import logging
from datetime import datetime

from optimization.experiment_config_parser import ExperimentConfig
from optimization import adaptive_spaces, objective_functions, space_refiner, pick_best_trial_weighted
from evaluation.eval_utils import get_training_metrics

class ExperimentController:

    def __init__(self, config_path):
        self.config = ExperimentConfig(config_path)

    def load_data(self, dataset_path):
        with open(dataset_path, "rb") as f:
            X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

        # Reshape (caso sliding window)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val   = X_val.reshape(X_val.shape[0], -1)
        X_test  = X_test.reshape(X_test.shape[0], -1)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _setup_logger(self, study_dir):
        """
        Creates a per-run logger that writes to study_dir/run.log
        and also echoes to stdout.
        """
        logger = logging.getLogger(study_dir)  # unique name per dir
        logger.setLevel(logging.INFO)
        logger.propagate = False  # avoid double logging if root is set

        # Clear handlers if rerun in same process
        if logger.handlers:
            logger.handlers.clear()

        log_path = os.path.join(study_dir, "run.log")

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)

        # Capture Optuna logs to same file
        optuna_logger = optuna.logging.get_logger("optuna")
        optuna_logger.handlers.clear()
        optuna_logger.addHandler(fh)
        optuna_logger.addHandler(sh)
        optuna.logging.set_verbosity(optuna.logging.INFO)

        return logger

    def run(self):
        for dataset_path, model_type, name in self.config.list_experiments():
            self.run_single_experiment(dataset_path, model_type, name)

    def run_single_experiment(self, dataset_path, model_type, name):
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data(dataset_path)

        # Espaço inicial
        space = getattr(adaptive_spaces, f"initial_{model_type}_space")
        suggest_fn = getattr(adaptive_spaces, f"suggest_{model_type}")
        objective_fn = getattr(objective_functions, f"objective_{model_type}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = os.path.join(self.config.runs_dir, f"{os.path.basename(name)}_{model_type}_{timestamp}")
        os.makedirs(study_dir, exist_ok=True)

        logger = self._setup_logger(study_dir)
        logger.info(f"🚀 Iniciando experimento: Dataset={dataset_path} | Modelo={model_type} | Name={name}")
        logger.info(f"📁 study_dir: {study_dir}")

        # Adaptive rounds
        for round_idx, trials in enumerate(self.config.trials_per_round, start=1):
            logger.info(f"🔄 Round {round_idx} com {trials} trials")

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
        weights = [  # example: equal weights (edit to your thesis weights)
            1, 1, 1, 1, 1, 1, 1, 1
        ]

        best_trial, best_score, row = self.pick_best_trial_weighted(
            study, weights=weights, norm="minmax"
        )

        best_params = best_trial.params
        logger.info(f"🏆 Elected best trial #{best_trial.number} | score={best_score:.6f}")
        logger.info(f"📌 values={best_trial.values}")
        logger.info(f"📌 params={best_params}")

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
        logger.info(f"📊 Test Metrics: {test_metrics}")

        with open(os.path.join(study_dir, "final_test_metrics.json"), "w") as f:
            import json
            json.dump(test_metrics, f, indent=4)
