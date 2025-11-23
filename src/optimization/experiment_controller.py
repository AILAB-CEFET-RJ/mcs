# src/optimization/experiment_controller.py

import os
import pickle
import optuna
import numpy as np
import logging
from datetime import datetime

from optimization.experiment_config_parser import ExperimentConfig
from optimization import adaptive_spaces, objective_functions, space_refiner
from optimization.pick_best_trial_weighted import pick_best_trial_weighted


class ExperimentController:

    def __init__(self, config_path):
        self.config = ExperimentConfig(config_path)

    def load_data(self, dataset_path):
        with open(dataset_path, "rb") as f:
            X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val   = X_val.reshape(X_val.shape[0], -1)
        X_test  = X_test.reshape(X_test.shape[0], -1)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _setup_logger(self, study_dir):
        logger = logging.getLogger(study_dir)
        logger.setLevel(logging.INFO)
        logger.propagate = False
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

        optuna_logger = optuna.logging.get_logger("optuna")
        optuna_logger.handlers.clear()
        optuna_logger.addHandler(fh)
        optuna_logger.addHandler(sh)
        optuna.logging.set_verbosity(optuna.logging.INFO)

        return logger

    @staticmethod
    def split_zip_params(flat_params):
        clf_params, reg_params = {}, {}

        for key, value in flat_params.items():
            if key.startswith("clf_"):
                k = key.replace("clf_", "")
                if k == "colsample":
                    k = "colsample_bytree"
                if k == "model_seed":
                    continue
                clf_params[k] = value

            elif key.startswith("reg_"):
                if key in ("reg_alpha", "reg_lambda"):
                    reg_params[key] = value
                    continue
                k = key.replace("reg_", "")
                if k == "colsample":
                    k = "colsample_bytree"
                if k == "model_seed":
                    continue
                reg_params[k] = value

        return {"clf": clf_params, "reg": reg_params}

    def run(self):
        for dataset_path, model_type, name, train_config in self.config.list_experiments():
            self.run_single_experiment(dataset_path, model_type, name, train_config)

    def run_single_experiment(self, dataset_path, model_type, name, train_config):
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data(dataset_path)
        # MSE   RMSE  MAE   -R²   MAPE  SMAPE  -Rho  PoisDev
        weights = np.array([0.75, 0.75, 1.5, 1.25, 0.25, 1.0, 1.25, 2.25], dtype=float)
        weights = weights / weights.sum()
        
        space = getattr(adaptive_spaces, f"initial_{model_type}_space")
        suggest_fn = getattr(adaptive_spaces, f"suggest_{model_type}")
        objective_fn = getattr(objective_functions, f"objective_{model_type}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = os.path.join(self.config.runs_dir, f"{os.path.basename(name)}_{model_type}_{timestamp}")
        os.makedirs(study_dir, exist_ok=True)

        logger = self._setup_logger(study_dir)
        logger.info(f"🚀 Iniciando experimento: Dataset={dataset_path} | Modelo={model_type} | Name={name}")
        logger.info(f"📁 study_dir: {study_dir}")
        logger.info(f"📄 train_config: {train_config}")

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
                space = space_refiner.refine_space(
                    study,
                    space,
                    weights=weights,
                    norm="minmax",
                    top_frac=0.2
                )

        # ---- Elect best trial (weighted) ----
        best_trial, best_score, _ = pick_best_trial_weighted(study, weights=weights, norm="minmax")

        logger.info(f"🏆 Elected best trial #{best_trial.number} | score={best_score:.6f}")
        logger.info(f"📌 values={best_trial.values}")
        logger.info(f"📌 raw params={best_trial.params}")

        # ---- Final robust training ----
        from train_pipeline import run_training

        seeds = [109, 220, 222, 241, 149, 107, 75, 248, 254, 140]

        hyperparams_by_model = {}

        if model_type == "poisson":
            best_params = {k:v for k,v in best_trial.params.items() if k != "model_seed"}
            hyperparams_by_model["xgb_poisson"] = best_params

        elif model_type == "rf":
            best_params = {k:v for k,v in best_trial.params.items() if k != "model_seed"}
            hyperparams_by_model["random_forest"] = best_params

        elif model_type == "zip":
            zip_hp = self.split_zip_params(best_trial.params)
            hyperparams_by_model["xgb_zip"] = zip_hp
            logger.info(f"📌 ZIP clf params={zip_hp['clf']}")
            logger.info(f"📌 ZIP reg params={zip_hp['reg']}")

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        run_training(
            config_path=train_config,
            hyperparams_by_model=hyperparams_by_model,
            seeds=seeds
        )

        logger.info("✅ Final robust training completed.")
