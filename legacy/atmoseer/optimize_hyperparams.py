import optuna
import os
import pickle
import numpy as np
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from sklearn.metrics import mean_squared_error
from optuna.visualization import plot_optimization_history, plot_param_importances
import pandas as pd

# === Caminhos ===
dataset_path = "data/datasets/RJ_WEEKLY.pickle"
study_storage = "sqlite:///optuna_poisson.db"
study_name = "xgb_poisson_opt"

# === Carregamento dos dados ===
with open(dataset_path, "rb") as file:
    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(file)

X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

# === Função objetivo ===
def objective(trial):
    params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "n_estimators": trial.suggest_int("learning_rate", 100, 1000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_jobs": 8,
        "random_state": 42,
        "verbosity": 0,
    }

    model = XGBRegressor(**params, callbacks=[EarlyStopping(rounds=20)])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

# === Criar ou carregar estudo ===
study = optuna.create_study(direction="minimize", study_name=study_name, storage=study_storage, load_if_exists=True)
study.optimize(objective, n_trials=50)

# === Exibir melhores parâmetros ===
print("✅ Best trial:")
print(study.best_trial.params)

# === Salvar melhores parâmetros ===
best_params_path = "test/best_params_poisson.json"
os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
pd.Series(study.best_trial.params).to_json(best_params_path, indent=4)