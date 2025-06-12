# === models.py ===

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor, callback
from sklearn.metrics import mean_squared_error
from config import N_JOBS_XGB, SEED, EARLY_STOPPING_ROUNDS


def objective_poisson(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": SEED,
        "n_jobs": N_JOBS_XGB,
        "verbosity": 0,
    }

    model = XGBRegressor(**params, callbacks=[callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)


def objective_zip(trial, X_train, y_train, X_val, y_val):
    mask_train = y_train > 0
    mask_val = y_val > 0

    # Hyperparams do classificador
    clf_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": trial.suggest_int("clf_n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("clf_learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("clf_max_depth", 3, 10),
        "subsample": trial.suggest_float("clf_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("clf_colsample", 0.5, 1.0),
        "random_state": SEED,
        "n_jobs": N_JOBS_XGB,
        "verbosity": 0
    }

    # Hyperparams do regressor
    reg_params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "n_estimators": trial.suggest_int("reg_n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("reg_learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("reg_max_depth", 3, 10),
        "subsample": trial.suggest_float("reg_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("reg_colsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": SEED,
        "n_jobs": N_JOBS_XGB,
        "verbosity": 0
    }

    threshold = trial.suggest_float("threshold", 0.01, 0.5)

    clf = XGBClassifier(**clf_params, callbacks=[callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)])
    clf.fit(X_train, (y_train > 0).astype(int), eval_set=[(X_val, (y_val > 0).astype(int))], verbose=False)

    reg = XGBRegressor(**reg_params, callbacks=[callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)])
    reg.fit(X_train[mask_train], y_train[mask_train], eval_set=[(X_val[mask_val], y_val[mask_val])], verbose=False)

    prob_val = clf.predict_proba(X_val)[:, 1]
    y_pred_reg = reg.predict(X_val)
    y_pred = y_pred_reg * (prob_val > threshold)
    y_pred = np.round(np.maximum(y_pred, 0)).astype(int)

    return mean_squared_error(y_val, y_pred)

def objective_rf(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": SEED,
        "n_jobs": N_JOBS_XGB  # aproveitando o mesmo parâmetro que já tens
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    preds = np.round(np.maximum(preds, 0)).astype(int)

    return mean_squared_error(y_val, preds)