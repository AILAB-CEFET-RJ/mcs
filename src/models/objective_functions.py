# src/models/objective_functions.py

import numpy as np
from xgboost import XGBRegressor, XGBClassifier, callback
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from evaluation.eval_utils import get_training_metrics
from optimization.config import N_JOBS_XGB, EARLY_STOPPING_ROUNDS

# ✅ Poisson adaptativo
def objective_poisson(trial, X_train, y_train, X_val, y_val, params):
    model = XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=params["model_seed"],
        n_jobs=N_JOBS_XGB,
        verbosity=0,
        callbacks=[callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)]
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)

    metrics = get_training_metrics(y_val, preds)
    trial.set_user_attr("metrics", metrics)

    return metrics["MSE"]

# ✅ ZIP adaptativo
def objective_zip(trial, X_train, y_train, X_val, y_val, params):
    mask_train = y_train > 0
    mask_val = y_val > 0

    # Classificador binário
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=params["clf_n_estimators"],
        learning_rate=params["clf_learning_rate"],
        max_depth=params["clf_max_depth"],
        subsample=params["clf_subsample"],
        colsample_bytree=params["clf_colsample"],
        random_state=params["clf_model_seed"],
        n_jobs=N_JOBS_XGB,
        verbosity=0,
        callbacks=[callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)]
    )

    clf.fit(X_train, (y_train > 0).astype(int), eval_set=[(X_val, (y_val > 0).astype(int))], verbose=False)

    # Regressor Poisson (apenas para y > 0)
    reg = XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=params["reg_n_estimators"],
        learning_rate=params["reg_learning_rate"],
        max_depth=params["reg_max_depth"],
        subsample=params["reg_subsample"],
        colsample_bytree=params["reg_colsample"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=params["reg_model_seed"],
        n_jobs=N_JOBS_XGB,
        verbosity=0,
        callbacks=[callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS)]
    )

    reg.fit(X_train[mask_train], y_train[mask_train], eval_set=[(X_val[mask_val], y_val[mask_val])], verbose=False)

    # ZIP final
    prob_val = clf.predict_proba(X_val)[:, 1]
    y_pred_reg = reg.predict(X_val)
    y_pred = y_pred_reg * (prob_val > params["threshold"])
    y_pred = np.round(np.maximum(y_pred, 0)).astype(int)

    metrics = get_training_metrics(y_val, y_pred)
    trial.set_user_attr("metrics", metrics)

    return metrics["MSE"]

# ✅ RandomForest adaptativo
def objective_rf(trial, X_train, y_train, X_val, y_val, params):
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
        random_state=params["model_seed"],
        n_jobs=N_JOBS_XGB
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    preds = np.round(np.maximum(preds, 0)).astype(int)

    metrics = get_training_metrics(y_val, preds)
    trial.set_user_attr("metrics", metrics)

    return metrics["MSE"]