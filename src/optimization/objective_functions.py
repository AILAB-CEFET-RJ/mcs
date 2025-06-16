# src/models/objective_functions.py

import numpy as np
from xgboost import XGBRegressor, XGBClassifier, callback
from sklearn.ensemble import RandomForestRegressor
from evaluation.eval_utils import get_training_metrics, optimize_threshold

# ---------- Poisson puro ----------
def objective_poisson(params, X_train, y_train, X_val, y_val, n_jobs, early_stopping):
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
        n_jobs=n_jobs,
        verbosity=0,
        callbacks=[callback.EarlyStopping(rounds=early_stopping)]
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)

    return get_training_metrics(y_val, y_pred)

# ---------- Random Forest ----------
def objective_rf(params, X_train, y_train, X_val, y_val, n_jobs, early_stopping):
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
        random_state=params["model_seed"],
        n_jobs=n_jobs
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred = np.round(np.maximum(y_pred, 0)).astype(int)

    return get_training_metrics(y_val, y_pred)

# ---------- ZIP ----------
def objective_zip(params, X_train, y_train, X_val, y_val, n_jobs, early_stopping):
    mask_train = y_train > 0
    mask_val = y_val > 0

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=params["clf_n_estimators"],
        learning_rate=params["clf_learning_rate"],
        max_depth=params["clf_max_depth"],
        subsample=params["clf_subsample"],
        colsample_bytree=params["clf_colsample"],
        random_state=params["clf_model_seed"],
        n_jobs=n_jobs,
        verbosity=0,
        callbacks=[callback.EarlyStopping(rounds=early_stopping)]
    )

    clf.fit(X_train, (y_train > 0).astype(int), eval_set=[(X_val, (y_val > 0).astype(int))], verbose=False)

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
        n_jobs=n_jobs,
        verbosity=0,
        callbacks=[callback.EarlyStopping(rounds=early_stopping)]
    )

    reg.fit(X_train[mask_train], y_train[mask_train], eval_set=[(X_val[mask_val], y_val[mask_val])], verbose=False)

    prob_val = clf.predict_proba(X_val)[:, 1]
    y_pred_reg = reg.predict(X_val)
    threshold = optimize_threshold(prob_val, y_pred_reg, y_val)

    y_pred = y_pred_reg * (prob_val > threshold)
    y_pred = np.round(np.maximum(y_pred, 0)).astype(int)

    return get_training_metrics(y_val, y_pred)
