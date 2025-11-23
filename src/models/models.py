# models/models.py
from xgboost import XGBClassifier, XGBRegressor, callback
from sklearn.ensemble import RandomForestRegressor

def get_xgb_poisson(seed=987, hyperparams=None, n_jobs=8, early_stopping_rounds=5):
    base = dict(
        n_estimators=1500,
        max_depth=10,
        learning_rate=0.1,          # add explicit default if you want
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=1,
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        callbacks=[callback.EarlyStopping(rounds=early_stopping_rounds)]
    )
    if hyperparams:
        base.update(hyperparams)
    return XGBRegressor(**base)

def get_xgb_clf(seed=987, hyperparams=None, n_jobs=8, early_stopping_rounds=5):
    base = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1500,
        max_depth=10,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        callbacks=[callback.EarlyStopping(rounds=early_stopping_rounds)]
    )
    if hyperparams:
        base.update(hyperparams)
    return XGBClassifier(**base)

def get_rf(seed=987, hyperparams=None, n_jobs=8):
    base = dict(
        criterion="poisson",
        n_estimators=1500,
        max_depth=10,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=True,
        max_features="sqrt", 
        bootstrap=True, 
    )
    if hyperparams:
        base.update(hyperparams)
    return RandomForestRegressor(**base)
