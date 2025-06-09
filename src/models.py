from xgboost import XGBRegressor, XGBClassifier, callback
from sklearn.ensemble import RandomForestRegressor

def get_xgb_poisson(seed=0):
    return XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=300,
        max_depth=10,
        random_state=seed,
        n_jobs=8,
        callbacks=[callback.EarlyStopping(rounds=20)],
        verbosity=1
    )

def get_xgb_classifier(seed=0):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=8,
        callbacks=[callback.EarlyStopping(rounds=20)]
    )

def get_random_forest(seed=0):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=seed,
        n_jobs=8,
        verbose=1
    )