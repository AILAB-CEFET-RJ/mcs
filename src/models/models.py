from xgboost import XGBClassifier, XGBRegressor, callback
from sklearn.ensemble import RandomForestRegressor


def get_xgb_poisson(seed=987):
    return XGBRegressor(
        n_estimators=1500,
        max_depth=10,
        random_state=seed,
        n_jobs=8,
        verbosity=1,
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        callbacks=[callback.EarlyStopping(rounds=5)]
    )
    
def get_xgb_clf(seed=987):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1500,
        random_state=seed,
        max_depth=10,
        learning_rate=0.1,
        n_jobs=8,
        callbacks=[callback.EarlyStopping(rounds=5)]
    )
    
def get_rf(seed=987):
    return RandomForestRegressor(
        criterion="poisson",
        n_estimators=1500,
        max_depth=10,
        random_state=seed,
        n_jobs=8,
        verbose=True      
    )