from xgboost import XGBClassifier, XGBRegressor, callback


def get_xgb_poisson(seed):
    return XGBRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=seed,
        n_jobs=8,
        verbosity=1,
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        callbacks=[callback.EarlyStopping(rounds=20)]
    )
    
def get_xgb_clf(seed):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        random_state=seed,
        max_depth=10,
        learning_rate=0.1,
        n_jobs=8,
        callbacks=[callback.EarlyStopping(rounds=20)]
    )