# src/optimization/adaptive_spaces.py

# Espaços iniciais de busca (fixos para cada modelo)
# Formatados como dicionário {param_name: (lower, upper)}

# Espaço inicial Poisson puro
initial_poisson_space = {
    "n_estimators": (100, 1000),
    "learning_rate": (0.01, 0.2),
    "max_depth": (3, 50),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (1e-8, 10.0),
    "reg_lambda": (1e-8, 10.0),
    "model_seed": (1, 10**6)
}

# Espaço inicial ZIP
initial_zip_space = {
    "clf_n_estimators": (50, 1000),
    "clf_learning_rate": (0.01, 0.2),
    "clf_max_depth": (3, 50),
    "clf_subsample": (0.5, 1.0),
    "clf_colsample": (0.5, 1.0),
    "clf_model_seed": (1, 10**6),
    "reg_n_estimators": (50, 1000),
    "reg_learning_rate": (0.01, 0.2),
    "reg_max_depth": (3, 50),
    "reg_subsample": (0.5, 1.0),
    "reg_colsample": (0.5, 1.0),
    "reg_alpha": (1e-8, 10.0),
    "reg_lambda": (1e-8, 10.0),
    "reg_model_seed": (1, 10**6),
    "threshold": (0.01, 0.5)
}

# Espaço inicial RandomForest
initial_rf_space = {
    "n_estimators": (100, 500),
    "max_depth": (3, 30),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 10),
    "max_features": (0, 2),  # codificado para categorical index
    "bootstrap": (0, 1),     # codificado para categorical index
    "model_seed": (1, 10**6)
}

# Mapping para decodificação dos categóricos (interno)
rf_categorical_mappings = {
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

# Função de sugestão dinâmica de hiperparâmetros adaptada para o novo espaço refinável
def suggest_poisson(trial, space):
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "subsample": trial.suggest_float("subsample", *space["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
        "model_seed": trial.suggest_int("model_seed", *space["model_seed"])
    }

def suggest_zip(trial, space):
    return {
        "clf_n_estimators": trial.suggest_int("clf_n_estimators", *space["clf_n_estimators"]),
        "clf_learning_rate": trial.suggest_float("clf_learning_rate", *space["clf_learning_rate"], log=True),
        "clf_max_depth": trial.suggest_int("clf_max_depth", *space["clf_max_depth"]),
        "clf_subsample": trial.suggest_float("clf_subsample", *space["clf_subsample"]),
        "clf_colsample": trial.suggest_float("clf_colsample", *space["clf_colsample"]),
        "clf_model_seed": trial.suggest_int("clf_model_seed", *space["clf_model_seed"]),
        "reg_n_estimators": trial.suggest_int("reg_n_estimators", *space["reg_n_estimators"]),
        "reg_learning_rate": trial.suggest_float("reg_learning_rate", *space["reg_learning_rate"], log=True),
        "reg_max_depth": trial.suggest_int("reg_max_depth", *space["reg_max_depth"]),
        "reg_subsample": trial.suggest_float("reg_subsample", *space["reg_subsample"]),
        "reg_colsample": trial.suggest_float("reg_colsample", *space["reg_colsample"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
        "reg_model_seed": trial.suggest_int("reg_model_seed", *space["reg_model_seed"]),
        "threshold": trial.suggest_float("threshold", *space["threshold"])
    }

# Sugestão dinâmica para RF
def suggest_rf(trial, space):
    return {
        "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
        "min_samples_split": trial.suggest_int("min_samples_split", *space["min_samples_split"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", *space["min_samples_leaf"]),
        "max_features": rf_categorical_mappings["max_features"][
            trial.suggest_int("max_features", *space["max_features"])
        ],
        "bootstrap": rf_categorical_mappings["bootstrap"][
            trial.suggest_int("bootstrap", *space["bootstrap"])
        ],
        "model_seed": trial.suggest_int("model_seed", *space["model_seed"])
    }