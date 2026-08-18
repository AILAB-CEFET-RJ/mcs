import pickle
from pathlib import Path

def load_data(file_path: str):
    with open(file_path, "rb") as file:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        return X_train, y_train, X_val, y_val, X_test, y_test


def load_selection_data(dataset_path: str):
    """Load only train and validation artifacts; never deserialize confirmation."""
    source = Path(dataset_path)
    root = source.parent if source.is_file() else source
    split_dir = root / "splits"
    train_path = split_dir / "train.pickle"
    validation_path = split_dir / "validation.pickle"
    if not train_path.exists() or not validation_path.exists():
        raise RuntimeError(
            "Dataset sem splits físicos protegidos. Reconstrua o dataset V2; "
            "selection não pode abrir o dataset.pickle monolítico."
        )
    with train_path.open("rb") as handle:
        X_train, y_train = pickle.load(handle)
    with validation_path.open("rb") as handle:
        X_val, y_val = pickle.load(handle)
    return (X_train.reshape(X_train.shape[0], -1), y_train,
            X_val.reshape(X_val.shape[0], -1), y_val)
    
