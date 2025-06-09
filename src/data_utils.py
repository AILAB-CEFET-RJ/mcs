import pickle

def load_data_for_xgb(file_path: str):
    with open(file_path, "rb") as file:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    