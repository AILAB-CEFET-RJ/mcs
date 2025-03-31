import pickle
import logging
import globals as globals
import pandas as pd

def load_datasets(pipeline_id: str, dump_csv: bool = False, output_dir: str = "./data/dump/"):
    '''
    Load numpy arrays (stored in a pickle file) from disk
    '''
    filename = globals.DATASETS_DIR + pipeline_id + ".pickle"
    logging.info(f"Loading train/val/test datasets from {filename}.")
    file = open(filename, 'rb')
    (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)
    logging.info(f"Shapes of train/val/test data matrices: {X_train.shape}/{X_val.shape}/{X_test.shape}")
    logging.info(f"Min values of train/val/test target: {min(y_train)}/{min(y_val)}/{min(y_test)}")
    logging.info(f"Max values of train/val/test target: {max(y_train)}/{max(y_val)}/{max(y_test)}")
    #assert X_train.shape[1] == X_val.shape[1], f"Shape mismatch: {X_train.shape} vs {X_val.shape}"
    #assert X_val.shape[1] == X_test.shape[1], f"Shape mismatch: {X_train.shape} vs {X_val.shape}"
    
    # Dump datasets to CSV if requested
    if dump_csv:
        logging.info(f"Dumping datasets to CSV in directory: {output_dir}")
        pd.DataFrame(X_train.reshape(X_train.shape[0], -1)).to_csv(f"{output_dir}X_train.csv", index=False)
        pd.DataFrame(y_train).to_csv(f"{output_dir}y_train.csv", index=False, header=["target"])
        pd.DataFrame(X_val.reshape(X_val.shape[0], -1)).to_csv(f"{output_dir}X_val.csv", index=False)
        pd.DataFrame(y_val).to_csv(f"{output_dir}y_val.csv", index=False, header=["target"])
        pd.DataFrame(X_test.reshape(X_test.shape[0], -1)).to_csv(f"{output_dir}X_test.csv", index=False)
        pd.DataFrame(y_test).to_csv(f"{output_dir}y_test.csv", index=False, header=["target"])
        logging.info(f"Datasets dumped successfully to {output_dir}")

    return X_train, y_train, X_val, y_val, X_test, y_test
