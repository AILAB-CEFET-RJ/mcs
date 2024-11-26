import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import argparse

# Abstract base model class using the Template Method pattern
class BaseModel:
    def __init__(self, name, scaler=MinMaxScaler(), batch_size=32, epochs=100):
        self.name = name
        self.scaler = scaler
        self.batch_size = batch_size
        self.epochs = epochs

    def prepare_data(self, X_train, X_val, X_test):
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_data(X_train, X_val, X_test)
        return self.reshape_data(X_train_scaled, X_val_scaled, X_test_scaled)

    def scale_data(self, X_train, X_val, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def reshape_data(self, X_train_scaled, X_val_scaled, X_test_scaled):
        return X_train_scaled, X_val_scaled, X_test_scaled  # Default behavior: no reshaping

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, output_path, unit_id, feature_set_name):
        # Step 1: Prepare data
        X_train_prepared, X_val_prepared, X_test_prepared = self.prepare_data(X_train, X_val, X_test)

        # Step 2: Train model
        history = self.train(X_train_prepared, y_train, X_val_prepared, y_val)

        # Step 3: Predict and evaluate
        y_pred = self.predict(X_test_prepared)
        y_pred_rounded = np.round(np.maximum(y_pred, 0)).astype(int)  # Ensure results are positive or zero

        # Debugging: Check shapes before computing loss
        logging.debug(f"Shape of y_test: {y_test.shape}, Shape of y_pred_rounded: {y_pred_rounded.shape}")

        # Make sure the shapes match
        if y_pred_rounded.shape != y_test.shape:
            raise ValueError(f"Shape mismatch: y_test shape {y_test.shape}, y_pred_rounded shape {y_pred_rounded.shape}")

        loss = mean_squared_error(y_test, y_pred_rounded)

        # Step 4: Log and return results
        return self.log_results(y_test, y_pred_rounded, loss, history, output_path, unit_id, feature_set_name)

    def train(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError("Must be implemented in subclass")

    def predict(self, X_test):
        raise NotImplementedError("Must be implemented in subclass")

    def log_results(self, y_test, y_pred_rounded, loss, history, output_path, unit_id, feature_set_name):
        logging.info(f'{self.name} with {feature_set_name} Test MSE: {loss}')
        return y_pred_rounded, loss, history

# LSTM model class (Keras version)
class LSTM_KerasModel(BaseModel):
    def __init__(self, input_shape, name="LSTM_KerasModel"):
        super().__init__(name)
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def reshape_data(self, X_train_scaled, X_val_scaled, X_test_scaled, timesteps=1):
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, X_train_scaled.shape[1]))
        X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], timesteps, X_val_scaled.shape[1]))
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, X_test_scaled.shape[1]))
        return X_train_reshaped, X_val_reshaped, X_test_reshaped

    def train(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val))
        return history

    def predict(self, X_test):
        return self.model.predict(X_test)

    def log_results(self, y_test, y_pred_rounded, loss, history, output_path, unit_id, feature_set_name):
        super().log_results(y_test, y_pred_rounded, loss, history, output_path, unit_id, feature_set_name)

        # Plot train vs validation loss
        plt.figure(figsize=(14, 7))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.name} with {feature_set_name} Train vs Validation Loss for {unit_id}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_path, f'{self.name}_{feature_set_name}_train_val_loss_{unit_id}.png'))
        plt.close()

        # Plot predictions vs actuals
        plt.figure(figsize=(14, 7))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred_rounded, label='Predicted')
        plt.title(f'{self.name} with {feature_set_name} Predictions vs Actuals for {unit_id}')
        plt.xlabel('Time')
        plt.ylabel('Cases')
        plt.legend()
        plt.savefig(os.path.join(output_path, f'{self.name}_{feature_set_name}_predictions_{unit_id}.png'))
        plt.close()

        return y_pred_rounded, loss, history

# XGBoost model class
class XGBoostModel(BaseModel):
    def __init__(self, name="XGBoost_Model"):
        super().__init__(name)
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=128, learning_rate=0.1)

    def train(self, X_train, y_train, X_val, y_val):
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.set_params(early_stopping_rounds=10)  # Set early stopping here
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        return self.model.evals_result()

    def predict(self, X_test):
        return self.model.predict(X_test)

    def log_results(self, y_test, y_pred_rounded, loss, history, output_path, unit_id, feature_set_name):
        super().log_results(y_test, y_pred_rounded, loss, history, output_path, unit_id, feature_set_name)

        # Plot train vs validation RMSE
        plt.figure(figsize=(14, 7))
        plt.plot(history['validation_0']['rmse'], label='Train RMSE')
        plt.plot(history['validation_1']['rmse'], label='Validation RMSE')
        plt.title(f'{self.name} with {feature_set_name} Train vs Validation RMSE for {unit_id}')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(os.path.join(output_path, f'{self.name}_{feature_set_name}_train_val_rmse_{unit_id}.png'))
        plt.close()

        # Plot predictions vs actuals
        plt.figure(figsize=(14, 7))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred_rounded, label='Predicted')
        plt.title(f'{self.name} with {feature_set_name} Predictions vs Actuals for {unit_id}')
        plt.xlabel('Time')
        plt.ylabel('Cases')
        plt.legend()
        plt.savefig(os.path.join(output_path, f'{self.name}_{feature_set_name}_predictions_{unit_id}.png'))
        plt.close()

        return y_pred_rounded, loss, history

# Training function
def train(dataset_path, output_path, train_cutoff, val_cutoff, model_type, id_unidade):
    # Load the dataset
    df = pd.read_parquet(dataset_path)

    # Filter by ID_UNIDADE if provided
    if id_unidade:
        df = df[df["ID_UNIDADE"] == id_unidade]

    # Define features and target
    features_cases = ['CASES', 'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21']
    features_inmet = ['IDEAL_TEMP_INMET', 'EXTREME_TEMP_INMET', 'SIGNIFICANT_RAIN_INMET', 'EXTREME_RAIN_INMET', 'TEMP_RANGE_INMET',
                      'TEM_AVG_INMET_MM_7', 'TEM_AVG_INMET_MM_14', 'TEM_AVG_INMET_MM_21',
                      'CHUVA_INMET_MM_7', 'CHUVA_INMET_MM_14', 'CHUVA_INMET_MM_21',
                      'TEMP_RANGE_INMET_MM_7', 'TEMP_RANGE_INMET_MM_14', 'TEMP_RANGE_INMET_MM_21',
                      'TEM_AVG_INMET_ACC_7', 'TEM_AVG_INMET_ACC_14', 'TEM_AVG_INMET_ACC_21',
                      'CHUVA_INMET_ACC_7', 'CHUVA_INMET_ACC_14', 'CHUVA_INMET_ACC_21']

    features_sat = ['IDEAL_TEMP_SAT', 'EXTREME_TEMP_SAT', 'SIGNIFICANT_RAIN_SAT', 'EXTREME_RAIN_SAT', 'TEMP_RANGE_SAT',
                    'TEM_AVG_SAT_MM_7', 'TEM_AVG_SAT_MM_14', 'TEM_AVG_SAT_MM_21',
                    'CHUVA_SAT_MM_7', 'CHUVA_SAT_MM_14', 'CHUVA_SAT_MM_21',
                    'TEMP_RANGE_SAT_MM_7', 'TEMP_RANGE_SAT_MM_14', 'TEMP_RANGE_SAT_MM_21',
                    'TEM_AVG_SAT_ACC_7', 'TEM_AVG_SAT_ACC_14', 'TEM_AVG_SAT_ACC_21',
                    'CHUVA_SAT_ACC_7', 'CHUVA_SAT_ACC_14', 'CHUVA_SAT_ACC_21']

    all_features = features_cases + features_inmet + features_sat
    inmet_and_cases = features_cases + features_inmet
    sat_and_cases = features_cases + features_sat

    # Define a collection of feature sets
    training_features = {
        'all_features': all_features,
        'inmet_and_cases': inmet_and_cases,
        'sat_and_cases': sat_and_cases,
        'cases' : features_cases
    }

    target = 'CASES'

    # Remove NaNs
    df = df.dropna()

    # Group by ID_UNIDADE
    grouped = df.groupby('ID_UNIDADE')

    # DataFrame to store performance metrics
    performance_metrics = pd.DataFrame(columns=['ID_UNIDADE', 'Model', 'Feature_Set', 'MSE', 'Total_Cases_Train', 'Estimated_Cases_Test', 'Actual_Cases_Test'])

    # Select model based on the model_type argument
    if model_type == "LSTM_KERAS":
        model_class = LSTM_KerasModel
    elif model_type == "XGBoost":
        model_class = XGBoostModel
    else:
        raise ValueError(f"Unknown model type: {model_type}. Please choose either 'LSTM_KERAS' or 'XGBoost'.")

    # Iterate over each group of ID_UNIDADE
    for name, group in grouped:
        logging.info(f'Training {model_type} model for ID_UNIDADE: {name}')
        group = group.sort_values(by='DT_NOTIFIC')

        # Split data into train, validation, and test sets
        train_df = group[group['DT_NOTIFIC'] <= train_cutoff]
        val_df = group[(group['DT_NOTIFIC'] > train_cutoff) & (group['DT_NOTIFIC'] <= val_cutoff)]
        test_df = group[group['DT_NOTIFIC'] > val_cutoff]

        # Train and evaluate the model for each feature set
        for feature_set_name, feature_set in training_features.items():
            X_train = train_df[feature_set].values
            y_train = train_df[target].values
            X_val = val_df[feature_set].values
            y_val = val_df[target].values
            X_test = test_df[feature_set].values
            y_test = test_df[target].values

            # Instantiate the model (for LSTM_KERAS and LSTM_PYTORCH we need to specify input shape/size)
            if model_type == "LSTM_KERAS":
                model = model_class(input_shape=(1, len(feature_set)))
            elif model_type == "XGBoost":
                model = model_class()

            # Train and evaluate the model
            y_pred_rounded, loss, history = model.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, output_path, name, feature_set_name)

            # Store performance metrics
            total_cases_train = y_train.sum()
            estimated_cases_test = y_pred_rounded.sum()
            actual_cases_test = y_test.sum()
            performance_metrics.loc[len(performance_metrics)] = [name, model.name, feature_set_name, loss, total_cases_train, estimated_cases_test, actual_cases_test]

    # Save performance metrics to a CSV file
    performance_metrics.to_csv(os.path.join(output_path, 'performance_metrics.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description="Train models for dengue case prediction")
    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument("output_path", help="Path to the output")
    parser.add_argument("train_cutoff", help="Date to split train/validation (YYYY-MM-DD)")
    parser.add_argument("val_cutoff", help="Date to split validation/test (YYYY-MM-DD)")
    parser.add_argument("model_type", choices=["LSTM_KERAS", "XGBoost"], help="Type of model to train ('LSTM_KERAS' or 'XGBoost')")
    parser.add_argument("--id_unidade", dest="id_unidade", default=None, help="Filter by ID_UNIDADE")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    train(args.dataset_path, args.output_path, args.train_cutoff, args.val_cutoff, args.model_type, args.id_unidade)

if __name__ == '__main__':
    #main()
    train(
        dataset_path="data/processed/sinan/sinan.parquet",
        output_path="data/processed/lstm",
        train_cutoff="2021-12-31",
        val_cutoff="2022-12-31",
        model_type="XGBoost",
        id_unidade=None
    )
