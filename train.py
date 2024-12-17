import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import argparse


# Dataset Class for PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ensure y is 2D

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# LSTM Model Class (BaseModel Integrated)
class LSTM_PyTorchModel:
    def __init__(self, input_size, hidden_size=50, batch_size=32, epochs=100, name="LSTM_PyTorchModel"):
        self.name = name
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler = MinMaxScaler()

        self.model = torch.nn.Sequential(
            torch.nn.LSTM(input_size, hidden_size, batch_first=True),
            torch.nn.Linear(hidden_size, 1)
        )

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def prepare_data(self, X_train, X_val, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        timesteps = 1
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, X_train_scaled.shape[1]))
        X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], timesteps, X_val_scaled.shape[1]))
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, X_test_scaled.shape[1]))

        return X_train_reshaped, X_val_reshaped, X_test_reshaped

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, output_path, unit_id):
        X_train, X_val, X_test = self.prepare_data(X_train, X_val, X_test)

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)

        history = {"loss": [], "val_loss": []}
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs, _ = self.model[0](batch_X)  # LSTM layer
                outputs = self.model[1](outputs[:, -1, :])  # Linear layer
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            history["loss"].append(epoch_loss / len(train_loader.dataset))

            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs, _ = self.model[0](batch_X)
                    outputs = self.model[1](outputs[:, -1, :])
                    val_loss += self.criterion(outputs, batch_y).item() * batch_X.size(0)
            history["val_loss"].append(val_loss / len(val_loader.dataset))

            logging.info(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {history["loss"][-1]}, Val Loss: {history["val_loss"][-1]}')

        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"LSTM Model Test MSE: {mse}")

        return mse, history

    def predict(self, X_test):
        test_loader = DataLoader(TimeSeriesDataset(X_test, np.zeros((X_test.shape[0], 1))), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs, _ = self.model[0](batch_X)
                outputs = self.model[1](outputs[:, -1, :])
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions).flatten()


# Training Function
def train(dataset_path, output_path, train_cutoff, val_cutoff, id_unidade=None):
    df = pd.read_parquet(dataset_path)
    if id_unidade:
        df = df[df["ID_UNIDADE"] == id_unidade]

    features = ['CASES', 'CASES_MM_14']
    target = 'CASES'
    df = df.dropna()

    grouped = df.groupby('ID_UNIDADE')
    for name, group in grouped:
        group = group.sort_values(by='DT_NOTIFIC')
        train_df = group[group['DT_NOTIFIC'] <= train_cutoff]
        val_df = group[(group['DT_NOTIFIC'] > train_cutoff) & (group['DT_NOTIFIC'] <= val_cutoff)]
        test_df = group[group['DT_NOTIFIC'] > val_cutoff]

        X_train, y_train = train_df[features].values, train_df[target].values
        X_val, y_val = val_df[features].values, val_df[target].values
        X_test, y_test = test_df[features].values, test_df[target].values

        model = LSTM_PyTorchModel(input_size=len(features))
        mse, history = model.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, output_path, name)

        logging.info(f"Finished training for unit {name} with Test MSE: {mse}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train(
        dataset_path="data/processed/sinan/sinan.parquet",
        output_path="data/processed/lstm",
        train_cutoff="2021-12-31",
        val_cutoff="2022-12-31",
        id_unidade=None
    )
