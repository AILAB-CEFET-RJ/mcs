from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import yaml
from train.training_utils import *
from train.evaluate import *

class Regressor(nn.Module):
    def __init__(self, learner, in_channels, y_mean_value):
        super(Regressor, self).__init__()
        self.learner = learner

        self.conv1d_1 = nn.Conv1d(
            in_channels=in_channels, out_channels=32, kernel_size=3, padding=2)
        self.gn_1 = nn.GroupNorm(1, 32)

        self.conv1d_2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.gn_2 = nn.GroupNorm(1, 64)

        self.conv1d_3 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.gn_3 = nn.GroupNorm(1, 64)

        self.conv1d_4 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.gn_4 = nn.GroupNorm(1, 128)

        # self.conv1d_5 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3, padding=2)
        # self.conv1d_6 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=2)
        # self.conv1d_7 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, padding=2)
        # self.conv1d_8 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 3, padding=2)

        self.max_pooling1d_1 = nn.MaxPool1d(2)
        # self.max_pooling1d_2 = nn.MaxPool1d(2)

        # self.relu = nn.ReLU()
        self.relu = nn.GELU()

        self.fc1 = nn.Linear(1280, 50)

        self.fc2 = nn.Linear(50, 1)
        self.fc2.bias.data.fill_(y_mean_value)

        # self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.gn_1(x)
        x = self.relu(x)

        # print('conv1d_1')

        x = self.max_pooling1d_1(x)

        x = self.conv1d_2(x)
        x = self.gn_2(x)
        x = self.relu(x)

        # print('conv1d_2')

        x = self.conv1d_3(x)
        x = self.gn_3(x)
        x = self.relu(x)

        # print('conv1d_3')

        x = self.conv1d_4(x)
        x = self.gn_4(x)
        x = self.relu(x)

        # print('conv1d_4')

        # x = self.conv1d_5(x)
        # x = self.relu(x)

        # # print('conv1d_5')

        # x = self.max_pooling1d_1(x)

        # x = self.conv1d_6(x)
        # x = self.relu(x)

        # x = self.conv1d_7(x)
        # x = self.relu(x)

        # x = self.conv1d_8(x)
        # x = self.relu(x)

        # # print('conv1d_8')

        x = x.view(x.shape[0], -1)
        # x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        return x

    def validation_step(self, batch):
        X_train, y_train = batch
        out = self(X_train)                    # Generate predictions
        loss = F.cross_entropy(out, y_train)   # Calculate loss
        acc = accuracy(out, y_train)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def evaluate(self, test_loader):
        """Evaluates the model using a DataLoader and returns predictions."""
        self.eval()  
        self.learner.eval()

        y_true_list, y_pred_list = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.float(), yb.float()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                xb, yb = xb.to(device), yb.to(device)

                # Debugging: Print shape before passing to model
                print(f"Before model: xb.shape = {xb.shape}")  # Should be (batch_size, seq_len, features)

                # If using LSTM, do NOT permute!
                output = self.learner(xb)  # Forward pass

                # Store results
                y_pred_list.append(output.cpu().numpy())  
                y_true_list.append(yb.cpu().numpy())  

        return np.vstack(y_true_list).reshape(-1, 1), np.vstack(y_pred_list).reshape(-1, 1)



    def print_evaluation_report(self, pipeline_id, test_loader, forecasting_task):
            print("\\begin{verbatim}")
            print(f"***Evaluation report for pipeline {pipeline_id}***")
            print("\\end{verbatim}")

            print("\\begin{verbatim}")
            print("***Hyperparameters***")
            with open('./config/config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            model_config = config['training']['oc']
            pretty_model_config = yaml.dump(model_config, indent=4)
            print(pretty_model_config)
            print("\\end{verbatim}")

            print("\\begin{verbatim}")
            print("***Model architecture***")
            print(self.learner)
            print("\\end{verbatim}")

            print("\\begin{verbatim}")
            print('***Confusion matrix***')
            print("\\end{verbatim}")
            y_true, y_pred = self.evaluate(test_loader)
            assert(y_true.shape == y_pred.shape)
            export_results_to_latex(y_true, y_pred, forecasting_task)

            print("\\begin{verbatim}")
            print('***Classification report***')
            print(mean_squared_error(y_true, y_pred))
            print(mean_absolute_error(y_true, y_pred))
            print("\\end{verbatim}")        