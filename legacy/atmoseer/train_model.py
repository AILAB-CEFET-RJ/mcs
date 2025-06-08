import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time

import numpy as np
import sys
import argparse
import time
import train.pipeline as pipeline

from train.ordinal_classifier import OrdinalClassifier
from train.binary_classifier import BinaryClassifier
from train.regression_net import Regressor
from train.training_utils import DeviceDataLoader, to_device, gen_learning_curve, seed_everything
from train.conv1d_neural_net import Conv1DNeuralNet 
from train.lstm_neural_net import LstmNeuralNet
import rainfall as rp

from globals import MODELS_DIR

import logging

def compute_weights_for_regression(temp_lagged, precip_lagged):
    weights = np.ones_like(temp_lagged, dtype=float)  # Default weight of 1

    return weights

def train(forecaster, X_train, y_train, X_val, y_val, forecasting_task_sufix, pipeline_id, learner, config, resume_training: bool = False):
    NUM_FEATURES = X_train.shape[2]
    print(f"Number of features: {NUM_FEATURES}")
    print("- Forecasting task: regression.")     
    train_weights = compute_weights_for_regression(X_train[:, :, 9], X_train[:, :, 12])
    val_weights = compute_weights_for_regression(X_val[:, :, 9], X_val[:, :, 12])
    train_weights = torch.FloatTensor(train_weights)
    val_weights = torch.FloatTensor(val_weights)            
    loss = nn.MSELoss()

    print(forecaster)

    BATCH_SIZE = config["training"][forecasting_task_sufix]["BATCH_SIZE"]
    LEARNING_RATE = config["training"][forecasting_task_sufix]["LEARNING_RATE"]
    N_EPOCHS = config["training"][forecasting_task_sufix]["N_EPOCHS"]
    PATIENCE = config["training"][forecasting_task_sufix]["PATIENCE"]
    WEIGHT_DECAY = float(config["training"][forecasting_task_sufix]["WEIGHT_DECAY"])  # Ensure float

    optimizer = torch.optim.AdamW(
        forecaster.learner.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f" - Setting up optimizer: {optimizer}")
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    print(f" - Creating data loaders.")
    train_loader = learner.create_dataloader(X_train, y_train, batch_size=BATCH_SIZE, weights=train_weights, shuffle=True)
    val_loader = learner.create_dataloader(X_val, y_val, batch_size=BATCH_SIZE, weights=val_weights)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" - Moving data and parameters to {device}.")
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    to_device(forecaster.learner, device)

    print(f" - Fitting model...", end=" ")
    train_loss, val_loss = forecaster.learner.fit(n_epochs=N_EPOCHS,
                                          optimizer=optimizer,
                                          train_loader=train_loader,
                                          val_loader=val_loader,
                                          patience=PATIENCE,
                                          criterion=loss,
                                          pipeline_id=pipeline_id)
    print("Done!")

    gen_learning_curve(train_loss, val_loss, pipeline_id)

    #
    # Load the best model obtainined throughout the training epochs.
    #
    forecaster.learner.load_state_dict(torch.load(MODELS_DIR + '/best_' + pipeline_id + '.pt'))


def main(argv):
    parser = argparse.ArgumentParser(description="Train a rainfall forecasting model.")
    parser.add_argument("-l", "--learner", choices=["Conv1DNeuralNet", "LstmNeuralNet"],
                        default="LstmNeuralNet", help="Learning algorithm to be used.")
    parser.add_argument("-p", "--pipeline_id", required=True, help="Pipeline ID")
    
    args = parser.parse_args(argv[1:])

    seed_everything()

    X_train, y_train, X_val, y_val, X_test, y_test = pipeline.load_datasets(args.pipeline_id, True)

    prediction_task_sufix = "reg"
    args.pipeline_id += "_reg"

    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    SEQ_LENGTH = config["preproc"]["SLIDING_WINDOW_SIZE"]
    BATCH_SIZE = config["training"][prediction_task_sufix]["BATCH_SIZE"]
    DROPOUT_RATE = config["training"][prediction_task_sufix]["DROPOUT_RATE"]
    OUTPUT_SIZE = config["training"][prediction_task_sufix]["OUTPUT_SIZE"]

    # Use globals() to access the global namespace and find the class by name
    class_name = args.learner
    print(class_name)
    class_obj = globals()[class_name]

    # Check if the class exists
    if not isinstance(class_obj, type):
        raise ValueError(f"Class '{class_name}' not found.")

    args.pipeline_id += "_" + class_name

    NUM_FEATURES = X_train.shape[2]
    print(f"Number of features: {NUM_FEATURES}")

    # Instantiate the class
    learner = class_obj(seq_length = SEQ_LENGTH,
                        input_size = NUM_FEATURES, 
                        output_size = OUTPUT_SIZE,
                        dropout_rate = DROPOUT_RATE)
    print(f'Learner: {learner}')

    y_mean_value = np.mean(y_train)
    forecaster = Regressor(learner, in_channels=NUM_FEATURES, y_mean_value=y_mean_value)

    # Build model
    start_time = time.time()
    train(forecaster, X_train, y_train, X_val, y_val, prediction_task_sufix, args.pipeline_id, learner, config)
    logging.info("Model training took %s seconds." % (time.time() - start_time))

    # Evaluate using the best model produced
    test_loader = learner.create_dataloader(X_test, y_test, batch_size=BATCH_SIZE )
    forecaster.save_evaluation_report(args.pipeline_id, test_loader)

if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    execution_time = end_time - start_time
    print("The execution time was", execution_time, "seconds.")
