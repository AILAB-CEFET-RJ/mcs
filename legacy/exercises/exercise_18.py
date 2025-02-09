# Error distribution test on temporal series

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import os
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the paths to your data
data_path = 'data'
processed_path = os.path.join(data_path, 'processed')
processed_lst_path_gaps = os.path.join(processed_path, 'lst_ts_gaps')

def explore_errors(data_path):
    """
    This function loads data, calculates errors, and performs explorations
    including visualizations and statistical analysis for a time series experiment.

    Args:
    data_path (str): The path to the directory containing the NPZ files.
    """

    def mean_absolute_error(original, filled):
        """Calculate mean absolute error where filled data is not NaN."""
        mask = ~np.isnan(filled)
        error = np.abs(original[mask] - filled[mask])
        return np.mean(error)

    def root_mean_square_error(original, filled):
        """Calculate root mean square error where filled data is not NaN."""
        mask = ~np.isnan(filled)
        error = (original[mask] - filled[mask]) ** 2
        return np.sqrt(np.mean(error))

    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    files.sort(key=lambda x: datetime.strptime(x[:-4], "%Y%m%d%H%M"))

    mae_values = []
    rmse_values = []

    for file_name in files:
        file_path = os.path.join(data_path, file_name)
        try:
            data = np.load(file_path)
            original_data = data['original_data']
            filled_data = data['filled_data']
            gap_mask = data['gap_mask']

            if np.all(np.isnan(gap_mask)):
                logging.info(f"Skipping file due to all NaN gap_mask: {file_name}")
                continue

            mae = mean_absolute_error(original_data, filled_data)
            rmse = root_mean_square_error(original_data, filled_data)

            mae_values.append(mae)
            rmse_values.append(rmse)

            logging.info(f"Calculated MAE: {mae} and RMSE: {rmse} for file: {file_name}")

        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")

    if mae_values and rmse_values:
        average_mae = np.mean(mae_values)
        average_rmse = np.mean(rmse_values)
        logging.info(f"Average Mean Absolute Error across all files: {average_mae}")
        logging.info(f"Average Root Mean Square Error across all files: {average_rmse}")

        # Plotting the error distributions
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(mae_values, bins='auto', color='blue', alpha=0.7)
        plt.title('Distribution of MAE')
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Frequency')
        plt.annotate(f'Average MAE: {average_mae:.2f}', xy=(0.70, 0.90), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white'))

        plt.subplot(1, 2, 2)
        plt.hist(rmse_values, bins='auto', color='red', alpha=0.7)
        plt.title('Distribution of RMSE')
        plt.xlabel('Root Mean Square Error')
        plt.ylabel('Frequency')
        plt.annotate(f'Average RMSE: {average_rmse:.2f}', xy=(0.70, 0.90), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'))

        plt.tight_layout()
        plt.savefig('error_distributions.png')
        plt.show()
    else:
        logging.info("No valid data to process.")

# Example usage
explore_errors(processed_lst_path_gaps)
