import numpy as np
import pickle
import pandas as pd
import sys
import os

def main():
    # Get arguments from command line
    dataset_path = sys.argv[1]
    csv_destination = sys.argv[2]

    # Generate CSV filename with the same name as dataset but with .csv extension
    csv_filename = os.path.join(csv_destination, os.path.basename(dataset_path).replace('.pickle', '.csv'))

    # Load dataset
    with open(dataset_path, 'rb') as file:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

    # Concatenate all y values
    all_y = np.concatenate([y_train, y_val, y_test])

    # Count occurrences of each unique value
    unique_values, counts = np.unique(all_y, return_counts=True)

    # Save data to CSV
    hist_data = pd.DataFrame({
        "Number": unique_values,
        "Count": counts
    })
    hist_data.to_csv(csv_filename, index=False)

    print(f"Data saved to {csv_filename}")

if __name__ == "__main__":
    main()
