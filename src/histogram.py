import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import seaborn as sns

def main():
    # Get arguments from command line
    dataset_path = sys.argv[1]
    image_destination = sys.argv[2]
    remove_zeros = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False

    # Modify filename based on whether zeros are removed
    filename_suffix = "no_zeroes" if remove_zeros else "with_zeroes"
    image_filename = os.path.join(image_destination, os.path.basename(dataset_path).replace('.pickle', f'_hist_{filename_suffix}.png'))

    # Load dataset
    with open(dataset_path, 'rb') as file:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

    # Concatenate all y values
    all_y = np.concatenate([y_train, y_val, y_test])
    
    # Remove zeros if specified
    if remove_zeros:
        all_y = all_y[all_y > 0]

    # Plot histogram of Y values
    plt.figure(figsize=(12, 6))
    sns.histplot(all_y, bins=50, kde=False, color='blue', alpha=0.7)
    
    # Labels and title
    plt.xlabel("Y Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Y Values in Dataset")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Save figure
    plt.savefig(image_filename)
    
    print(f"Histogram saved to {image_filename}")

if __name__ == "__main__":
    main()
