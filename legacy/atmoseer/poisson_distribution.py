import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import sys
import os

def main():
    # Get arguments from command line
    dataset_path = sys.argv[1]
    image_destination = sys.argv[2]
    remove_zeros = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False

    # Modify filename based on whether zeros are removed
    filename_suffix = "no_zeroes" if remove_zeros else "with_zeroes"
    image_filename = os.path.join(image_destination, os.path.basename(dataset_path).replace('.pickle', f'_{filename_suffix}.png'))

    # Load dataset
    with open(dataset_path, 'rb') as file:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

    # Concatenate all y values
    all_y = np.concatenate([y_train, y_val, y_test])
    
    # Remove zeros if specified
    if remove_zeros:
        all_y = all_y[all_y > 0]

    # Compute mean and variance
    mean_y = np.mean(all_y)
    var_y = np.var(all_y)

    # Generate Poisson distribution with the same mean
    poisson_dist = stats.poisson(mu=mean_y)
    y_values = np.arange(0, int(mean_y + 3 * np.sqrt(var_y)))
    poisson_pmf = poisson_dist.pmf(y_values)

    # Plot Poisson distribution
    plt.figure(figsize=(10, 6))
    plt.plot(y_values, poisson_pmf, 'r-', marker='o', label=f'Poisson Distribution (mean={mean_y:.2f}, variance={var_y:.2f})')
    
    # Annotate mean and variance
    plt.axvline(mean_y, color='green', linestyle='--', label=f'Mean (μ) = {mean_y:.2f}')
    plt.axvline(mean_y + np.sqrt(var_y), color='purple', linestyle=':', label=f'Std Dev (σ) = {np.sqrt(var_y):.2f}')
    plt.axvline(mean_y - np.sqrt(var_y), color='purple', linestyle=':')
    
    # Labels and title
    plt.xlabel("Y Values")
    plt.ylabel("Probability Mass Function (PMF)")
    plt.title("Poisson Distribution of Dataset")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save figure
    plt.savefig(image_filename)
    
    print(f"Figure saved to {image_filename}")

if __name__ == "__main__":
    main()
