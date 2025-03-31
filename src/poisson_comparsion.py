import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import sys
import os

def main(dataset_path, image_destination, remove_zeros):
    # Get arguments from command line
    # dataset_path = sys.argv[1]
    # image_destination = sys.argv[2]
    # remove_zeros = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False

    # Modify filename based on whether zeros are removed
    filename_suffix = "no_zeroes" if remove_zeros else "with_zeroes"
    image_filename = os.path.join(image_destination, os.path.basename(dataset_path).replace('.pickle', f'_poisson_vs_neg_binom_zip_{filename_suffix}.png'))

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
    zero_proportion = np.sum(all_y == 0) / len(all_y)

    # Generate Poisson distribution with the same means
    poisson_dist = stats.poisson(mu=mean_y)
    y_values = np.arange(0, int(mean_y + 3 * np.sqrt(var_y)))
    poisson_pmf = poisson_dist.pmf(y_values)

    # Estimate Negative Binomial parameters
    if var_y > mean_y:
        r = (mean_y ** 2) / (var_y - mean_y)  # Shape parameter
        p = mean_y / var_y  # Probability of success
        neg_binom_pmf = stats.nbinom.pmf(y_values, r, p)
    else:
        neg_binom_pmf = None  # Can't fit Negative Binomial if variance is not greater than mean

    # Estimate Zero-Inflated Poisson (ZIP) probabilities
    zip_pmf = (1 - zero_proportion) * stats.poisson.pmf(y_values, mu=mean_y)

    # Plot Poisson, Negative Binomial, and ZIP distributions
    plt.figure(figsize=(10, 6))
    plt.plot(y_values, poisson_pmf, 'r-', marker='o', label=f'Poisson (mean={mean_y:.2f}, var={var_y:.2f})')
    set
    if neg_binom_pmf is not None:
        plt.plot(y_values, neg_binom_pmf, 'b-', marker='s', label=f'Negative Binomial (r={r:.2f}, p={p:.2f})')
    
    plt.plot(y_values, zip_pmf, 'g-', marker='^', label=f'Zero-Inflated Poisson (ZIP)')
    
    # Labels and title
    plt.xlabel("Y Values")
    plt.ylabel("Probability Mass Function (PMF)")
    plt.title("Poisson vs Negative Binomial vs Zero-Inflated Poisson (ZIP) Distribution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save figure
    plt.savefig(image_filename)
        
    print(f"Figure saved to {image_filename}")

if __name__ == "__main__":
    main("data/datasets/FULL.pickle", "data/comparsion", False)
    main("data/datasets/7427549.pickle", "data/comparsion", False)
    main("data/datasets/2268922.pickle", "data/comparsion", False)
    main("data/datasets/7149328.pickle", "data/comparsion", False)
    main("data/datasets/2299216.pickle", "data/comparsion", False)
    main("data/datasets/0106453.pickle", "data/comparsion", False)
    main("data/datasets/6870066.pickle", "data/comparsion", False)
    main("data/datasets/6042619.pickle", "data/comparsion", False)
    main("data/datasets/2288893.pickle", "data/comparsion", False)
    main("data/datasets/5106702.pickle", "data/comparsion", False)
    main("data/datasets/6635148.pickle", "data/comparsion", False)
    main("data/datasets/2269481.pickle", "data/comparsion", False)
    main("data/datasets/2708353.pickle", "data/comparsion", False)
    main("data/datasets/7591136.pickle", "data/comparsion", False)
    main("data/datasets/2283395.pickle", "data/comparsion", False)
    main("data/datasets/2287579.pickle", "data/comparsion", False)
    main("data/datasets/2291533.pickle", "data/comparsion", False)
    main("data/datasets/2292386.pickle", "data/comparsion", False)
    main("data/datasets/0012505.pickle", "data/comparsion", False)
    main("data/datasets/2292084.pickle", "data/comparsion", False)
    main("data/datasets/6518893.pickle", "data/comparsion", False)