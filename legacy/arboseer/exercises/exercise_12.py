# Plot gaps created on LST files for testing

import matplotlib.pyplot as plt
import os
import numpy as np

def plot(dataset):
    lst_dir = "data/processed/lst_ts_gaps"
    processed_dir = "data/processed/lst_ts_gaps/images"
    lst_files = sorted([f for f in os.listdir(lst_dir) if f.endswith('.npz')])
    os.makedirs(processed_dir, exist_ok=True)  # Ensure output directory exists    

    cont = 0
    for lst_file in lst_files:
        cont = cont + 1
        data = np.load(os.path.join(lst_dir, lst_file)) 
        data_to_plot = data[dataset]
        file_name = lst_file[:-4]

        # Plotting setup
        plt.figure(figsize=(7, 7))
        ax = plt.axes()
        ax.imshow(data_to_plot, cmap='jet', alpha=1, origin='upper')
        
        plt.savefig(f"{processed_dir}/{file_name}_{dataset}.png")
        plt.close()  # Close the plot to free memory
        
        print(cont)
        if cont >= 50:
            break

#plot('ws_data')
#plot('original_data')
#plot('data_with_gaps')
plot('gap_mask')
plot('filled_data')