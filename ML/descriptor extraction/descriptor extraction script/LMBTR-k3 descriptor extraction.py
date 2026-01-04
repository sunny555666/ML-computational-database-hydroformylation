import os
from ase.io import read
from ase import Atoms
from dscribe.descriptors import LMBTR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set LMBTR descriptor parameters
lmbtr_params = {
    'species': ['Rh', 'C', 'O', 'P', 'H', 'Br', 'F', 'Cl', 'S', 'N'],  # Elements to analyze, including Rh
    'geometry': {'function': 'cosine'},  # Geometry function
    'grid': {'min': 0, 'max': 5, 'sigma': 0.01, 'n': 200},  # Grid parameters
    'weighting': {'function': 'exp', "scale": 0.5, 'threshold': 1e-3},  # Weighting function
    'periodic': False,  # Whether to use periodic boundary conditions
    'normalization': 'l2',  # Normalization method
}

# Create LMBTR descriptor object
lmbtr = LMBTR(**lmbtr_params)

# Target folder path (modify this path according to your needs)
target_folder = r'path/to/your/xyz/files'

# Traverse all .xyz files in the target folder
for filename in os.listdir(target_folder):
    if filename.endswith('.xyz'):
        file_path = os.path.join(target_folder, filename)
        base_name = os.path.splitext(filename)[0]
        
        # Read .xyz file using ASE and extract atomic coordinates
        try:
            atoms = read(file_path, format='xyz')
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue
        
        # Find the index of Rh atom
        rh_index = [i for i, atom in enumerate(atoms) if atom.symbol == 'Rh']
        if not rh_index:
            print(f"No Rh atom found in {filename}, skipping.")
            continue
        center_position = atoms[rh_index[0]].position
        
        # Calculate LMBTR descriptor
        try:
            lmbtr_output = lmbtr.create(atoms, centers=[center_position])
        except Exception as e:
            print(f"Failed to calculate LMBTR for {filename}: {e}")
            continue

        # Create mapping between indices and chemical symbols
        n_elements = len(lmbtr_params['species'])
        x = np.linspace(lmbtr_params['grid']['min'], lmbtr_params['grid']['max'], lmbtr_params['grid']['n'])

        # Prepare data for saving numerical results
        data_dict = {'Cosine': x}
        plotted = False  # Check if at least one line was plotted
        
        # Plot graphs and collect data
        fig, ax = plt.subplots()
        for i_species in lmbtr_params['species']:
            for j_species in lmbtr_params['species']:
                for k_species in lmbtr_params['species']:
                    if i_species == 'Rh' or j_species == 'Rh' or k_species == 'Rh':
                        try:
                            # Ensure the center species is valid
                            loc = lmbtr.get_location(('X', j_species, k_species))
                            col_name = f"Rh-{j_species}-{k_species}"
                            
                            if loc is not None:  # Ensure location is valid
                                y_values = lmbtr_output[0, loc]
                                plt.plot(x, y_values, label=col_name)
                                data_dict[col_name] = y_values
                                plotted = True
                        except Exception as e:
                            print(f"Failed to plot {i_species}-{j_species}-{k_species} for {filename}: {e}")
                            continue

        if plotted:
            # Save plot
            ax.set_xlabel("Cosine")
            ax.legend()
            plt.title(f"LMBTR plot for {filename}")
            plt.savefig(os.path.join(target_folder, f"{base_name}_lmbtr.png"))
            plt.close(fig)
            
            # Save numerical results as CSV file
            df = pd.DataFrame(data_dict)
            csv_filename = os.path.join(target_folder, f"{base_name}_lmbtrk3.csv")
            df.to_csv(csv_filename, index=False)
            print(f"LMBTR plot and data saved for {filename} (plot: {base_name}_lmbtr.png, data: {base_name}_lmbtrk3.csv)\n")
        else:
            print(f"No valid plots or data for {filename}\n")