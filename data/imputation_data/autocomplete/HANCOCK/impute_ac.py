import pandas as pd
import os
import numpy as np
import subprocess

# Iterate over complete simulated data directory and impute all datasets.
# Directory containing your pickle files
input_directory = 'simulated_ohe_data/'
output_directory = 'imputed_ohe_data/'

# Iterate over all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        print("Processing ", filename, "...")
        # Full file path
        file_path = os.path.join(input_directory, filename)
        
        # Get filename without extension.
        name_without_ext = os.path.splitext(filename)[0]
        
        # Runn Autocomplete imputation.
        output_file = os.path.join(output_directory, f'{name_without_ext}.csv')
        command = f'conda activate ac; python ~/Projects/AutoComplete.git/fit.py {file_path} --id_name ID --batch_size 1024 --epochs 50 --lr 0.1 --save_imputed --device cpu:0 --output --quality {output_file}'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)