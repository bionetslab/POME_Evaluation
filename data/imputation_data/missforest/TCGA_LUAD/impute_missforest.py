import pandas as pd
import os
from missforest import MissForest
import numpy as np
import re

# Function to clean feature names (LightGBM-safe)
def clean_feature_names(columns):
    return [
        re.sub(r'[^A-Za-z0-9_]+', '_', str(col))
        for col in columns
    ]

input_directory = 'simulated_data/'
output_directory = 'imputed_data/'

cat_var_df = pd.read_csv("TCGA_LUAD_cat_vars.csv")
cat_vars = cat_var_df["cat_var"].tolist()

for filename in os.listdir(input_directory):
    if filename.endswith('.tsv'):

        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path, sep='\t', index_col=0)

        df = df.replace(pd.NA, np.nan)

        # 🔹 Store original column names
        original_columns = df.columns.tolist()

        # 🔹 Clean column names for LightGBM
        cleaned_columns = clean_feature_names(original_columns)

        # Create mapping
        col_mapping = dict(zip(original_columns, cleaned_columns))

        df.columns = cleaned_columns

        # 🔹 IMPORTANT: Also clean categorical variable names
        cleaned_cat_vars = [col_mapping[col] for col in cat_vars if col in col_mapping]

        # Run MissForest
        imputer = MissForest(categorical=cleaned_cat_vars)
        imputed_array = imputer.fit_transform(df)

        # Convert back to DataFrame
        imputed_df = pd.DataFrame(
            imputed_array,
            columns=cleaned_columns,
            index=df.index
        )

        # 🔹 Restore original column names
        imputed_df.columns = original_columns

        # Save
        name_without_ext = os.path.splitext(filename)[0]
        imputed_filename = os.path.join(output_directory, f'{name_without_ext}.tsv')
        imputed_df.to_csv(imputed_filename, sep='\t', index=True)