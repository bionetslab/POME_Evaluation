from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import umap
import os

os.chdir("../../data/input_datasets")

# Specify which dataset to embed with UMAP.
dataset = "HANCOCK"
NUM_EMBEDDINGS = 10
num_dimensions = 16

if dataset=="HANCOCK":
    df = pd.read_csv("hancock_wo_targets.csv", index_col=0)
    cat_var_df = pd.read_csv("hancock_cat_variables.csv")
elif dataset=="MIMIC":
    df = pd.read_csv("mimic_aggregated_wo_targets_umap.csv", index_col=0)
    cat_var_df = pd.read_csv("mimic_aggregated_cat_variables.csv")
else:
    df = pd.read_csv("TCGA_LUAD_wo_targets.csv", index_col=0)
    cat_var_df = pd.read_csv("TCGA_LUAD_cat_vars.csv")

cat_variables = list(set(cat_var_df["cat_var"]).intersection(set(df.columns)))
cont_variables = list(set(df.columns) - set(cat_variables))

# Separate numerical and categorical features.
numeric = df[cont_variables].copy()
scaled_numeric = RobustScaler().fit_transform(numeric)
categorical = df[cat_variables].copy()

for num_run in range(NUM_EMBEDDINGS):
    print(f"Running UMAP {num_run} of {NUM_EMBEDDINGS}...")
    
    # Deactivate categorical embedder for MIMIC dataset.
    numeric_umap = umap.UMAP(n_components=num_dimensions, random_state=num_run)
    if dataset != "MIMIC":
        cat_umap = umap.UMAP(n_components=num_dimensions, metric="hamming", random_state=num_run)
    
    numeric_mapper = numeric_umap.fit(scaled_numeric.copy(), ensure_all_finite='allow-nan')
    if dataset != "MIMIC":
        ordinal_mapper = cat_umap.fit(categorical.values.copy(), ensure_all_finite='allow-nan')

    # Intersect separately fitted UMAP embeddings.
    if dataset != "MIMIC":
        intersection_mapper = numeric_mapper * ordinal_mapper
    else:
        intersection_mapper = numeric_mapper

    # Save UMAP embeddings to dataframe.
    embedding_index = df.index 
    embedding_cols = [f'dim_{i}' for i in range(num_dimensions)]
    embedding_df = pd.DataFrame(intersection_mapper.embedding_, index=embedding_index, columns=embedding_cols)
    embedding_df.to_csv(f'{dataset}_UMAP_{num_dimensions}_{num_run}.csv', index=True)