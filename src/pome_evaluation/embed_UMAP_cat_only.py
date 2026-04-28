from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import umap
import os

os.chdir("../../data/input_datasets")

# Specify which dataset to embed with UMAP.
dataset = "TCGA_LUAD"
NUM_EMBEDDINGS = 10
num_dimensions = 16

if dataset=="HANCOCK":
    df = pd.read_csv("hancock_wo_targets_cat_only_UMAP.csv", index_col=0)
else:
    df = pd.read_csv("TCGA_LUAD_wo_targets_cat_only_UMAP.csv", index_col=0)

categorical = df.copy()

OUT_DIR = f'/home/wollerf/Projects/POME_Evaluation.git/data/embeddings/{dataset}/embeddings'

for num_run in range(NUM_EMBEDDINGS):
    print(f"Running categorical UMAP {num_run} of {NUM_EMBEDDINGS}...")
    
    cat_umap = umap.UMAP(n_components=num_dimensions, metric="hamming", random_state=num_run)
    cat_mapper = cat_umap.fit(categorical.values.copy(), ensure_all_finite='allow-nan')

    # Save UMAP embeddings to dataframe.
    embedding_index = df.index 
    embedding_cols = [f'dim_{i}' for i in range(num_dimensions)]
    embedding_df = pd.DataFrame(cat_mapper.embedding_, index=embedding_index, columns=embedding_cols)
    embedding_df.to_csv(os.path.join(OUT_DIR, f'{dataset}_UMAP_cat_only_{num_dimensions}_{num_run}.csv'), index=True)