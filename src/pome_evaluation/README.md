# Analysis scripts for POME Evaluation

## Compute low-dimensional UMAP embeddings

For computing required low-dimensional UMAP embeddings on input datasets, run the script `embed_UMAP_several_runs.py` with the desired dataset to embed specified on the top part of the file. 

## Project POME embeddings to 2D

For computing 2D visualizations (using PCA, t-SNE, UMAP) of POME's 16-, 32-, and 64-dimensional embeddings, you can run the notebook `project_embeddings_to_2D.ipynb`.

## Analyze unsupervised clusterability

For computing clusterability metrics on POME and UMAP embeddings, run the script `analyze_unsupervised_clustering.py` with the desired dataset specified in the beginning of the file.

## Analyze visualization techniques

For analyzing which visualization technique best preserves high-dimension cluster, run the notebook `analyze_cluster_preservation_2D.ipynb`. For analyzing which visualization technique best preservese local neighborhood structures, simply run the notebook `analyze_distance_preservation_2D.ipynb`.

## Analyze supervised learning results

For comparing how well POME's and UMAP's embeddings are suitable for predicting held-out target variables by using a simple logistic regression model, we provide one notebook for each dataset separately: `analyze_HANCOCK_embedding_separability.ipynb`, `analyze_LUAD_embedding_separability.ipynb`, and `analyze_MIMIC_embedding_separability.ipynb`.

## Generate simulated missingness datasets

In order to simulate certain amounts of missingness into the given datasets, simply run the notebook `generate_simulated_missingness.ipynb` with updated paths pointing to the files of the desired dataset. Scripts and notebooks to impute simulated datasets with the different imputation methods can be found in the respective subdirectories under `data/imputation_data/`.

## Analyze imputation results

In order to compute mean absolute errors and multiclass accuracies of imputed values against ground truth values, you can make use of the Python notebook `compute_imputation_results.ipynb`.
