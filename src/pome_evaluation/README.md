# Analysis scripts for POME Evaluation

## Compute low-dimensional UMAP embeddings

For computing required low-dimensional UMAP embeddings on input datasets, run the script `embed_UMAP_several_runs.py` with the desired dataset to embed specified on the top part of the file. 

## Project POME embeddings to 2D

For computing 2D visualizations (using PCA, t-SNE, UMAP) of POME's 16-, 32-, and 64-dimensional embeddings, you can run the notebook `project_embeddings_to_2D.ipynb`.

## Analyze unsupervised clusterability

For computing clusterability metrics on POME and UMAP embeddings, run the script `analyze_unsupervised_clustering.py` with the desired dataset specified in the beginning of the file.

## Analyze visualization techniques

For analyzing which visualization technique best preserves high-dimension cluster, run the notebook `analyze_cluster_preservation_2D.ipynb`. For analyzing which visualization technique best preservese local neighborhood structures, simply run the notebook `analyze_distance_preservation_2D.ipynb`.
