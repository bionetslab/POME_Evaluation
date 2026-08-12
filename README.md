# POME Evaluation

Python code to reproduce all figures and analyses used for the evaluation of POME.

## Installation

### Using conda

```bash
conda env create -f environment.yml
conda activate hancock_survival
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Generating manuscript figures

### Imputation analysis
Simply run
```bash
python scripts/generate_imputation_figure.py
```

### Unsupervised stratification results
The figure can be reproduced with the notebook `scripts/plot_fig4_unsupervised.ipynb`. The scripts to produce the required input for this notebook under `src/pome_evaluation` are `analyze_unsupervised_clustering.py`, `analyze_cluster_preservation_2D.ipynb`, and `analyze_distance_preservation_2D.ipynb`. Necessary UMAP embeddings were computed using the script `embed_UMAP_several_runs.py`. POME's embeddings 2D visualizations were computed with the help of the notebook `project_embeddings_to_2D.ipynb`.

### Linear probing analysis
The results figure showing POME's embeddings' supervised learning capability can be reproduced with `scripts/plot_inductive_linear_probing.py`. As input, it takes `output/linear_probing/inductive_linear_probing_results.csv`, produced by `scripts/linear_probe_inductive_embeddings.py` from the train/test-split embeddings of `scripts/generate_inductive_embeddings.py`.

The companion data-integration figure (does embedding continuous *and* categorical variables beat either type alone?) is drawn by `scripts/plot_inductive_type_combination.py --dim 64`, off the same results CSV; its `numeric_only` / `cat_only` embeddings come from `generate_inductive_embeddings.py --modes`.

The earlier transductive versions of both figures are retained for comparison: `scripts/plot_fig3_linear_probing.ipynb` and `scripts/plot_type_combination_effect.ipynb`, fed by the `src/pome_evaluation/analyze_*_embedding_separability.ipynb` notebooks via `output/supervised/`.

### Survival analysis on HANCOCK
Simply run
```bash
python scripts/generate_survival_figure.py
```

### Exploratory analysis of variable embeddings
The results figure showing POME's variable embedding results can be reproduced with the notebook `scripts/plot_fig5_variable_embeddings.ipynb`. As input, it takes output files of the analysis script `analyze_variable_embeddings.ipynb`. The required files storing feature importances for Aplasia and Neutropenic Fever are located under `data/feature_ranks_NF.csv` and `data/feature_ranks_aplasia.csv`.

### Supplement imputation plots
Simply run
```bash
python scripts/generate_imputation_binning_figure.py
python scripts/generate_imputation_dim_figure.py
```

### Supplement unsuperivsed results per embedding sizes
Simply use the notebook located at `scripts/plot_supplement_unsupervised_per_dimension.ipynb`.

### Supplement resource benchmark

Simply use the notebook located at `scripts/plot_supplement_resource_benchmark.ipynb`.

### Supplement imputation across epochs

Simply use the notebook located at `scripts/plot_supplement_imputation_epochs.ipynb`.
It reads `data/{DATASET}_imputation_per_epoch.csv`, produced by the GPU script
`scripts/generate_imputation_epoch_results.py` (see
`REPRODUCE_IMPUTATION_FIGURES.md`, Step 3).

## Analysis scripts for POME Evaluation
All of the following scripts for re-running the performed analyses on POME are located in the directory `src/pome_evaluation`.
### Compute low-dimensional UMAP embeddings

For computing required low-dimensional UMAP embeddings on input datasets, run the script `embed_UMAP_several_runs.py` with the desired dataset to embed specified on the top part of the file. 

### Project POME embeddings to 2D

For computing 2D visualizations (using PCA, t-SNE, UMAP) of POME's 16-, 32-, and 64-dimensional embeddings, you can run the notebook `project_embeddings_to_2D.ipynb`.

### Analyze unsupervised clusterability

For computing clusterability metrics on POME and UMAP embeddings, run the script `analyze_unsupervised_clustering.py` with the desired dataset specified in the beginning of the file.

### Analyze visualization techniques

For analyzing which visualization technique best preserves high-dimension cluster, run the notebook `analyze_cluster_preservation_2D.ipynb`. For analyzing which visualization technique best preservese local neighborhood structures, simply run the notebook `analyze_distance_preservation_2D.ipynb`.

### Analyze supervised learning results

For comparing how well POME's and UMAP's embeddings are suitable for predicting held-out target variables by using a simple logistic regression model, run `scripts/linear_probe_inductive_embeddings.py`. It probes every dataset, method and variable-type mode it finds into one tidy CSV, fitting on each training split and scoring the held-out test split. `scripts/linear_probe_full_dataset_embeddings.py` is the transductive control over the same splits.

### Generate simulated missingness datasets

In order to simulate certain amounts of missingness into the given datasets, simply run the notebook `generate_simulated_missingness.ipynb` with updated paths pointing to the files of the desired dataset. Scripts and notebooks to impute simulated datasets with the different imputation methods can be found in the respective subdirectories under `data/imputation_data/`.

### Analyze imputation results

In order to compute mean absolute errors and multiclass accuracies of imputed values against ground truth values, you can make use of the Python notebook `compute_imputation_results.ipynb`.
