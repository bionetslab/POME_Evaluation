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

On the Helma cluster, use the Apptainer image instead: `container/run.sh <script>` for CPU
steps and `sbatch container/submit.sh <script>` for GPU steps (H200). See `container/README.md`.

## POME versions and configurations

The GPU scripts (`scripts/generate_embeddings.py`, `scripts/generate_inductive_embeddings.py`,
`scripts/generate_variable_representations.py`, `src/pome_evaluation/impute_graph_based.py`)
take `--pome-config NAME`, a named set of `Embedder` options defined in
`src/pome_evaluation/pome_configs.py`. Pair every non-default config with its own output
location (`--embeddings-root`, `--output-dir`, `--output-suffix`, `--tag`) so that the published
results are never overwritten. Outputs of a config carry its tag, e.g. `_pome14` or `_stable_k8`.

| Config | Model | Needs POME |
|---|---|---|
| `default` | as published: random encoding table, constant lr, one sampled negative | 1.2 or 1.3 (tag `v1.3`) |
| `stable`, `stable_k8` | training-noise reduction: spectral table (frozen), output Gram anchor (k = 8 for `stable_k8`), cosine lr decay | commit `cdc77be` |
| `stable_k8_learnvar[_a{w}]` | `stable_k8` with trainable variable encodings `e_j`, anchored to their spectral init with weight w | commit `cdc77be` |
| `pome_1.4` | **current POME defaults**: fixed spectral sample/value encodings, learned `e_j` anchored with w = 0.1, numeric imputation from the learned `e_j` | ≥ 1.4 |

`pome_1.4` is the same model as `stable_k8_learnvar_a0.1_rawreg`. On a spot check of the
imputation benchmark, POME 1.4 gave bit-identical results to that config on commit `cdc77be`. If
the installed POME does not accept a config's options, `pome_config()` raises an error that names
the POME version to install.

POME fits are only reproducible across processes with a fixed hash seed. Set
`PYTHONHASHSEED=0` when exact reproducibility matters.

## Generating manuscript figures

### Imputation analysis
Simply run
```bash
python scripts/generate_imputation_figure.py
```
For the full recipe (imputation, scoring, figure), see `REPRODUCE_IMPUTATION_FIGURES.md`.

**POME 1.4 rerun** (d = 64, z-discretization with 15 bins) →
`output/imputation_ranks_with_competitors_64_learnvar_rawreg.pdf`:
```bash
sbatch container/submit.sh src/pome_evaluation/impute_graph_based.py --discretizations z --bins 15 \
    --dims 64 --datasets HANCOCK --pome-config pome_1.4 --output-suffix _learnvar_rawreg   # per dataset
container/run.sh scripts/generate_imputation_results.py --discretization z --bins 15 --dims 64 \
    --pome-suffix _learnvar_rawreg
container/run.sh scripts/generate_imputation_figure.py --dim 64 --suffix _learnvar_rawreg
```
The shipped `_learnvar_rawreg` files were produced with the equivalent config
`stable_k8_learnvar_a0.1_rawreg` on POME `cdc77be`.

### Unsupervised stratification results
The published figure can be reproduced with the notebook `scripts/plot_fig4_unsupervised.ipynb`. The scripts to produce the required input for this notebook under `src/pome_evaluation` are `analyze_unsupervised_clustering.py`, `analyze_cluster_preservation_2D.ipynb`, and `analyze_distance_preservation_2D.ipynb`. Necessary UMAP embeddings were computed using the script `embed_UMAP_several_runs.py`. POME's embeddings 2D visualizations were computed with the help of the notebook `project_embeddings_to_2D.ipynb`.

`scripts/generate_unsupervised_figure.py` is a scriptable port of this chain, with Leiden
seeded so that the scores are reproducible. It recomputes the POME side (clustering scores, 2D
projections, distance and cluster preservation) for re-trained embeddings. It copies the UMAP
rows from the published score tables, because the MIMIC UMAP embeddings are not stored in this
repo. As published, each method is clustered and scored in its own metric: POME in squared
Euclidean distance (cosine distance, since its embeddings are unit length), UMAP in Euclidean
distance.

**POME 1.4 rerun** → `output/unsupervised/unsupervised_clustering_pome14.pdf`:
```bash
sbatch container/submit.sh scripts/generate_embeddings.py --dims 16 32 64 --runs 10 \
    --pome-config pome_1.4 --embeddings-root data/embeddings_pome14
container/run.sh scripts/generate_unsupervised_figure.py --embeddings-root data/embeddings_pome14 --tag _pome14
```

### Linear probing analysis
The results figure showing POME's embeddings' supervised learning capability can be reproduced with `scripts/plot_inductive_linear_probing.py`. As input, it takes `output/linear_probing/inductive_linear_probing_results.csv`, produced by `scripts/linear_probe_inductive_embeddings.py` from the train/test-split embeddings of `scripts/generate_inductive_embeddings.py`.

**POME 1.4 rerun** → `output/linear_probing/inductive_linear_probing_results_pome14.pdf`. Only POME
is re-probed; `scripts/merge_probing_variant.py` adds the published UMAP rows.
```bash
sbatch container/submit.sh scripts/generate_inductive_embeddings.py --methods pome --modes combined \
    --datasets hancock --dims 16 --runs 5 --pome-config pome_1.4 \
    --output-dir output/inductive_pome14 --log-tag hancock_d16          # per dataset x dim
container/run.sh scripts/linear_probe_inductive_embeddings.py --methods pome --modes combined \
    --dims 16 32 64 --runs 0 1 2 3 4 --emb-root output/inductive_pome14 \
    --output output/linear_probing/inductive_linear_probing_results_pome14_pome.csv
container/run.sh scripts/merge_probing_variant.py \
    --pome output/linear_probing/inductive_linear_probing_results_pome14_pome.csv \
    --output output/linear_probing/inductive_linear_probing_results_pome14.csv
container/run.sh scripts/plot_inductive_linear_probing.py --dims 16 32 64 \
    --results output/linear_probing/inductive_linear_probing_results_pome14.csv \
    --output output/linear_probing/inductive_linear_probing_results_pome14.pdf
```

The companion data-integration figure (does embedding continuous *and* categorical variables beat either type alone?) is drawn by `scripts/plot_inductive_type_combination.py --dim 64`, off the same results CSV; its `numeric_only` / `cat_only` embeddings come from `generate_inductive_embeddings.py --modes`.

The earlier transductive versions of both figures are retained for comparison: `scripts/plot_fig3_linear_probing.ipynb` and `scripts/plot_type_combination_effect.ipynb`, fed by the `src/pome_evaluation/analyze_*_embedding_separability.ipynb` notebooks via `output/supervised/`.

### Survival analysis on HANCOCK
Simply run
```bash
python scripts/generate_survival_figure.py
```

### Exploratory analysis of variable embeddings
The published figure showing POME's variable embedding results can be reproduced with the notebook `scripts/plot_fig5_variable_embeddings.ipynb`. As input, it takes output files of the analysis script `analyze_variable_embeddings.ipynb`. The required files storing feature importances for Aplasia and Neutropenic Fever are located under `data/feature_ranks_NF.csv` and `data/feature_ranks_aplasia.csv`.

**Revised figure** (10 seeds × encoding sizes 16/32/64 on MIMIC including both targets, with
the learned variable encodings `e_j`) →
`output/variable_embeddings/variable_results_revised_stable_k8_learnvar_a0.1_raw.pdf`:
- **a**: Spearman correlation between CatBoost importance and target-to-lab cosine, per size.
- **b**: cross-seed reproducibility of the target-to-lab profiles.
- **c**: fraction of seeds in which a lab falls into Aplasia's Leiden cluster (d = 32).

```bash
sbatch container/submit.sh scripts/generate_variable_representations.py --pome-config stable_k8_learnvar_a0.1
container/run.sh scripts/analyze_variable_representation_stability.py \
    --pome-config stable_k8_learnvar_a0.1 --importance-dir data/input_datasets
container/run.sh scripts/plot_fig5_variable_representations.py \
    --pome-config stable_k8_learnvar_a0.1 --option raw
```
`stable_k8_learnvar_a0.1` (POME `cdc77be`) trains the same variable encodings as POME 1.4.
`scripts/analyze_variable_anchor_scan.py` summarizes the scan over the anchor weight w (drift
from the init, cross-seed stability, importance correlation) →
`output/variable_embeddings/variable_anchor_scan.csv`.

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

### Training-noise ablation (run-to-run variance)

Explains where POME's seed-to-seed variation comes from: constant learning rate, sampled
negatives, random encoding table. It is the basis for the `stable*` configs and for the POME 1.3
and 1.4 defaults. `scripts/generate_training_noise_runs.py` (GPU, POME ≤ 1.3 / `cdc77be`) fits
repeated seeds of each training variant on one fixed split per dataset.
`scripts/analyze_training_noise_variance.py` (CPU) compares them against inductive UMAP and
published POME. It measures seed agreement (CKA, kNN overlap, ARI), the SD of probing AP and
imputation error across seeds, and cluster structure. Outputs go to `output/training_noise*/`
(per-seed metrics are committed; the raw runs are not).

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
