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

## POME version

All POME scripts use POME's own defaults and need **POME ≥ 1.4**. In POME 1.4 the sample and
value encodings are fixed at their spectral initialization, and only the variable encodings `e_j`
are learned, anchored to their initialization. Numeric imputation uses the learned `e_j`.

POME 1.4 results carry the tag `_pome14` and are the scripts' default output locations, e.g.
`data/embeddings_pome14`, `output/inductive_pome14` and `imputed_*_pome14`. The published results
(untagged files) are never overwritten. To regenerate the published results themselves, check out
the tag `pre-pome-1.4` of this repo and install POME `v1.3`.

POME fits are only reproducible across processes with a fixed hash seed. Set
`PYTHONHASHSEED=0` when exact reproducibility matters.

## Generating manuscript figures

### Imputation analysis
Simply run
```bash
python scripts/generate_imputation_figure.py
```
For the full recipe (imputation, scoring, figure), see `REPRODUCE_IMPUTATION_FIGURES.md`.

**POME 1.4** (d = 64, z-discretization with 15 bins) →
`output/imputation_ranks_with_competitors_64_pome14.pdf`:
```bash
sbatch container/submit.sh src/pome_evaluation/impute_graph_based.py --discretizations z --bins 15 \
    --dims 64 --datasets HANCOCK                                   # per dataset; -> imputed_z_15_64_pome14
container/run.sh scripts/generate_imputation_results.py --discretization z --bins 15 --dims 64 \
    --pome-suffix _pome14
container/run.sh scripts/generate_imputation_figure.py --dim 64 --suffix _pome14
```

**Competitors.** k-NN, MissForest and AutoComplete (imputed by the scripts/notebooks under
`data/imputation_data/<method>/`), plus **HI-VAE** (Nazabal et al., 2020; the authors' original
TF1 code) and **TabImpute** (Feitelberg et al., 2025; zero-shot pre-trained transformer). The
latter two each have their own Apptainer image and script; both read the MissForest copies of the
masked files and write `data/imputation_data/{hivae,tabimpute}/{DATASET}/imputed_data/`, which
`generate_imputation_results.py` scores as tools `HI-VAE` and `TabImpute`:
```bash
bash container/build.sh hivae && bash container/build.sh tabimpute    # once, on a frontend
# --export is required: submit.sh runs with --export=NONE, so a plain env var never reaches the job
sbatch --export=POME_EVAL_SIF=$(ws_find pome_eval)/containers/hivae.sif \
    container/submit.sh scripts/impute_hivae.py --jobs 4                # CPU only (TF 1.15)
sbatch --export=POME_EVAL_SIF=$(ws_find pome_eval)/containers/tabimpute.sif \
    container/submit.sh scripts/impute_tabimpute.py --datasets hancock # GPU; ~1.5/3.5/5.5 min per
                                                                       # HANCOCK/LUAD/MIMIC file
```
Configuration choices (type mapping, HI-VAE's batch-size rule for sparse tables, TabImpute's
100 x 50 windows and permutation ensemble) are documented in the two scripts' docstrings.

### Unsupervised stratification results
The published figure can be reproduced with the notebook `scripts/plot_fig4_unsupervised.ipynb`. The scripts to produce the required input for this notebook under `src/pome_evaluation` are `analyze_unsupervised_clustering.py`, `analyze_cluster_preservation_2D.ipynb`, and `analyze_distance_preservation_2D.ipynb`. Necessary UMAP embeddings were computed using the script `embed_UMAP_several_runs.py`. POME's embeddings 2D visualizations were computed with the help of the notebook `project_embeddings_to_2D.ipynb`.

`scripts/generate_unsupervised_figure.py` is a scriptable port of this chain, with Leiden
seeded so that the scores are reproducible. It recomputes the POME side (clustering scores, 2D
projections, distance and cluster preservation) for re-trained embeddings. It copies the UMAP
rows from the published score tables, because the MIMIC UMAP embeddings are not stored in this
repo. As published, each method is clustered and scored in its own metric: POME in squared
Euclidean distance (cosine distance, since its embeddings are unit length), UMAP in Euclidean
distance.

**POME 1.4** → `output/unsupervised/unsupervised_clustering_pome14.pdf`:
```bash
sbatch container/submit.sh scripts/generate_embeddings.py --dims 16 32 64 --runs 10   # -> data/embeddings_pome14
container/run.sh scripts/generate_unsupervised_figure.py --methods POME UMAP
```

**Competitors (HI-VAE, SAINT, missForest + autoencoder)** → `output/unsupervised/unsupervised_clustering_pome14_competitors.pdf`.
The script also scores tabular embeddings of the same sizes (16/32/64) and runs (0-9). They are
read from `data/embeddings_competitors/{DATASET}/embeddings/{DATASET}_{METHOD}_{dim}_{run}.tsv`
and clustered and scored like UMAP (Euclidean kNN graph, Euclidean Silhouette, sklearn
Davies-Bouldin). All competitors read POME's own input graphs, so they see exactly the same
data (`src/pome_evaluation/tabular_inputs.py`).
- `HIVAE`: HI-VAE on the incomplete tables, using the setup of the imputation benchmark with
  dim z set to the embedding size. The embedding is the posterior mean of z
  (`scripts/embed_hivae.py`, `hivae.sif`, CPU).
- `SAINT`: SAINT's self-supervised pretraining on the incomplete tables, using SAINT's own
  missing-value mask embeddings. The embedding is the CLS-token output
  (`scripts/embed_saint.py`, `saint.sif`, GPU).
- `MFAE`: missForest imputation of the full tables, with the settings of the imputation
  benchmark (`scripts/impute_full_missforest.py`). A plain MLP autoencoder then runs on the
  completed tables (z-scored continuous + one-hot categorical; p-256-128-d, ReLU; MSE; Adam
  1e-3, batch 64, 100 epochs). The embedding is the bottleneck
  (`scripts/embed_autoencoder.py`, `pome_eval.sif`, CPU).
```bash
bash container/build.sh saint                                   # once, on a frontend
sbatch --export=POME_EVAL_SIF=$(ws_find pome_eval)/containers/hivae.sif,PYTHONHASHSEED=0 \
    container/submit.sh scripts/embed_hivae.py --jobs 8
sbatch --export=POME_EVAL_SIF=$(ws_find pome_eval)/containers/saint.sif,PYTHONHASHSEED=0 \
    container/submit.sh scripts/embed_saint.py
sbatch --export=PYTHONHASHSEED=0 container/submit.sh scripts/impute_full_missforest.py
container/run.sh scripts/embed_autoencoder.py                   # after the missForest job
container/run.sh scripts/generate_unsupervised_figure.py        # default --methods POME UMAP HI-VAE SAINT missForest+AE
```

### Linear probing analysis
The results figure showing POME's embeddings' supervised learning capability can be reproduced with `scripts/plot_inductive_linear_probing.py`. As input, it takes `output/linear_probing/inductive_linear_probing_results.csv`, produced by `scripts/linear_probe_inductive_embeddings.py` from the train/test-split embeddings of `scripts/generate_inductive_embeddings.py`.

**POME 1.4** → `output/linear_probing/inductive_linear_probing_results_pome14.pdf`. Only POME
is re-probed; `scripts/merge_probing_variant.py` adds the published UMAP rows.
```bash
sbatch container/submit.sh scripts/generate_inductive_embeddings.py --methods pome --modes combined \
    --datasets hancock --dims 16 --runs 5 --log-tag hancock_d16   # per dataset x dim; -> output/inductive_pome14
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

**POME 1.4** (10 seeds × encoding sizes 16/32/64 on MIMIC including both targets, with the
learned variable encodings `e_j`) → `output/variable_embeddings/variable_results_revised_pome14.pdf`:
- **a**: Spearman correlation between CatBoost importance and target-to-lab cosine, per size.
- **b**: cross-seed reproducibility of the target-to-lab profiles.
- **c**: fraction of seeds in which a lab falls into Aplasia's Leiden cluster (d = 32).

```bash
sbatch container/submit.sh scripts/generate_variable_representations.py   # -> variable_representations_pome14
container/run.sh scripts/analyze_variable_representation_stability.py --importance-dir data/input_datasets
container/run.sh scripts/plot_fig5_variable_representations.py
```

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
It reads the published `data/{DATASET}_imputation_per_epoch.csv`. The GPU script
`scripts/generate_imputation_epoch_results.py` (see `REPRODUCE_IMPUTATION_FIGURES.md`, Step 3)
now writes the POME 1.4 tables `data/{DATASET}_imputation_per_epoch_pome14.csv`.

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
