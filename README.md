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
`scripts/linear_probe_inductive_embeddings.py` probes the inductive sample embeddings of POME and its
competitors. Every embedding is fit on a training split and transforms the held-out test split. The
probe is fit on the training-split embedding and scored on the test-split embedding (average precision)
for each binary target. There are 10 train/test splits per dataset, embedding sizes 16/32/64 and runs 0-4.
Two probes are used; both z-score the embedding coordinates with the training statistics as
preprocessing:

- `logreg` (default): logistic regression (L2, liblinear) →
  `output/linear_probing/inductive_logreg_probing_results.csv`
- `mlp`: one hidden layer of 64 ReLU units with early stopping; the L2 penalty is tuned by an inner
  3-fold CV on the training split → `output/linear_probing/inductive_mlp_probing_results.csv`

`scripts/plot_inductive_linear_probing.py` draws both probes in one figure,
`output/linear_probing/inductive_probing_results.pdf`. Panels b-i show one target each: for every
embedding size and method, the median AP of the logistic regression (filled circle) and of the MLP
(empty circle), joined by a line in the method's colour. Panel a has two bars per method (filled =
logistic regression, empty = MLP) counting the targets on which the method has the best median AP.

**Datasets.** HANCOCK, MIMIC-IV and the leakage-free TCGA-LUAD (`luad_noleak`). Four LUAD inputs record
the outcome or the follow-up after diagnosis and leak the targets: "New Neoplasm Event Post Initial
Therapy Indicator" (single-variable AUC 0.93 for Disease Free Status), "Person Neoplasm Cancer Status"
and two follow-up-time variables. `scripts/make_luad_noleak_splits.py` draws the LUAD train/test splits
with the generator of `scripts/generate_splits.py` (seed 42) and drops these variables. `luad_noleak` is
the only LUAD dataset of the inductive analyses.

**Methods** (the probing script reads each one from its own embedding root):
- **POME 1.4**: `scripts/generate_inductive_embeddings.py --methods pome` → `output/inductive_pome14`.
- **UMAP** (NA-aware fork): `scripts/generate_inductive_embeddings.py --methods umap` → `output/inductive`.
- The tabular competitors of the clustering analysis, run inductively on the same POME-format split
  files. Each one is fit on the training split only and embeds the test samples with the frozen model,
  with the clustering versions' training and architecture → `output/inductive_competitors/`:
  - **HI-VAE**: `scripts/embed_hivae_inductive.py` (`hivae.sif`, CPU). The training embedding comes
    from the authors' restore pass over the training split. The test embedding comes from one more
    restore pass of the frozen network over train + test, keeping the test rows. The authors' code
    re-codes categories per input file and normalizes per batch, so a test-only pass would be wrong.
  - **SAINT**: `scripts/embed_saint_inductive.py` (`saint.sif`, GPU). Label encoding, mean filling and
    standardization use training statistics only. Test categories never seen in training go to SAINT's
    mask token, and test samples are encoded in their own batches.
  - **missForest + autoencoder**: `scripts/embed_mfae_inductive.py` (CPU). missForest is fit on the
    training split and imputes train + test (cached in `_mfae_imputed/`). The design matrix uses
    training z-scores and training one-hot categories, and the autoencoder is trained on the training
    rows only.

**Supervised reference: TabICL.** `scripts/tabicl_inductive_reference.py` fits the pre-trained tabular
foundation model TabICL (`tabicl` 2.2.0, default settings and checkpoint `tabicl-classifier-v2-20260212.ckpt`)
with its classical `fit`/`predict_proba` on the raw training-split table and the target labels, and
scores the test split → `output/linear_probing/inductive_tabicl_reference.csv` (one row per split and
target). It produces no embedding and uses no probe, and it sees the labels, so it is not a competitor
but an upper reference: each target panel draws its median AP over splits as a dashed line. It takes no
part in the panel-a win count; the plot script prints how many wins also beat it (none so far). TabICL
is used instead of TabPFN because TabPFN ≥ 2.5 requires a Prior Labs license login before its weights
download. `tabicl`, `einops` and `huggingface_hub` are installed with
`pip install --no-deps --target $(ws_find pome_eval)/pylib` (`run.sh`/`submit.sh` put it on
`PYTHONPATH`); the checkpoint is stored in `$(ws_find pome_eval)/models/tabicl/`.

```bash
WS=$(ws_find pome_eval)
container/run.sh scripts/make_luad_noleak_splits.py
# POME: per dataset x dim (hancock mimic luad_noleak x 16 32 64)
sbatch --export=PYTHONHASHSEED=0 container/submit.sh scripts/generate_inductive_embeddings.py \
    --methods pome --modes combined --datasets hancock --dims 16 --runs 5 --log-tag hancock_d16
# UMAP: per dataset x dim on the GPU node, or on CPU
sbatch --export=PYTHONHASHSEED=0 container/submit.sh scripts/generate_inductive_embeddings.py \
    --methods umap --modes combined --datasets mimic --dims 16 --runs 5
# competitors
sbatch --job-name=ind_hivae --export=POME_EVAL_SIF=$WS/containers/hivae.sif \
    container/submit.sh scripts/embed_hivae_inductive.py --jobs 16
sbatch --job-name=ind_saint --export=POME_EVAL_SIF=$WS/containers/saint.sif,PYTHONHASHSEED=0 \
    container/submit.sh scripts/embed_saint_inductive.py --datasets hancock luad_noleak   # + mimic, per --dims
sbatch --job-name=ind_mfae container/submit.sh scripts/embed_mfae_inductive.py
container/run.sh scripts/tabicl_inductive_reference.py --device cpu   # supervised reference
# probe all methods and plot
for P in logreg mlp; do
    container/run.sh scripts/linear_probe_inductive_embeddings.py --probe $P --jobs 32 --dims 16 32 64 --runs 0 1 2 3 4
done
container/run.sh scripts/plot_inductive_linear_probing.py
```
`plot_inductive_linear_probing.py --methods POME UMAP` draws a subset of the methods.

The companion data-integration figure (does embedding continuous *and* categorical variables beat
either type alone?) is drawn by `scripts/plot_inductive_type_combination.py --dim 64`. It reads the
probing results with the `numeric_only` / `cat_only` modes (`linear_probe_inductive_embeddings.py
--methods pome umap --modes combined numeric_only cat_only`), whose embeddings come from
`generate_inductive_embeddings.py --modes`.

`scripts/linear_probe_full_dataset_embeddings.py` is the transductive control: it probes the
full-dataset embeddings of HANCOCK and MIMIC on the same splits. LUAD is left out because its
full-dataset embeddings contain the leaking variables. This control and the OOD, stopping, epoch and
initialization ablations use the logistic regression on the raw (not z-scored) embedding coordinates
(`probe_one(..., standardize=False)`).

The earlier transductive versions of the figures are retained for comparison: `scripts/plot_fig3_linear_probing.ipynb` and `scripts/plot_type_combination_effect.ipynb`, fed by the `src/pome_evaluation/analyze_*_embedding_separability.ipynb` notebooks via `output/supervised/`.

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

For comparing how well the embeddings of POME and its competitors predict held-out target variables, run `scripts/linear_probe_inductive_embeddings.py` (logistic regression, or `--probe mlp`, both on z-scored coordinates; TabICL fit on the raw data with the labels as an upper reference). It probes every dataset and method into one tidy CSV, fitting on each training split and scoring the held-out test split. `scripts/linear_probe_full_dataset_embeddings.py` is the transductive control over the same splits.

### Generate simulated missingness datasets

In order to simulate certain amounts of missingness into the given datasets, simply run the notebook `generate_simulated_missingness.ipynb` with updated paths pointing to the files of the desired dataset. Scripts and notebooks to impute simulated datasets with the different imputation methods can be found in the respective subdirectories under `data/imputation_data/`.

### Analyze imputation results

In order to compute mean absolute errors and multiclass accuracies of imputed values against ground truth values, you can make use of the Python notebook `compute_imputation_results.ipynb`.
