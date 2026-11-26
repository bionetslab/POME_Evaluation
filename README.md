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

**Competitors (HI-VAE, SAINT, missForest + autoencoder)** →
`output/linear_probing/inductive_linear_probing_results_pome14_competitors.pdf`. These are the
tabular competitors of the clustering analysis, run inductively on the same train/test splits
(the POME-format split files, so they see exactly the data POME sees). Each one is fit on the
training split only and embeds the held-out test samples with the frozen model. Training and
architecture are the same as in the clustering versions, at dims 16/32/64 with runs 0-4.
Outputs go to `output/inductive_competitors/{hivae,saint,mfae}/` in the inductive layout.
- `scripts/embed_hivae_inductive.py` (`hivae.sif`, CPU): the training embedding comes from the
  authors' restore pass over the training split. The test embedding comes from one more restore
  pass of the frozen network over train + test, keeping the test rows. The authors' code re-codes
  categories per input file and normalizes per batch, so a test-only pass would be wrong.
- `scripts/embed_saint_inductive.py` (`saint.sif`, GPU): label encoding, mean filling and
  standardization use training statistics only. Test categories never seen in training go to
  SAINT's mask token. Test samples are encoded in their own batches, so they never attend to
  training samples.
- `scripts/embed_mfae_inductive.py` (CPU): missForest is fit on the training split and then
  imputes train + test with the fitted model (cached in `_mfae_imputed/`). The design matrix
  uses training z-score statistics and training one-hot categories. The autoencoder is trained
  on the training rows only.
```bash
WS=$(ws_find pome_eval)
sbatch --job-name=ind_hivae --export=POME_EVAL_SIF=$WS/containers/hivae.sif \
    container/submit.sh scripts/embed_hivae_inductive.py --jobs 16
sbatch --job-name=ind_saint --export=POME_EVAL_SIF=$WS/containers/saint.sif,PYTHONHASHSEED=0 \
    container/submit.sh scripts/embed_saint_inductive.py --datasets hancock luad   # + mimic, per --dims
sbatch --job-name=ind_mfae container/submit.sh scripts/embed_mfae_inductive.py
container/run.sh scripts/linear_probe_inductive_embeddings.py --methods hivae saint mfae \
    --modes combined --dims 16 32 64 --runs 0 1 2 3 4 --emb-root output/inductive_competitors \
    --output output/linear_probing/inductive_linear_probing_results_competitors.csv
container/run.sh scripts/merge_probing_variant.py \
    --pome output/linear_probing/inductive_linear_probing_results_pome14_pome.csv \
    --competitors output/linear_probing/inductive_linear_probing_results_competitors.csv \
    --output output/linear_probing/inductive_linear_probing_results_pome14_competitors.csv
container/run.sh scripts/plot_inductive_linear_probing.py --dims 16 32 64 \
    --results output/linear_probing/inductive_linear_probing_results_pome14_competitors.csv \
    --output output/linear_probing/inductive_linear_probing_results_pome14_competitors.pdf
```
`plot_inductive_linear_probing.py --methods POME UMAP` on the same CSV reproduces the
two-method figure.

**Leakage-free TCGA-LUAD** → `output/linear_probing/inductive_linear_probing_results_pome14_competitors_noleak.pdf`
(and `..._pome14_noleak.pdf` for POME vs UMAP only). Four LUAD inputs record the outcome or the
follow-up after diagnosis ("New Neoplasm Event Post Initial Therapy Indicator", single-variable
AUC 0.93 for Disease Free Status; "Person Neoplasm Cancer Status"; and two follow-up-time
variables). `scripts/make_luad_noleak_splits.py` writes the same splits without them as dataset
`luad_noleak`, which every inductive script accepts. Outputs land in their own `luad_noleak`
directories, so the original LUAD results stay untouched.
```bash
container/run.sh scripts/make_luad_noleak_splits.py
sbatch --export=PYTHONHASHSEED=0 container/submit.sh scripts/generate_inductive_embeddings.py \
    --methods pome --modes combined --datasets luad_noleak --dims 16 --runs 5 --log-tag luad_noleak_d16  # per dim
container/run.sh scripts/generate_inductive_embeddings.py --methods umap --modes combined \
    --datasets luad_noleak --dims 16 32 64 --runs 5
# competitors: the three embed_*_inductive.py jobs above with --datasets luad_noleak
# probe each method family on luad_noleak (--datasets luad_noleak, emb roots as above) into
# *_pome14_pome_luad_noleak.csv / *_umap_luad_noleak.csv / *_competitors_luad_noleak.csv, then:
container/run.sh scripts/merge_probing_variant.py \
    --pome output/linear_probing/inductive_linear_probing_results_pome14_pome.csv \
           output/linear_probing/inductive_linear_probing_results_pome14_pome_luad_noleak.csv \
    --competitors output/linear_probing/inductive_linear_probing_results_competitors.csv \
                  output/linear_probing/inductive_linear_probing_results_umap_luad_noleak.csv \
                  output/linear_probing/inductive_linear_probing_results_competitors_luad_noleak.csv \
    --output output/linear_probing/inductive_linear_probing_results_pome14_competitors_noleak.csv
container/run.sh scripts/plot_inductive_linear_probing.py --dims 16 32 64 --luad-dataset luad_noleak \
    --results output/linear_probing/inductive_linear_probing_results_pome14_competitors_noleak.csv \
    --output output/linear_probing/inductive_linear_probing_results_pome14_competitors_noleak.pdf
```

**Tabular foundation model (TabICL)** → `output/linear_probing/inductive_linear_probing_results_pome14_competitors_tabicl_noleak.pdf`.
Same as the figure above, plus TabICL (`tabicl` 2.2.0, checkpoint `tabicl-classifier-v2-20260212.ckpt`).
TabICL is used instead of TabPFN because TabPFN ≥ 2.5 requires a Prior Labs license login
before its weights download. TabICLv2's column embedding is target-aware, so
`scripts/embed_tabicl_inductive.py` builds one embedding per probing target. Each embedding is
the final in-context hidden state of a query row (512-d). Test rows use the labelled training
split as context. Training rows are embedded out-of-fold (5-fold), so a row never sees its own
label. PCA is fit on train and reduces the embedding to 16/32/64. Each run is one ensemble member
with a run-seeded feature order. Because it sees the probed label, TabICL is a **supervised
reference**, and the figure labels it "TabICL (supervised)". The package (with einops and
huggingface_hub) is installed with `pip install --no-deps --target $(ws_find pome_eval)/pylib`,
and the checkpoint is stored in `$(ws_find pome_eval)/models/tabicl/`. Outputs are written to
`output/inductive_competitors/tabicl/<dataset>/target_<target>/split_NN/dim_D/`. The probing
script probes each `target_<target>` directory for its own target only. Panel a counts wins
among the label-free methods only, so TabICL has no bar there. The same comparison is also run
with standardized logistic regression and the MLP, XGBoost, RBF-SVM and k-NN probes (see "Further
non-linear probes" below):
`output/linear_probing/inductive_{logreg_std,mlp,xgboost,svm,knn}_probing_results_pome14_competitors_tabicl_noleak.pdf`.

The same figures also show a **label-free TabICL** ("TabICL", method `tabicl_lf`), which competes
in panel a. `scripts/embed_tabicl_labelfree_inductive.py` uses the v1.1 checkpoint
`tabicl-classifier-v1.1-20250506.ckpt`, whose column embedding is not target-aware. The
embedding is the output of TabICL's row-wise interaction stage (512-d), taken before the
in-context stage that uses labels. One forward pass per split and run embeds the training and
test rows together. The column statistics come from the training rows only. The sklearn wrapper
needs labels for `fit`, so a dummy label vector is passed; it only reaches the discarded
in-context stage. The script checks that a second, different dummy vector gives identical
embeddings. PCA is fit on train and reduces the embedding to 16/32/64. Each run is one ensemble
member with a run-seeded feature order. One embedding per split serves all targets, in the
standard layout `output/inductive_competitors/tabicl_lf/<dataset>/split_NN/dim_D/`.
```bash
sbatch --job-name=ind_tabicl container/submit.sh scripts/embed_tabicl_inductive.py   # hancock mimic luad_noleak
sbatch --job-name=ind_tabicl_lf container/submit.sh scripts/embed_tabicl_labelfree_inductive.py
for M in tabicl tabicl_lf; do
    container/run.sh scripts/linear_probe_inductive_embeddings.py --methods $M --modes combined \
        --datasets hancock mimic luad_noleak --dims 16 32 64 --emb-root output/inductive_competitors \
        --output output/linear_probing/inductive_linear_probing_results_$M.csv
done
container/run.sh scripts/merge_probing_variant.py \
    --pome output/linear_probing/inductive_linear_probing_results_pome14_pome.csv \
           output/linear_probing/inductive_linear_probing_results_pome14_pome_luad_noleak.csv \
    --competitors output/linear_probing/inductive_linear_probing_results_competitors.csv \
                  output/linear_probing/inductive_linear_probing_results_umap_luad_noleak.csv \
                  output/linear_probing/inductive_linear_probing_results_competitors_luad_noleak.csv \
                  output/linear_probing/inductive_linear_probing_results_tabicl.csv \
                  output/linear_probing/inductive_linear_probing_results_tabicl_lf.csv \
    --output output/linear_probing/inductive_linear_probing_results_pome14_competitors_tabicl_noleak.csv
container/run.sh scripts/plot_inductive_linear_probing.py --dims 16 32 64 --luad-dataset luad_noleak \
    --results output/linear_probing/inductive_linear_probing_results_pome14_competitors_tabicl_noleak.csv \
    --output output/linear_probing/inductive_linear_probing_results_pome14_competitors_tabicl_noleak.pdf
# standardized logistic regression (logreg_std), MLP, XGBoost, RBF-SVM and k-NN; the other files come from the
# "Further non-linear probes" block below
for PR in "logreg_std:--probe logreg --standardize" "mlp:--probe mlp" "xgboost:--probe xgboost" \
          "svm:--probe svm" "knn:--probe knn"; do
    X=output/linear_probing/inductive_${PR%%:*}_probing_results
    for M in tabicl tabicl_lf; do
        container/run.sh scripts/linear_probe_inductive_embeddings.py ${PR#*:} --jobs 24 --methods $M \
            --modes combined --datasets hancock mimic luad_noleak --dims 16 32 64 \
            --emb-root output/inductive_competitors --output ${X}_$M.csv
    done
    container/run.sh scripts/merge_probing_variant.py --reference none --pome ${X}_pome14_pome.csv \
        --competitors ${X}_umap.csv ${X}_competitors.csv ${X}_tabicl.csv ${X}_tabicl_lf.csv \
        --output ${X}_pome14_competitors_tabicl.csv
    container/run.sh scripts/plot_inductive_linear_probing.py --dims 16 32 64 --luad-dataset luad_noleak \
        --results ${X}_pome14_competitors_tabicl.csv --output ${X}_pome14_competitors_tabicl_noleak.pdf
done
```

**Non-linear probing (XGBoost)** → `output/linear_probing/inductive_xgboost_probing_results_pome14_competitors.pdf`
(and `..._noleak.pdf` with the leakage-free LUAD). Same embeddings, splits, targets, metrics and
layout as the logistic-regression figures, but the probe is an XGBoost classifier
(`linear_probe_inductive_embeddings.py --probe xgboost`: 300 trees, depth 3, learning rate 0.05,
subsampling 0.8, fixed seed, no tuning). The published results file holds logistic-regression
scores only, so every method is probed afresh, including UMAP (`--reference none`). The MIMIC
UMAP embeddings were never stored and are regenerated first. xgboost is not in the image built
earlier: it is installed into `$(ws_find pome_eval)/pylib`, which `run.sh`/`submit.sh` put on
`PYTHONPATH`.
```bash
WS=$(ws_find pome_eval)
apptainer exec $WS/containers/pome_eval.sif pip install --no-deps --target $WS/pylib xgboost==2.1.3
sbatch --export=PYTHONHASHSEED=0 container/submit.sh scripts/generate_inductive_embeddings.py \
    --methods umap --modes combined --datasets mimic --dims 16 --runs 5   # per dim
X=output/linear_probing/inductive_xgboost_probing_results
P="scripts/linear_probe_inductive_embeddings.py --probe xgboost --jobs 48 --modes combined --dims 16 32 64 --runs 0 1 2 3 4"
container/run.sh $P --methods pome --emb-root output/inductive_pome14 --output ${X}_pome14_pome.csv
container/run.sh $P --methods hivae saint mfae --emb-root output/inductive_competitors --output ${X}_competitors.csv
container/run.sh $P --methods umap --output ${X}_umap.csv
container/run.sh scripts/merge_probing_variant.py --reference none --pome ${X}_pome14_pome.csv \
    --competitors ${X}_umap.csv ${X}_competitors.csv --output ${X}_pome14_competitors.csv
container/run.sh scripts/plot_inductive_linear_probing.py --dims 16 32 64 \
    --results ${X}_pome14_competitors.csv --output ${X}_pome14_competitors.pdf
container/run.sh scripts/plot_inductive_linear_probing.py --dims 16 32 64 --luad-dataset luad_noleak \
    --results ${X}_pome14_competitors.csv --output ${X}_pome14_competitors_noleak.pdf
```

**Further non-linear probes (RBF-SVM, k-NN, MLP) and standardized logistic regression** →
`output/linear_probing/inductive_{svm,knn,mlp,logreg_std}_probing_results_pome14_competitors{,_noleak}.pdf`.
`--probe svm|knn|mlp` tune their hyperparameters by inner 3-fold CV on the training split (see the
script docstring). `logreg_std` is the headline probe with `--standardize`, i.e. with features
z-scored on the training split. Same pipeline as the XGBoost block above, with `--probe <probe>`
(and `--probe logreg --standardize` for `logreg_std`) and `X=output/linear_probing/inductive_<probe>_probing_results`.

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
