# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code to reproduce every figure and analysis in the evaluation of **POME**, a
graph-based (GNN) embedder for mixed continuous/categorical tabular clinical data. This
repo does **not** contain the model itself — `pome` is an external package
(`~/Projects/POME.git`), imported only by the GPU scripts. Everything here consumes POME's
embeddings/imputations and turns them into manuscript figures (imputation benchmarks,
unsupervised clustering, linear probing, survival analysis, variable embeddings).

Datasets referenced throughout: **HANCOCK**, **TCGA_LUAD**, **MIMIC** (MIMIC-IV). MIMIC
inputs and derived data are gitignored (see `.gitignore`); scripts skip MIMIC steps when
its files are absent.

## Two-stage architecture: generate → plot

Almost every figure follows the same two-phase pattern, and the two phases run in
**different environments**:

1. **Generate** (`scripts/generate_*.py`, `src/pome_evaluation/impute_graph_based.py`) —
   produce CSVs under `data/` and `output/`. The ones that `import pome`/torch are
   GPU-heavy (`generate_embeddings.py`, `generate_inductive_embeddings.py`,
   `generate_inductive_epoch_snapshots.py`, `generate_imputation_epoch_results.py`,
   `impute_graph_based.py`); the rest only read
   CSVs and are CPU-only.
2. **Plot** (`scripts/plot_*.ipynb`, `scripts/generate_*_figure.py`) — read the generated
   CSVs and render PDFs into `output/` (and `scripts/` for some supplements).

**Stale-cache trap:** `generate_imputation_figure.py` writes intermediate cache CSVs
(`output/imputation_{cont_ranks,cat_ranks,metric_distributions}_dim_*.csv`). If these
already exist the figure silently reuses them — delete them before re-running with new
inputs. This and the full multi-figure recipe are documented in
`REPRODUCE_IMPUTATION_FIGURES.md` (the authoritative runbook for the imputation figures).

## Linear probing: inductive (current) vs transductive (legacy)

Two generations of the probing analysis coexist. **New work uses the inductive path**; the
transductive one is retained for comparison against the published transductive numbers.

- **Inductive (current)** — `generate_inductive_embeddings.py` (GPU for POME, CPU for
  UMAP) fits the encoder on each `data/train_test_splits/<ds>/split_NN_train_*` split and
  transforms the held-out test samples; `linear_probe_inductive_embeddings.py` fits a
  logistic regression on the train embedding and scores the test embedding into the single
  tidy CSV `output/linear_probing/inductive_linear_probing_results.csv`;
  `plot_inductive_linear_probing.py` and `plot_inductive_type_combination.py` draw it.
  `linear_probe_full_dataset_embeddings.py` is the transductive control on the *same*
  splits and reuses the inductive script's `probe_one`/`load_binary_targets` — keep those
  two functions stable, both analyses depend on them being identical.
- **Transductive (legacy)** — the hand-toggled `src/pome_evaluation/analyze_*_embedding_
  separability.ipynb` notebooks (k-fold CV inside one full-dataset embedding, dataset/dim/
  mode edited as literals in the cells, `os.chdir`-based paths) writing
  `output/supervised/*_regression_results_*.csv`, drawn by `plot_fig3_linear_probing.ipynb`
  and `plot_type_combination_effect.ipynb`. Superseded by the inductive scripts above —
  kept for comparison, don't extend these.

**Re-probing traps.** `linear_probe_inductive_embeddings.py` rewrites the whole results CSV
from whatever embeddings it finds, and the repo does not contain everything the committed
CSV was built from:

- **MIMIC** has no inductive embeddings here (its splits are gitignored,
  `output/inductive/*/mimic` does not exist), so a re-probe silently drops all MIMIC rows
  and empties fig-3 panels b/c. Re-merge them from the previous CSV (`git show
  HEAD:output/linear_probing/inductive_linear_probing_results.csv`) instead.
- **POME has only runs 0-4** committed, while older CSVs carry runs 0-9. Probe with
  `--runs 0 1 2 3 4` so UMAP (which has 10) does not outweigh POME; the analysis is
  currently a uniform 5-run one across all datasets.
- Scores do **not** reproduce the older CSV bit-for-bit (max |ΔAP| ~0.008, largest at
  dim 128, which no recent work touched). The image ships sklearn 1.8, which deprecates the
  `penalty=` argument the probe passes — an environment difference, not an embedding one.

**Variable-type modes** (the data-integration analysis: does embedding continuous *and*
categorical variables beat either alone?). `--modes {combined,numeric_only,cat_only}` on
`generate_inductive_embeddings.py` restricts the *existing* split files — a row filter on
POME's graph `type` column, a column filter on the UMAP matrix's `cat_var` list — so no new
splits are needed and all three modes share byte-identical sample partitions. `combined`
keeps the original mode-less output layout; restricted modes add a `{mode}/` path segment.
The probing script tags every row with `mode` and probes all discovered modes by default;
`plot_inductive_linear_probing.py` filters `mode == "combined"`. MIMIC has no categorical
UMAP block, so it has no type-combination arm.

## UMAP must be the NA-aware fork, not umap-learn

All inductive UMAP work requires **`~/Projects/umap-na-aware.git`** (override with
`UMAP_REPO`), *not* the stock umap-learn that ships inside the `.sif`. Two additions are
load-bearing:

- **Pairwise-removal distances** — `parallel_nan_euclidean` / `parallel_nan_hamming`
  compute distances over the features observed in *both* vectors, in `fit()` and in the
  bipartite `transform()`. Triggered by passing `ensure_all_finite="allow-nan"` to
  `fit`/`transform` with metric `euclidean`/`hamming` (it is a **`fit()` argument**, not a
  constructor argument). Both models that get intersected are built this way.
- **`umap.transform_combined(components, combined, X_list, op="intersection", …)`** — the
  `a * b` artifact stores no fit state, so it cannot transform unseen samples on its own.

The fork keeps its package contents at the repo root while importing itself absolutely
(`import umap.distances`), so it cannot be put on `PYTHONPATH` directly.
`container/{run,submit}.sh` materialise a shim dir (`$WS/pythonpath/umap` → `$UMAP_REPO`)
and prepend it, shadowing site-packages; they warn if the fork is missing.
`generate_inductive_embeddings.py` additionally aborts via `import_na_aware_umap()` rather
than silently producing stock-UMAP embeddings.

Symptom of the wrong umap being picked up: `AttributeError: transform_combined` for the
combined mode, or a non-zero `n_test_nonfinite` in the probing results — stock
umap-learn's single-modality `transform()` returns non-finite coordinates for every test
sample carrying a NaN (~50% of HANCOCK, ~80% of TCGA-LUAD), which would silently reduce
those arms to complete cases. With the fork all modes embed the full test split.

## Conda environments (important)

The reproduction workflow uses **two** conda envs, and mixing them up produces confusing
failures:

- **`torch`** — has `pome` as an editable install + torch/PyG. Used only for the GPU
  generate steps (model runs, imputation).
- **`work`** — scoring and plotting (pandas/sklearn/seaborn/lifelines/nbconvert). Used for
  everything else.

Note: `README.md` mentions an env named `hancock_survival` and `environment.yml` names it
`pome_evaluation` — both are stale/aspirational. Follow `REPRODUCE_IMPUTATION_FIGURES.md`
and `container/README.md`, which use `torch`/`work`. On the Helma cluster the envs are
replaced entirely by an Apptainer image (see below).

## Package layout & imports

- `src/pome_evaluation/` — the library: analysis helpers (`imputation_analysis.py`,
  `analyze_unsupervised_clustering.py`, `survival_*.py`) and plotting helpers
  (`imputation_plotting.py`, `survival_plotting.py`). Some files here are *also* runnable
  standalone analysis scripts/notebooks (e.g. `impute_graph_based.py`, the `embed_UMAP_*`
  scripts, the `analyze_*.ipynb` notebooks).
- `scripts/` — CLI entrypoints. Each does `sys.path.insert(0, .../src)` at import time to
  find `pome_evaluation`, so **there is no installed package** — `pip install -e .` from
  the README will not work (no `pyproject.toml`/`setup.py` exists). Run scripts directly
  from the repo root.
- All path handling is relative to the repo root via `PROJECT_ROOT = Path(__file__)...`.
  **Always run from the repo root.** A few older `embed_UMAP_*.py` scripts still contain
  hardcoded `os.chdir(...)` / absolute paths (`/home/wollerf/...`) — fix those before use.

## Common commands

```bash
# Imputation figure for one embedding dim (CPU)
python scripts/generate_imputation_figure.py --dim 32

# Survival figure (CPU)
python scripts/generate_survival_figure.py

# Benchmark CSVs feeding the imputation figures (CPU; --overwrite forces recompute)
python scripts/generate_imputation_results.py --discretization z --bins 15 --overwrite

# GPU generate steps (need conda env `torch` with pome installed)
conda run -n torch python scripts/generate_embeddings.py --datasets hancock --dims 32
conda run -n torch python src/pome_evaluation/impute_graph_based.py

# Data-integration (type-combination) figure, inductive. UMAP is CPU-only, POME needs GPU.
# --runs 5 matters: POME only has runs 0-4, and the probe is run with --runs 0 1 2 3 4,
# so the default of 10 UMAP runs would compute ~79 MB that nothing ever reads.
python scripts/generate_inductive_embeddings.py \
    --methods umap --datasets hancock luad --modes numeric_only cat_only --dims 64 --runs 5
container/submit_pome_modes.sh --dry-run   # then drop --dry-run: 8 sharded GPU jobs
container/run.sh scripts/merge_tuned_epochs.py   # after ALL shards finish
python scripts/linear_probe_inductive_embeddings.py          # probes all modes found
python scripts/plot_inductive_type_combination.py --dim 64

# Execute a plotting notebook headless (re-runs all cells in place)
jupyter nbconvert --to notebook --execute --inplace scripts/plot_fig4_unsupervised.ipynb
```

Most `generate_*.py` scripts share flags: `--datasets`, `--dims`, `--overwrite`,
`--dry-run`, and skip work whose output already exists (resumable). There is **no test
suite, linter, or build step** — this is figure-reproduction code.

## Running on the Helma cluster (NHR@FAU)

`container/` holds an Apptainer + Slurm workflow that replaces the conda envs with one
`.sif` image; the `POME_Evaluation`, `POME` and NA-aware-`umap` sources are all
bind-mounted (editing code never needs a rebuild). See `container/README.md` /
`container/COMMANDS.md`. Key points:

- `container/run.sh <script.py|notebook.ipynb> [args]` — CPU work on the current node (the
  common case). Add `--gpu` only inside a GPU allocation.
- `container/submit.sh <script.py>` — sbatch the GPU scripts. **Always H200, never H100**
  (the project has 0 H100 budget → H100 jobs pend forever).
- `container/submit_pome_modes.sh` — fans the inductive POME generation out over one job
  per (dataset × mode × split-group). Needed because POME's inductive epoch tuning runs a
  sample-holdout CV inside *every* fit, so one job easily exceeds the 6 h walltime. Each
  shard gets a unique `--log-tag`; `record_tuned_epochs()` rewrites the whole log on every
  run, so **concurrent jobs sharing one `tuned_epochs.csv` would silently drop rows**. Run
  `scripts/merge_tuned_epochs.py` once all shards finish.
- `STAGE_OUT="<repo-relative dir>"` redirects a job's small-file write storm to node-local
  NVMe and copies back at the end — use for scripts that emit many files
  (`impute_graph_based.py`, `generate_inductive_embeddings.py`). Note it hides the dir's
  existing contents from the job, so resume/skip sees nothing and the copy-back replaces
  files like `output/inductive/pome/tuned_epochs.csv` with only that job's rows.
- Both wrappers set `PYTHONPATH` to `$WS/pythonpath` (the `umap` fork shim) `:$REPO/src`
  `:$POME_REPO/src`. Anything invoking UMAP outside these wrappers must reproduce that or
  it silently gets stock umap-learn — see the fork section above.
