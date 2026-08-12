# Reproducing the imputation figures

Steps to regenerate the three imputation-related figures after the POME numeric
imputation changed (normalized MAE + Mean/Mode baseline). Run everything **from
the project root**. POME steps use conda env `torch`; scoring/plotting use env
`work`. On the Helma cluster both envs are replaced by the Apptainer image —
substitute `container/run.sh <script>` for `conda run -n {torch,work} python
<script>`, and `sbatch container/submit.sh <script>` for the GPU steps (Step 3
below is written in that form; see `container/README.md`).

Target figures:
- `output/imputation_ranks_with_competitors_{16,32,64}.pdf` — competitor ranks (normalized MAE + Mean/Mode)
- `output/imputation_binning_effects_pome.pdf` — POME binning-strategy effects
- `scripts/supplement_imputation_per_epoch.pdf` — POME imputation quality vs. training epochs

## Prerequisites

- **POME must be the current `~/Projects/POME.git`** (regression head is now the
  only continuous-imputation path; the `numeric_imputation` kwarg no longer
  exists). The `torch` env should have it as an editable install
  (`pip install -e ~/Projects/POME.git`). Verify:
  ```bash
  conda run -n torch python -c "import pome, os; print(os.path.dirname(pome.__file__))"
  ```
- Input data present under `data/imputation_groundtruth/`,
  `data/imputation_data/pome_based/{DATASET}/simulated_data/`, and
  `data/input_datasets/`. MIMIC is gitignored — its steps are skipped if absent.

## Code changes already in the repo (no action needed, listed for context)

- `scripts/generate_imputation_results.py` — adds the **Mean/Mode** baseline and
  the normalized-MAE column `nmae_cont`.
- `scripts/generate_imputation_epoch_results.py` — **new** producer for the
  per-epoch figure data; scores both `mae_cont` and `nmae_cont` by reusing
  `generate_imputation_results.py`'s normalization helpers.
- `scripts/plot_supplement_imputation_epochs.ipynb` — now reads the single
  `data/{DATASET}_imputation_per_epoch.csv` per dataset and plots `nmae_cont` in
  its numeric panels. A dataset whose CSV is absent (e.g. gitignored MIMIC) is
  skipped with a message instead of raising; its panels stay empty.
- `src/pome_evaluation/impute_graph_based.py` — dropped the removed
  `numeric_imputation` kwarg and the `_regression` output-dir suffix; `__main__`
  covers the full `{z,nonlinear}×{7,11,15}×{16,32,64}` grid.

---

## Step 0 — (Prerequisite) Regenerate POME imputed data

Only needed if `data/imputation_data/pome_based/{DATASET}/imputed_{disc}_{bins}_{dim}/`
holds stale (pre-change) data. The script **skips files that already exist**, so
**delete the relevant `imputed_*` dirs first if you want them recomputed.**

```bash
conda run -n torch python src/pome_evaluation/impute_graph_based.py
```

- Figure 1 needs only `z` / `bins=15` / all dims.
- Figure 2 needs the full grid (the script's default).
- Trim the `dims` / `bins` / `discretizations` lists at the top of `__main__` to
  scope the run.

GPU-heavy and resumable. Outputs go to
`data/imputation_data/pome_based/{DATASET}/imputed_{disc}_{bins}_{dim}/`.

---

## Step 1 — `imputation_ranks_with_competitors_{16,32,64}.pdf`

```bash
# 1a. Rebuild benchmark CSVs (adds Mean/Mode + nmae_cont). z/bins=15 is all this figure needs.
conda run -n work python scripts/generate_imputation_results.py \
    --discretization z --bins 15 --overwrite

# 1b. Clear stale figure-input caches, or the figure silently reuses old data.
rm -f output/imputation_cont_ranks_dim_*.csv output/imputation_cat_ranks_dim_*.csv \
      output/imputation_metric_distributions_dim_*.csv \
      output/imputation_cont_ranks.csv output/imputation_cat_ranks.csv \
      output/imputation_metric_distributions.csv

# 1c. One run per embedding dim.
for d in 16 32 64; do
  conda run -n work python scripts/generate_imputation_figure.py --dim $d
done
```

**Overwrites:**
- `data/{HANCOCK,TCGA_LUAD,MIMIC}_imputation_z_bins_15.csv`
- `output/imputation_{cont_ranks,cat_ranks,metric_distributions}_dim_{16,32,64}.csv` (caches)
- `output/imputation_ranks_with_competitors_{16,32,64}.pdf`

**Check:** each CSV's `tool` column should list
`AutoComplete, KNN, Mean/Mode, MissForest, POME`, and `nmae_cont` should be
fully populated.

---

## Step 2 — `imputation_binning_effects_pome.pdf`

Needs **all** strategy × bin-count combinations regenerated (requires Step 0 to
have produced the `nonlinear` and `bins=7,11` POME data).

```bash
# 2a. Rebuild every {z,nonlinear} × {7,11,15} benchmark CSV.
conda run -n work python scripts/generate_imputation_results.py --overwrite

# 2b. Render the figure (POME rows only; uses raw mae_cont + acc_cat).
conda run -n work python scripts/generate_imputation_binning_figure.py
```

**Overwrites:**
- all 18 `data/{DATASET}_imputation_{z,nonlinear}_bins_{7,11,15}.csv`
- `output/imputation_binning_effects_pome.pdf`

Fails with `Missing required combinations` if any of the 18 CSVs lacks POME rows
(i.e. Step 0 didn't produce that combination).

---

## Step 3 — `scripts/supplement_imputation_per_epoch.pdf`

```bash
# 3a. Produce per-epoch imputation scores (GPU; resumable; ~150 training runs).
#     One job per dataset so each fits the 24 h h200 walltime limit.
for ds in hancock luad mimic; do
  sbatch --time=24:00:00 --job-name=pome_epochs_$ds \
      container/submit.sh scripts/generate_imputation_epoch_results.py --datasets $ds
done

# 3b. Execute the notebook to render the PDF.
container/run.sh scripts/plot_supplement_imputation_epochs.ipynb
```

- Defaults: snapshot epochs `100 500 1000 1500 2000` (`--epochs ...`),
  dim `32` (`--dim`), `z` / `bins=15`.
- Optional flags: `--datasets hancock luad mimic`, `--dry-run`, `--overwrite`.

**Creates:** `data/{DATASET}_imputation_per_epoch.csv`
(columns: `run, na_ratio, epoch, mae_cont, nmae_cont, acc_cat, dataset`; 250 rows
= 50 masked files × 5 epochs; replaces the old `*_checkpoint*.csv` files).
**Overwrites:** `scripts/supplement_imputation_per_epoch.pdf`.

`nmae_cont` is computed by importing `_load_variable_range` / `_macro_nmae` from
`generate_imputation_results.py`, so it is on exactly the same per-variable
range-normalized scale as Step 1's figure. The notebook's numeric panels (d-f)
plot `nmae_cont`; the categorical panels (a-c) plot `acc_cat`. Keep those two
helpers stable — both analyses depend on them being identical.

**Note:** trains each masked file once and snapshots weights in memory at each
epoch. Observed runtimes on one H200: ~25 min each for HANCOCK and TCGA-LUAD,
~2.5 h for MIMIC. Spot-check the first dataset before letting the full sweep
finish — accuracy should rise with epochs and `nmae_cont` fall slightly.

**Reproducibility caveat:** despite the fixed seed, re-running does *not*
reproduce `mae_cont` bit-for-bit (observed epoch-mean drift up to ~1.9 on
HANCOCK, ~10.8 on TCGA-LUAD), while `acc_cat` is stable to ~0.002. The drift sits
in the continuous regression head and is largest at `na_ratio=0.01`, where a run
holds out only ~50 continuous entries. Do not read small per-run MAE differences
between reruns as a real effect.

---

## Quick reference

| Figure | Data step | Plot step |
|---|---|---|
| `imputation_ranks_with_competitors_{16,32,64}.pdf` | Step 0 (z/15) → 1a | 1b + 1c |
| `imputation_binning_effects_pome.pdf` | Step 0 (full grid) → 2a | 2b |
| `supplement_imputation_per_epoch.pdf` | 3a | 3b |

Environments: **`torch`** = POME model runs (Steps 0, 3a); **`work`** = scoring
and plotting (Steps 1, 2, 3b).
