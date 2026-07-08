#!/usr/bin/env python
"""Compute imputation-accuracy result tables for POME and the imputation baselines.

For each (dataset, discretization, bins) combination this script walks the
copy-masked ground-truth files under ``data/imputation_groundtruth/{DATASET}``,
looks up the corresponding imputed values produced by each tool, and scores them
against the held-out ground truth. It reproduces
``src/pome_evaluation/compute_imputation_results.ipynb`` as a resumable CLI.

Scored tools:
- **KNN**, **AutoComplete**, **MissForest** baselines (imputed once per
  masked file, so their rows are dim-independent, ``dim=-1``).
- **POME** for every requested embedding dimension, read from
  ``imputed_{discretization}_{bins}_{dim}`` directories.

Metrics: MAE, MAPE (``mre_cont``) and per-variable IQR-normalized MAE
(``nmae_cont``) on the continuous entries and accuracy on the categorical
entries. ``nmae_cont`` divides each continuous variable's MAE by that variable's
interquartile range (from the original unmasked data) and macro-averages across
variables, so every variable contributes on a common, outlier-robust scale with
equal weight. One CSV is written per (dataset, discretization, bins):
``data/{DATASET}_imputation_{discretization}_bins_{bins}.csv``, holding rows for
every tool / dim / masked file. Any tool whose imputed files are missing is
skipped with a warning (e.g. MIMIC baselines when that data is not present
locally), so partial reruns still produce POME rows.

Run from the project root (conda env ``torch`` is not required — this only reads
CSVs and scikit-learn metrics):

    python scripts/generate_imputation_results.py
    python scripts/generate_imputation_results.py --datasets hancock --bins 15 --discretization z
    python scripts/generate_imputation_results.py --dims 16 32 --overwrite
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GROUNDTRUTH_ROOT = DATA_ROOT / "imputation_groundtruth"
IMPUTED_ROOT = DATA_ROOT / "imputation_data"
INPUT_ROOT = DATA_ROOT / "input_datasets"

# The mask sentinel written by the missingness-simulation pipeline; excluded when
# computing per-variable statistics from the original data.
NA_SENTINEL = -99.0

# label -> original unmasked sample-format CSV (samples x variables) used to
# derive per-variable IQR. Naming is irregular across datasets, so it is explicit.
INPUT_FILES = {
    "HANCOCK": "hancock_with_targets.csv",
    "TCGA_LUAD": "TCGA_LUAD_with_targets.csv",
    "MIMIC": "mimic_with_targets_patientIDs.csv",
}

# label -> {variable: IQR} computed once from the original unmasked data.
_IQR_CACHE: dict = {}

# dataset CLI key -> on-disk dataset label (directory / output-file name).
DATASETS = {
    "hancock": "HANCOCK",
    "mimic": "MIMIC",
    "luad": "TCGA_LUAD",
}

DEFAULT_BINS = (7, 11, 15)
DEFAULT_DISCRETIZATIONS = ("z", "nonlinear")
DEFAULT_DIMS = (16, 32, 64)


def _empty_results() -> dict:
    return {
        "run": [],
        "na_ratio": [],
        "tool": [],
        "mae_cont": [],
        "nmae_cont": [],
        "acc_cat": [],
        "mre_cont": [],
        "dim": [],
        "bins": [],
        "discretization": [],
    }


def save_result_to_dict(results_dict, num_run, na_ratio, tool_name, mae_cont,
                        acc_cat, mre_cont, nmae_cont=np.nan, dim=-1, bins=-1,
                        discretization=""):
    results_dict["run"].append(num_run)
    results_dict["na_ratio"].append(na_ratio)
    results_dict["tool"].append(tool_name)
    results_dict["mae_cont"].append(mae_cont)
    results_dict["nmae_cont"].append(nmae_cont)
    results_dict["acc_cat"].append(acc_cat)
    results_dict["mre_cont"].append(mre_cont)
    results_dict["dim"].append(dim)
    results_dict["bins"].append(bins)
    results_dict["discretization"].append(discretization)
    return results_dict


def _load_sample_by_variable(path: Path, ac: bool = False):
    """Load a baseline imputed file (rows = samples, columns = variables)."""
    if not path.exists():
        return None
    if ac:
        df = pd.read_csv(path)
        df.set_index("ID", inplace=True)
        return df
    return pd.read_csv(path, sep="\t", index_col=0)


def _load_variable_by_sample(path: Path):
    """Load a POME imputed file (rows = variables, columns = samples)."""
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t", index_col=0)


def _load_variable_iqr(label: str) -> dict:
    """Return ``{variable: IQR}`` from the original unmasked data for ``label``.

    Reads the sample-format ``{stem}_with_targets.csv`` (rows = samples, columns =
    variables), excluding ``NaN`` and the ``NA_SENTINEL`` mask value, and computes
    ``Q3 - Q1`` per column. Used to per-variable normalize the continuous MAE. The
    result is cached per label. Returns ``{}`` (with a warning) when the input
    file is absent (e.g. gitignored MIMIC), so scoring degrades to ``NaN`` rather
    than crashing.
    """
    if label in _IQR_CACHE:
        return _IQR_CACHE[label]

    candidates = []
    if label in INPUT_FILES:
        candidates.append(INPUT_ROOT / INPUT_FILES[label])
    candidates.append(INPUT_ROOT / f"{label}_with_targets.csv")
    candidates.append(INPUT_ROOT / f"{label.lower()}_with_targets.csv")
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        print(f"    [warn] no input dataset for {label}; nmae_cont will be NaN")
        _IQR_CACHE[label] = {}
        return _IQR_CACHE[label]

    df = pd.read_csv(path, index_col=0)
    df = df.mask(df == NA_SENTINEL)
    quantiles = df.quantile([0.25, 0.75], numeric_only=True)
    iqr = (quantiles.loc[0.75] - quantiles.loc[0.25]).to_dict()
    _IQR_CACHE[label] = iqr
    return iqr


def _macro_nmae(gt_cont: np.ndarray, pred_cont: np.ndarray,
                cont_vars: np.ndarray, scales: dict) -> float:
    """Per-variable IQR-normalized, macro-averaged MAE over continuous entries.

    Computes each variable's MAE, divides by that variable's IQR, then averages
    across variables (equal weight per variable). Variables with a missing or
    non-positive IQR are skipped; returns ``np.nan`` if none remain.
    """
    if len(gt_cont) == 0:
        return np.nan
    abs_err = np.abs(gt_cont - pred_cont)
    nmae_terms = []
    for variable in np.unique(cont_vars):
        scale = scales.get(variable)
        if scale is None or not np.isfinite(scale) or scale <= 0:
            continue
        mae_v = np.mean(abs_err[cont_vars == variable])
        nmae_terms.append(mae_v / scale)
    return float(np.mean(nmae_terms)) if nmae_terms else np.nan


def process_groundtruth_file(gt_path: Path, label: str, dims, bins: int,
                             discretization: str, results: dict,
                             regression: bool = False) -> None:
    """Score every available tool for one copy-masked ground-truth file."""
    with open(gt_path, "rb") as f:
        gt_dict = pickle.load(f)

    # masked_values_{na_ratio}_{run}.pkl -> the imputed files use a float run.
    base = gt_path.name.rsplit(".", 1)[0]
    parts = base.split("_")
    parts[-1] = str(float(parts[-1]))
    tsv_name = "_".join(parts) + ".tsv"
    csv_name = "_".join(parts) + ".csv"
    num_run = parts[-1]
    na_ratio = parts[2]

    # --- Assemble the predictors that actually have files on disk -----------
    # Each predictor: (tool_name, lookup(sample, variable), round_cat, dim).
    predictors = []

    knn_df = _load_sample_by_variable(
        IMPUTED_ROOT / "knn" / label / "imputed_data" / tsv_name)
    if knn_df is not None:
        predictors.append(("KNN", lambda s, v, d=knn_df: d.loc[s, v], True, -1))

    ac_df = _load_sample_by_variable(
        IMPUTED_ROOT / "autocomplete" / label / "imputed_data" / csv_name, ac=True)
    if ac_df is not None:
        predictors.append(
            ("AutoComplete", lambda s, v, d=ac_df: d.loc[s, v], True, -1))

    miss_df = _load_sample_by_variable(
        IMPUTED_ROOT / "missforest" / label / "imputed_data" / tsv_name)
    if miss_df is not None:
        predictors.append(
            ("MissForest", lambda s, v, d=miss_df: d.loc[s, v], True, -1))

    suffix = "_regression" if regression else ""
    for dim in dims:
        graph_dir = f"imputed_{discretization}_{bins}_{dim}{suffix}"
        graph_df = _load_variable_by_sample(
            IMPUTED_ROOT / "pome_based" / label / graph_dir / tsv_name)
        if graph_df is not None:
            predictors.append(
                ("POME", lambda s, v, d=graph_df: d.loc[v, s], False, dim))

    if not predictors:
        print(f"    [skip] {gt_path.name}: no imputed files found")
        return

    # --- Collect ground-truth and predicted values -------------------------
    gt_cont, gt_cat = [], []
    # variable name per continuous entry, aligned with gt_cont and each tool's
    # collected[idx]["cont"]; used to per-variable normalize the MAE.
    cont_vars = []
    # tool index -> {"cont": [...], "cat": [...]}
    collected = [{"cont": [], "cat": []} for _ in predictors]
    for pos, value in gt_dict.items():
        sample, variable = pos[0], pos[1]
        gt_value, gt_type = value[0], value[1]
        if gt_type == "cont":
            gt_cont.append(gt_value)
            cont_vars.append(variable)
        else:
            gt_cat.append(gt_value)
        for idx, (_name, lookup, round_cat, _dim) in enumerate(predictors):
            pred = lookup(sample, variable)
            if gt_type == "cont":
                collected[idx]["cont"].append(pred)
            else:
                collected[idx]["cat"].append(np.round(pred) if round_cat else pred)

    gt_cont_array = np.array(gt_cont, dtype=float)
    cont_var_array = np.array(cont_vars)
    scales = _load_variable_iqr(label)

    # --- Score each predictor ----------------------------------------------
    for idx, (name, _lookup, _round_cat, dim) in enumerate(predictors):
        cont = collected[idx]["cont"]
        cat = collected[idx]["cat"]

        # AutoComplete may emit NaNs for values it could not impute: penalise
        # continuous misses with the worst observed error and flag categoricals.
        if name == "AutoComplete":
            cont_array = np.array(cont, dtype=float)
            cat_array = np.array(cat, dtype=float)
            finite = ~np.isnan(cont_array) & ~np.isnan(gt_cont_array)
            if np.isnan(cont_array).sum() > 0 or np.isnan(cat_array).sum() > 0:
                max_difference = np.max(
                    np.abs(cont_array[finite] - gt_cont_array[finite]))
                nan_cont = np.isnan(cont_array)
                cont_array[nan_cont] = gt_cont_array[nan_cont] + max_difference
                cat_array[np.isnan(cat_array)] = -1
                cont = list(cont_array)
                cat = list(cat_array)

        mae_cont = mean_absolute_error(gt_cont, cont)
        mre_cont = mean_absolute_percentage_error(gt_cont, cont)
        acc_cat = accuracy_score(gt_cat, cat)
        nmae_cont = _macro_nmae(gt_cont_array, np.array(cont, dtype=float),
                                cont_var_array, scales)
        save_result_to_dict(
            results_dict=results,
            num_run=num_run,
            na_ratio=na_ratio,
            tool_name=name,
            mae_cont=mae_cont,
            nmae_cont=nmae_cont,
            acc_cat=acc_cat,
            mre_cont=mre_cont,
            dim=dim if name == "POME" else -1,
            bins=bins if name == "POME" else -1,
            discretization=discretization if name == "POME" else "",
        )


def build_dataset_results(label: str, dims, bins: int, discretization: str,
                          regression: bool = False):
    """Return the imputation-error DataFrame for one (dataset, disc, bins)."""
    groundtruth_dir = GROUNDTRUTH_ROOT / label
    if not groundtruth_dir.is_dir():
        print(f"  [skip] {label}: no ground-truth directory {groundtruth_dir}")
        return None

    results = _empty_results()
    gt_files = sorted(os.listdir(groundtruth_dir))
    for gt_file in gt_files:
        gt_path = groundtruth_dir / gt_file
        if gt_path.suffix != ".pkl":
            continue
        process_groundtruth_file(
            gt_path, label, dims, bins, discretization, results, regression)

    df = pd.DataFrame(results)
    if df.empty:
        return None
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS),
                        default=list(DATASETS),
                        help="datasets to score (default: all three)")
    parser.add_argument("--bins", nargs="+", type=int, default=list(DEFAULT_BINS),
                        help=f"bin counts (default: {list(DEFAULT_BINS)})")
    parser.add_argument("--discretization", nargs="+",
                        default=list(DEFAULT_DISCRETIZATIONS),
                        help="POME discretization strategies "
                             f"(default: {list(DEFAULT_DISCRETIZATIONS)})")
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS),
                        help=f"POME embedding dimensions (default: {list(DEFAULT_DIMS)})")
    parser.add_argument("--regression", action="store_true",
                        help="use POME regression-mode imputed files "
                             "(imputed_{disc}_{bins}_{dim}_regression), and write "
                             "to {DATASET}_imputation_{disc}_bins_{bins}_regression.csv")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute and overwrite existing result CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the (dataset, disc, bins) tables to build and exit")
    args = parser.parse_args()

    print(f"Ground-truth root: {GROUNDTRUTH_ROOT}")
    print(f"Imputed data root: {IMPUTED_ROOT}")
    print(f"Datasets: {args.datasets} | discretization: {args.discretization} | "
          f"bins: {args.bins} | dims: {args.dims} | regression: {args.regression}")

    out_suffix = "_regression" if args.regression else ""
    for key in args.datasets:
        label = DATASETS[key]
        for discretization in args.discretization:
            for bins in args.bins:
                out_path = DATA_ROOT / \
                    f"{label}_imputation_{discretization}_bins_{bins}{out_suffix}.csv"
                if args.dry_run:
                    print(f"  [plan] {out_path.name}")
                    continue
                if out_path.exists() and not args.overwrite:
                    print(f"  [skip] {out_path.name} (exists)")
                    continue

                print(f"  [build] {label} | {discretization} | bins={bins}")
                df = build_dataset_results(
                    label, args.dims, bins, discretization, args.regression)
                if df is None:
                    print(f"    [warn] no results for {out_path.name}, not written")
                    continue
                df.to_csv(out_path, index=True)
                print(f"    [write] {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
