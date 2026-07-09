#!/usr/bin/env python
"""Fill the residual NaNs left in AutoComplete's imputed files.

AutoComplete leaves a fraction of masked cells unimputed (it emits ``NaN`` for
them, chiefly high-cardinality one-hot categoricals and some continuous
variables). ``scripts/generate_imputation_results.py`` currently penalises those
misses with the worst observed error, which inflates AutoComplete's reported
error. To score AutoComplete on equal footing with a trivial fallback imputer,
this script post-imputes those residual NaNs:

- **Continuous** variables are filled with the **mean** of that variable's
  observed values.
- **Categorical** variables are filled with the **majority class** among that
  variable's observed values (ties broken toward the smallest class code).

The observed values are read per (dataset, masked file) from the graph-format
inputs under ``data/imputation_data/autocomplete/<DATASET>/simulated_data`` —
variables x samples, with a ``type`` column of ``cont``/``cat``. In that format
masked-out entries are written as a per-dataset sentinel (e.g. ``-99999`` for
HANCOCK, ``-99`` for TCGA-LUAD); the sentinel is auto-detected as the negative
value that appears in categorical rows (categorical class codes are always
non-negative), and those cells are excluded when computing the mean/majority.

Filled files are written, with the same layout (samples x variables plus the
``ID`` column), to ``.../<DATASET>/<output-subdir>`` (default
``imputed_data_postimputed``), leaving the original ``imputed_data`` untouched.
To make ``generate_imputation_results.py`` consume them, either re-run with
``--output-subdir imputed_data`` or repoint that script's AutoComplete path.

MIMIC is excluded by default: its AutoComplete output has no residual NaNs and
its ``simulated_data`` uses a different (sample-format) layout.

Run from the project root:

    python scripts/postimpute_autocomplete_na.py
    python scripts/postimpute_autocomplete_na.py --datasets hancock
    python scripts/postimpute_autocomplete_na.py --output-subdir imputed_data  # in place
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTOCOMPLETE_ROOT = PROJECT_ROOT / "data" / "imputation_data" / "autocomplete"

# CLI key -> on-disk dataset label (directory name). MIMIC is intentionally
# absent from the default set (no residual NaNs, different simulated layout).
DATASETS = {
    "hancock": "HANCOCK",
    "luad": "TCGA_LUAD",
    "mimic": "MIMIC",
}
DEFAULT_DATASETS = ("hancock", "luad")

# Non-variable column in the imputed sample-format CSVs.
ID_COLUMN = "ID"
# Column in the simulated graph-format TSVs flagging continuous vs categorical.
TYPE_COLUMN = "type"


def _detect_sentinel(numeric_body: pd.DataFrame, types: pd.Series):
    """Auto-detect the mask sentinel from the categorical rows.

    Categorical variables are encoded as non-negative integer class codes, so any
    negative value in a categorical row is the mask sentinel. Returns the set of
    detected sentinel values (normally a single value); empty if none are found.
    """
    cat_vars = types[types == "cat"].index
    cat_vars = [v for v in cat_vars if v in numeric_body.index]
    if not cat_vars:
        return set()
    cat_values = numeric_body.loc[cat_vars].to_numpy()
    negatives = cat_values[np.isfinite(cat_values) & (cat_values < 0)]
    return set(np.unique(negatives).tolist())


def _observed_values(row: pd.Series, sentinels: set) -> np.ndarray:
    """Return a variable's observed values: finite, non-sentinel entries."""
    values = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    for sentinel in sentinels:
        mask &= values != sentinel
    return values[mask]


def _majority_class(observed: np.ndarray) -> float:
    """Most frequent observed class code; ties broken toward the smallest code."""
    counts = pd.Series(observed).value_counts()
    top = counts[counts == counts.max()].index
    return float(min(top))


def process_file(imputed_path: Path, simulated_path: Path, out_path: Path,
                 verbose: bool = False) -> dict:
    """Post-impute one AutoComplete file. Returns a per-file stats dict."""
    # Read the imputed cells as raw strings (empty -> NaN) so every value
    # AutoComplete already produced is written back byte-for-byte; only the NaN
    # cells are substituted. Round-tripping through float would perturb the last
    # digit of untouched values.
    imputed = pd.read_csv(imputed_path, dtype=str, keep_default_na=True,
                          na_values=[""])
    simulated = pd.read_csv(simulated_path, sep="\t", index_col=0)

    if TYPE_COLUMN not in simulated.columns:
        raise ValueError(
            f"{simulated_path.name}: no '{TYPE_COLUMN}' column "
            "(unexpected simulated-data layout)")
    types = simulated[TYPE_COLUMN]
    body = simulated.drop(columns=[TYPE_COLUMN])
    numeric_body = body.apply(pd.to_numeric, errors="coerce")
    sentinels = _detect_sentinel(numeric_body, types)

    var_cols = [c for c in imputed.columns if c != ID_COLUMN]
    na_cols = [c for c in var_cols if imputed[c].isna().any()]

    stats = {"n_na_before": int(imputed[var_cols].isna().sum().sum()),
             "cont_filled": 0, "cat_filled": 0, "unresolved": []}

    for col in na_cols:
        n_missing = int(imputed[col].isna().sum())
        if col not in numeric_body.index:
            stats["unresolved"].append((col, "absent-in-simulated"))
            continue
        observed = _observed_values(numeric_body.loc[col], sentinels)
        if observed.size == 0:
            stats["unresolved"].append((col, "no-observed-values"))
            continue
        var_type = types.get(col)
        if var_type == "cont":
            fill_value = float(np.mean(observed))
            stats["cont_filled"] += n_missing
        elif var_type == "cat":
            fill_value = _majority_class(observed)
            stats["cat_filled"] += n_missing
        else:
            stats["unresolved"].append((col, f"unknown-type:{var_type!r}"))
            continue
        # Substitute a string so untouched cells keep their exact original text.
        imputed[col] = imputed[col].fillna(repr(float(fill_value)))

    stats["n_na_after"] = int(imputed[var_cols].isna().sum().sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imputed.to_csv(out_path, index=False)

    if verbose:
        detail = (f"    {imputed_path.name}: "
                  f"{stats['n_na_before']}->{stats['n_na_after']} NaN "
                  f"(cont+{stats['cont_filled']}, cat+{stats['cat_filled']})")
        if stats["unresolved"]:
            detail += f"  UNRESOLVED: {stats['unresolved']}"
        print(detail)
    return stats


def process_dataset(label: str, output_subdir: str, verbose: bool) -> None:
    imputed_dir = AUTOCOMPLETE_ROOT / label / "imputed_data"
    simulated_dir = AUTOCOMPLETE_ROOT / label / "simulated_data"
    out_dir = AUTOCOMPLETE_ROOT / label / output_subdir

    if not imputed_dir.is_dir():
        print(f"  [skip] {label}: no imputed_data dir ({imputed_dir})")
        return

    imputed_files = sorted(glob.glob(str(imputed_dir / "masked_values_*.csv")))
    if not imputed_files:
        print(f"  [skip] {label}: no masked_values_*.csv in {imputed_dir}")
        return

    print(f"  [build] {label} -> {out_dir}")
    totals = {"files": 0, "na_before": 0, "na_after": 0,
              "cont_filled": 0, "cat_filled": 0, "unresolved": 0}
    for imputed_path in map(Path, imputed_files):
        simulated_path = simulated_dir / (imputed_path.stem + ".tsv")
        if not simulated_path.exists():
            print(f"    [warn] no simulated file for {imputed_path.name}, skipped")
            continue
        out_path = out_dir / imputed_path.name
        try:
            stats = process_file(imputed_path, simulated_path, out_path, verbose)
        except ValueError as exc:
            print(f"    [warn] {exc}")
            continue
        totals["files"] += 1
        totals["na_before"] += stats["n_na_before"]
        totals["na_after"] += stats["n_na_after"]
        totals["cont_filled"] += stats["cont_filled"]
        totals["cat_filled"] += stats["cat_filled"]
        totals["unresolved"] += len(stats["unresolved"])

    print(f"    [done] {label}: {totals['files']} files | "
          f"NaN {totals['na_before']}->{totals['na_after']} | "
          f"filled cont={totals['cont_filled']} cat={totals['cat_filled']}"
          + (f" | unresolved cols={totals['unresolved']}"
             if totals["unresolved"] else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS),
                        default=list(DEFAULT_DATASETS),
                        help=f"datasets to process (default: {list(DEFAULT_DATASETS)})")
    parser.add_argument("--output-subdir", default="imputed_data_postimputed",
                        help="output subdirectory under each dataset "
                             "(default: imputed_data_postimputed; pass "
                             "'imputed_data' to overwrite in place)")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-file fill statistics")
    args = parser.parse_args()

    print(f"AutoComplete root: {AUTOCOMPLETE_ROOT}")
    print(f"Datasets: {args.datasets} | output subdir: {args.output_subdir}")
    if args.output_subdir == "imputed_data":
        print("  [note] writing in place: original imputed_data will be overwritten")

    for key in args.datasets:
        process_dataset(DATASETS[key], args.output_subdir, args.verbose)


if __name__ == "__main__":
    main()
