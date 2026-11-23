#!/usr/bin/env python
"""Impute the simulated-missingness files with TabImpute (Feitelberg et al., 2025).

TabImpute is a pre-trained transformer that imputes zero-shot (no fitting). This runs
the authors' package (github.com/jacobf18/tabular @ 2cf54ac, checkpoint
``tabimpute_001.ckpt``; both pinned in ``container/tabimpute.def``) on every masked file
of the benchmark and writes the imputed tables next to the other baselines:

    input : data/imputation_data/missforest/{DATASET}/simulated_data/masked_values_{r}_{run}.tsv
    output: data/imputation_data/tabimpute/{DATASET}/imputed_data/masked_values_{r}_{run}.tsv

Inputs are the MissForest copies of the masked files (the only samples x variables
copies with NaN for every missing entry); outputs keep their row/column labels, so
``scripts/generate_imputation_results.py`` scores them like the KNN/MissForest files.

Configuration (all upstream code, no changes to the model or package):
- ``TabImputeCategorical``, the package's mixed-type entry point: categorical variables
  (the dataset's categorical list) are one-hot encoded, the matrix is imputed, and each
  categorical takes the argmax over its one-hot block. Continuous values are the median
  of the predicted distribution (the package default).
- The ensemble of four random row/column permutations the authors use in their own
  benchmarks (``benchmark/missbench/methods.py``,
  ``benchmark/categorical/get_openml_categorical_errors.py``); predictions are averaged.
- Windowing. TabImpute treats every cell as a token with full attention, so memory grows
  ~quadratically in rows x columns (CPU: 50x50 5 GB, 100x50 25 GB, 100x100 124 GB),
  and the paper notes its attention cost and leaves chunking to future work. Each
  permuted matrix is therefore tiled into windows of ``--max-rows`` x ``--max-cols``
  (default 100 x 50, the table size the paper states for pre-training) with the
  package's own ``max_num_rows``/``max_num_cols`` tiling; because the tiling runs after
  the random permutation, every ensemble member sees different row/column groupings.

``--seed`` seeds numpy (the permutations) and torch for every file, so runs repeat.
Needs a GPU for realistic run times (``--device cuda``, the default when available).

Run from the repo root inside the TabImpute image, on an H200 node:

    sbatch --export=POME_EVAL_SIF=$(ws_find pome_eval)/containers/tabimpute.sif \\
        container/submit.sh scripts/impute_tabimpute.py --datasets hancock luad mimic

(submit.sh runs with ``--export=NONE``, so a plain ``POME_EVAL_SIF=... sbatch`` does not
reach the job and it silently runs in pome_eval.sif instead.)
    ... --ratios 0.1 --runs 0 --dry-run     # list what would be done
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPUTATION_ROOT = PROJECT_ROOT / "data" / "imputation_data"
INPUT_METHOD_DIR = "missforest"          # where the samples x variables inputs live
OUTPUT_METHOD_DIR = "tabimpute"
# Baked into container/tabimpute.def (HF revision 53c062ee, sha256-checked at build).
DEFAULT_CHECKPOINT = "/opt/tabimpute/tabimpute_001.ckpt"

# dataset CLI key -> (on-disk label, categorical-variable list inside the input dir)
DATASETS = {
    "hancock": ("HANCOCK", "cat_variables.csv"),
    "luad": ("TCGA_LUAD", "TCGA_LUAD_cat_vars.csv"),
    "mimic": ("MIMIC", "mimic_cat_variables.csv"),
}


def build_imputer(opts):
    from tabimpute.interface import TabImputeCategorical
    from tabimpute.prepreocess import (
        RandomColumnPermutation,
        RandomRowColumnPermutation,
        RandomRowPermutation,
    )

    preprocessors = [
        RandomRowColumnPermutation(),
        RandomRowColumnPermutation(),
        RandomRowPermutation(),
        RandomColumnPermutation(),
    ]
    imputer = TabImputeCategorical(
        device=opts.device, nhead=2, preprocessors=preprocessors,
        checkpoint_path=opts.checkpoint)
    # TabImputeCategorical does not expose the window size; set it on the wrapped
    # ImputePFN. max_num_chunks=1 makes the row window exactly max_num_rows (with the
    # default inf the row window silently becomes the whole table).
    imputer.imputer.max_num_rows = opts.max_rows
    imputer.imputer.max_num_cols = opts.max_cols
    imputer.imputer.max_num_chunks = 1
    return imputer


def impute_file(imputer, in_path: Path, out_path: Path, cat_vars, opts) -> float:
    import torch

    t0 = time.time()
    df = pd.read_csv(in_path, sep="\t", index_col=0)
    values = df.to_numpy(dtype=float)
    cat_idx = [j for j, c in enumerate(df.columns) if c in cat_vars]

    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    out = imputer.impute(values.copy(), categorical_columns=cat_idx)

    observed = ~np.isnan(values)
    out[observed] = values[observed]                   # keep observed values exact
    if np.isnan(out).any():
        raise RuntimeError(f"{int(np.isnan(out).sum())} imputed values are NaN")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tsv.tmp")
    pd.DataFrame(out, index=df.index, columns=df.columns).to_csv(tmp, sep="\t")
    os.replace(tmp, out_path)
    if opts.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return time.time() - t0


def _ratio_run(path: Path):
    # masked_values_{ratio}_{run}.tsv
    _, _, ratio, run = path.stem.split("_")
    return float(ratio), int(float(run))


def collect_tasks(opts):
    tasks = []
    for key in opts.datasets:
        label, cat_file = DATASETS[key]
        in_dir = IMPUTATION_ROOT / INPUT_METHOD_DIR / label
        files = sorted((in_dir / "simulated_data").glob("masked_values_*.tsv"))
        if not files:
            print(f"[skip] {label}: no inputs under {in_dir}")
            continue
        cat_vars = set(pd.read_csv(in_dir / cat_file)["cat_var"])
        for path in files:
            ratio, run = _ratio_run(path)
            if opts.ratios and ratio not in opts.ratios:
                continue
            if opts.runs and run not in opts.runs:
                continue
            out = IMPUTATION_ROOT / OUTPUT_METHOD_DIR / label / "imputed_data" / path.name
            if out.exists() and not opts.overwrite:
                continue
            tasks.append((path, out, cat_vars))
    return tasks


def main() -> None:
    import torch

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--ratios", nargs="+", type=float, help="only these missingness ratios")
    p.add_argument("--runs", nargs="+", type=int, help="only these simulation runs")
    p.add_argument("--max-rows", type=int, default=100, help="rows per window")
    p.add_argument("--max-cols", type=int, default=50, help="columns per window")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", default=os.environ.get("TABIMPUTE_CHECKPOINT", DEFAULT_CHECKPOINT),
                   help=f"model checkpoint (default: $TABIMPUTE_CHECKPOINT or {DEFAULT_CHECKPOINT})")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    opts = p.parse_args()

    tasks = collect_tasks(opts)
    print(f"{len(tasks)} file(s) to impute with TabImpute on {opts.device} "
          f"(windows {opts.max_rows} x {opts.max_cols})")
    if opts.dry_run:
        for in_path, out, _ in tasks:
            print(f"  {in_path.relative_to(PROJECT_ROOT)} -> {out.relative_to(PROJECT_ROOT)}")
        return
    if not tasks:
        return
    if not os.path.isfile(opts.checkpoint):
        raise SystemExit(f"checkpoint not found: {opts.checkpoint} (pass --checkpoint)")

    imputer = build_imputer(opts)
    for in_path, out, cat_vars in tasks:
        secs = impute_file(imputer, in_path, out, cat_vars, opts)
        peak = (f", peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
                if opts.device.startswith("cuda") else "")
        print(f"[done {secs:6.1f}s{peak}] {out.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
