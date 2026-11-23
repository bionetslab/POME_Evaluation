"""Impute the full datasets (their original missingness) with missForest, as autoencoder input.

First stage of the "impute, then embed" competitor of the unsupervised clustering analysis
(``scripts/generate_unsupervised_figure.py``): an autoencoder (``scripts/embed_autoencoder.py``)
is trained on the completed tables written here. The imputer and its settings are those of the imputation
benchmark (``src/pome_evaluation/impute_missforest.ipynb``): the ``missforest`` package with
its defaults (LightGBM classifier/regressor with linear trees, median initial guess,
max_iter 5, early stopping), categorical variables passed as ``categorical``, column names
replaced by their positions while fitting. LightGBM is run without bagging/feature
subsampling, so the imputation is deterministic and done once per dataset; the 10 autoencoder runs
share it.

    input : data/input_datasets/<graph file>  (via pome_evaluation.tabular_inputs.load_table)
    output: data/embeddings_competitors/{DATASET}/imputed/{DATASET}_missforest.tsv
            (samples x variables, original labels, no missing values)

CPU:  container/run.sh scripts/impute_full_missforest.py [--datasets HANCOCK ...] [--overwrite]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pome_evaluation.tabular_inputs import DATASETS, imputed_path, load_table  # noqa: E402


def impute(dataset):
    from missforest import MissForest

    table, cat_cols = load_table(dataset)
    positions = {col: str(i) for i, col in enumerate(table.columns)}
    df = table.rename(columns=positions)
    imputer = MissForest(categorical=[positions[c] for c in cat_cols], verbose=0)
    out = imputer.fit_transform(df)
    out = out.rename(columns={v: k for k, v in positions.items()})[table.columns]
    out.index = table.index
    if out.isna().any().any():
        raise RuntimeError("%s: %d values still missing" % (dataset, int(out.isna().sum().sum())))
    observed = table.notna().to_numpy()
    if not np.allclose(out.to_numpy(float)[observed], table.to_numpy(float)[observed]):
        raise RuntimeError("%s: observed values were changed" % dataset)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    for dataset in args.datasets:
        out_path = imputed_path(dataset)
        if os.path.exists(out_path) and not args.overwrite:
            print("[skip] %s exists" % out_path)
            continue
        t0 = time.time()
        out = impute(dataset)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, sep="\t")
        print("[done %4.0fs] %s -> %s" % (time.time() - t0, dataset, out_path), flush=True)


if __name__ == "__main__":
    main()
