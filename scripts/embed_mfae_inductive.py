"""Inductive missForest + autoencoder embeddings (train-fit, test-transform) for linear probing.

Competitor of POME in the inductive linear-probing analysis
(``scripts/linear_probe_inductive_embeddings.py``). It is the train/test version of the
"impute, then embed" competitor of the clustering analysis (``scripts/impute_full_missforest.py``
and ``scripts/embed_autoencoder.py``, method label MFAE / "missForest+AE"), with the same
settings. It runs on the train/test splits that POME's inductive embeddings use
(``data/train_test_splits/<dataset>/split_NN_{train,test}_pome.tsv``). Nothing is fit on
test samples:

1. missForest (the ``missforest`` package with the benchmark defaults) is fit on the training
   split. The fitted imputer then fills in the training and test samples in one
   ``transform`` call on train + test. Each feature's imputation of a sample depends only on
   that sample's row, so the test rows come out the same as from ``transform(test)``; the
   concatenation only avoids the package's refusal of a test split in which a variable is
   entirely missing. (The stopping check between iterations sees all rows.) The imputation
   is deterministic and done once per split; it is cached under
   ``output/inductive_competitors/_mfae_imputed/<dataset>/split_NN_{train,test}.tsv`` and
   shared by all dims and runs.
2. Design matrix: continuous variables are z-scored with the training mean and std. Categorical
   variables are one-hot encoded over the categories seen in the training split, so a test
   category never seen in training gets an all-zero block. (LUAD's ID-like categoricals,
   "Other Patient ID" and "Form completion date", produce most of these.)
3. The autoencoder of ``embed_autoencoder.py`` (p-256-128-d, ReLU, MSE, Adam 1e-3, batch 64,
   100 epochs, seeds = run) is trained on the training rows only. The bottleneck of the
   frozen encoder embeds the training and test samples.

    output: output/inductive_competitors/mfae/<dataset>/split_NN/dim_D/run_RR_{train,test}.csv

CPU:  container/run.sh scripts/embed_mfae_inductive.py [--datasets hancock] [--runs 0 1 2 3 4]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import embed_autoencoder as ae  # noqa: E402
from pome_evaluation.tabular_inputs import (INDUCTIVE_COMPETITOR_ROOT, SPLIT_DATASETS,  # noqa: E402
                                            inductive_paths, load_split_tables, split_ids,
                                            write_embedding_csv)

METHOD = "mfae"
IMPUTED_ROOT = os.path.join(INDUCTIVE_COMPETITOR_ROOT, "_mfae_imputed")


def impute_split(dataset, split, overwrite=False):
    """Return (imputed train, imputed test, categorical columns) of one split, cached."""
    from missforest import MissForest

    train, test, cat_cols = load_split_tables(dataset, split)
    paths = [os.path.join(IMPUTED_ROOT, dataset, "split_%02d_%s.tsv" % (split, part))
             for part in ("train", "test")]
    if not overwrite and all(os.path.exists(p) for p in paths):
        out = [pd.read_csv(p, sep="\t", index_col=0) for p in paths]
        for df in out:
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)
        return out[0].loc[train.index, train.columns], out[1].loc[test.index, train.columns], cat_cols

    t0 = time.time()
    positions = {col: str(i) for i, col in enumerate(train.columns)}
    imputer = MissForest(categorical=[positions[c] for c in cat_cols], verbose=0)
    imputer.fit(train.rename(columns=positions))
    both = pd.concat([train, test]).rename(columns=positions)
    out = imputer.transform(both).rename(columns={v: k for k, v in positions.items()})
    out = out[train.columns]
    out.index = both.index
    raw = pd.concat([train, test])
    if out.isna().any().any():
        raise RuntimeError("%s split %d: values still missing" % (dataset, split))
    observed = raw.notna().to_numpy()
    if not np.allclose(out.to_numpy(float)[observed], raw.to_numpy(float)[observed]):
        raise RuntimeError("%s split %d: observed values were changed" % (dataset, split))
    imp_train, imp_test = out.iloc[:len(train)], out.iloc[len(train):]
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    for df, p in zip((imp_train, imp_test), paths):
        df.to_csv(p + ".tmp", sep="\t")
        os.replace(p + ".tmp", p)
    print("  [missForest %4.0fs] %s split %02d" % (time.time() - t0, dataset, split), flush=True)
    return imp_train, imp_test, cat_cols


def design_matrices(imp_train, imp_test, cat_cols):
    """z-score (train statistics) + one-hot (training categories) -> float32 train/test."""
    cat_set = set(cat_cols)
    cont_cols = [c for c in imp_train.columns if c not in cat_set]
    tr = imp_train[cont_cols].to_numpy(float)
    mean, std = tr.mean(0), tr.std(0)
    std = np.where(std < 1e-12, 1.0, std)
    blocks_tr, blocks_te = [(tr - mean) / std], [(imp_test[cont_cols].to_numpy(float) - mean) / std]
    for col in cat_cols:
        cats = sorted(imp_train[col].astype(str).unique())
        blocks_tr.append(pd.get_dummies(pd.Categorical(imp_train[col].astype(str), cats)).to_numpy(float))
        blocks_te.append(pd.get_dummies(pd.Categorical(imp_test[col].astype(str), cats)).to_numpy(float))
    return (np.hstack(blocks_tr).astype(np.float32), np.hstack(blocks_te).astype(np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=SPLIT_DATASETS, default=list(SPLIT_DATASETS))
    p.add_argument("--splits", nargs="+", type=int, default=None, help="default: all")
    p.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    opts = p.parse_args()

    import torch
    device = torch.device(opts.device if opts.device == "cpu" or torch.cuda.is_available()
                          else "cpu")
    for dataset in opts.datasets:
        for split in (opts.splits or split_ids(dataset)):
            todo = [(dim, run) for dim in opts.dims for run in opts.runs
                    if opts.overwrite or not all(os.path.exists(q) for q in
                                                 inductive_paths(METHOD, dataset, split, dim, run))]
            if not todo:
                continue
            imp_train, imp_test, cat_cols = impute_split(dataset, split, opts.overwrite)
            X_tr, X_te = design_matrices(imp_train, imp_test, cat_cols)
            for dim, run in todo:
                t0 = time.time()
                encoder, final_loss = ae.train(X_tr, dim, run, opts, device)
                train_out, test_out = inductive_paths(METHOD, dataset, split, dim, run)
                write_embedding_csv(ae.encode(encoder, X_tr, device), imp_train.index, train_out)
                write_embedding_csv(ae.encode(encoder, X_te, device), imp_test.index, test_out)
                print("[done %4.0fs, last-epoch MSE %.4f] mfae %s split %02d dim %d run %d  "
                      "design %s / %s" % (time.time() - t0, final_loss, dataset, split, dim, run,
                                          X_tr.shape, X_te.shape), flush=True)


if __name__ == "__main__":
    main()
