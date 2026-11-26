"""Inductive SAINT embeddings (train-fit, test-transform) for the linear-probing analysis.

Competitor of POME in ``scripts/linear_probe_inductive_embeddings.py``. It is the train/test
version of ``scripts/embed_saint.py`` (the clustering competitor), with identical model and
pretraining (the authors' unmodified ``SAINT_pretrain``, colrow attention, depth 1, 4 heads,
token size = embedding size, 50 epochs, batch 256, seeds = run). The input is the train/test
splits that POME's inductive embeddings use
(``data/train_test_splits/<dataset>/split_NN_{train,test}_pome.tsv``).

Per (split, dim, run), SAINT is pretrained on the training split only. All preprocessing
statistics are taken from the training split:

* categorical columns are label-encoded over the training categories, plus "MissingValue" if
  the column has missing training values (as ``embed_saint.prepare``). A missing test value, or
  a test category never seen in training, is flagged in the mask, so its token is replaced by
  the learned mask embedding. (LUAD's ID-like categoricals, "Other Patient ID" and "Form
  completion date", produce most of the unseen categories.)
* continuous columns: missing values are filled with the training column mean and flagged in
  the mask. Values are standardized with the training mean/std (std floored at 1e-6).

Embeddings are the CLS-token output of the frozen model in eval mode, as in ``embed_saint.py``.
Because of SAINT's intersample (row) attention, a sample's encoding depends on its batch. The
training and test samples are therefore encoded separately, each in consecutive batches of 256
in input order: test samples attend only to other test samples, never to training samples.

    output: output/inductive_competitors/saint/<dataset>/split_NN/dim_D/run_RR_{train,test}.csv

GPU (SAINT's mixup calls .cuda()):
    sbatch --job-name=ind_saint --export=POME_EVAL_SIF=$(ws_find pome_eval)/containers/saint.sif,PYTHONHASHSEED=0 \\
        container/submit.sh scripts/embed_saint_inductive.py --datasets hancock luad_noleak
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
import embed_saint as es  # noqa: E402
from pome_evaluation.tabular_inputs import (SPLIT_DATASETS, inductive_paths,  # noqa: E402
                                            load_split_tables, split_ids, write_embedding_csv)

METHOD = "saint"


def prepare_split(train, test, cat_cols):
    """Return (X_train, X_test, cat_idxs, con_idxs, cat_dims, mean_std, n_unseen)."""
    cat_set = set(cat_cols)
    cat_idxs = [j for j, c in enumerate(train.columns) if c in cat_set]
    con_idxs = [j for j, c in enumerate(train.columns) if c not in cat_set]
    d_tr = np.zeros(train.shape, dtype=np.float64)
    d_te = np.zeros(test.shape, dtype=np.float64)
    m_tr = train.notna().to_numpy().astype(int)
    m_te = test.notna().to_numpy().astype(int)
    cat_dims, n_unseen = [], 0

    def label(v):
        return "MissingValue" if pd.isna(v) else repr(float(v))

    for j, col in enumerate(train.columns):
        if j in cat_idxs:
            labels_tr = train[col].map(label)
            classes = sorted(labels_tr.unique())
            lookup = {c: i for i, c in enumerate(classes)}
            d_tr[:, j] = labels_tr.map(lookup).to_numpy()
            codes_te = test[col].map(label).map(lookup)
            unseen = (test[col].notna() & codes_te.isna()).to_numpy()
            n_unseen += int(unseen.sum())
            m_te[unseen, j] = 0                                   # -> mask embedding
            d_te[:, j] = codes_te.fillna(0).to_numpy()            # value ignored where masked
            cat_dims.append(len(classes))
        else:
            mean = train[col].mean()
            d_tr[:, j] = train[col].fillna(mean).to_numpy()
            d_te[:, j] = test[col].fillna(mean).to_numpy()
    cont = d_tr[:, con_idxs].astype(np.float32)
    mean, std = cont.mean(0), cont.std(0)
    std = np.where(std < 1e-6, 1e-6, std)
    mean_std = np.array([mean, std]).astype(np.float32)
    return ({"data": d_tr, "mask": m_tr}, {"data": d_te, "mask": m_te},
            cat_idxs, con_idxs, cat_dims, mean_std, n_unseen)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=SPLIT_DATASETS, default=list(SPLIT_DATASETS))
    p.add_argument("--splits", nargs="+", type=int, default=None, help="default: all")
    p.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--epochs", type=int, default=50, help="pretraining epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    opts = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("SAINT's pretraining (mixup) needs a GPU")
    device = torch.device("cuda:0")
    saint = es.import_saint()
    for dataset in opts.datasets:
        for split in (opts.splits or split_ids(dataset)):
            todo = [(dim, run) for dim in opts.dims for run in opts.runs
                    if opts.overwrite or not all(os.path.exists(q) for q in
                                                 inductive_paths(METHOD, dataset, split, dim, run))]
            if not todo:
                continue
            train, test, cat_cols = load_split_tables(dataset, split)
            X_tr, X_te, cat_idxs, con_idxs, cat_dims, mean_std, n_unseen = prepare_split(
                train, test, cat_cols)
            for dim, run in todo:
                t0 = time.time()
                model = es.pretrain(saint, X_tr, cat_idxs, con_idxs, cat_dims, mean_std,
                                    dim, run, opts, device)
                train_out, test_out = inductive_paths(METHOD, dataset, split, dim, run)
                write_embedding_csv(es.encode(saint, model, X_tr, cat_idxs, mean_std, dim, opts,
                                              device), train.index, train_out)
                write_embedding_csv(es.encode(saint, model, X_te, cat_idxs, mean_std, dim, opts,
                                              device), test.index, test_out)
                print("[done %4.0fs, unseen test categories %d] saint %s split %02d dim %d run %d"
                      % (time.time() - t0, n_unseen, dataset, split, dim, run), flush=True)


if __name__ == "__main__":
    main()
