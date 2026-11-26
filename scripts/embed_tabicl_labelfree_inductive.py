"""Inductive, label-free TabICL embeddings for the probing analysis.

Pre-trained tabular-foundation-model competitor of POME in the inductive probing
(``scripts/linear_probe_inductive_embeddings.py``, method ``tabicl_lf``, figure name "TabICL").
TabICL (``pip install tabicl``, version 2.2.0; ungated weights on the Hugging Face Hub, repo
``jingang/TabICL``) is used instead of TabPFN because TabPFN >= 2.5 needs a Prior Labs license
login before its weights download, and TabPFN offers no label-free embedding (the labels enter
from its first layer on).

TabICL runs in three stages: a column-wise embedding (each value embedded within the
distribution of its column, summarized from the context rows), a row-wise interaction
(a transformer over a row's features whose 4 CLS tokens x 128 give a 512-d row
representation), and the in-context learning over the labelled rows. Only the last stage
needs labels in the v1.1 checkpoint ``tabicl-classifier-v1.1-20250506.ckpt``: its column
embedding is not target-aware (``col_target_aware=False``), unlike TabICLv2. The embedding is
the output of the row-wise interaction, so no label enters it:

1. Inputs: the POME-format split files that POME's inductive embeddings use
   (``data/train_test_splits/<dataset>/split_NN_{train,test}_pome.tsv``). Categorical variables
   are passed as pandas categories, so TabICL's own preprocessing ordinal-encodes them
   (missing and unseen categories -> code -1) and mean-imputes the continuous ones (training
   means); its power normalization and outlier clipping are fit on the training split.
2. One forward pass per split and run with the training rows as context and the test rows
   as queries. The column statistics (the induced points of the column-wise set
   transformer) are computed from the training rows only (``embed_with_test=False``), so
   the test rows do not influence any embedding, and training and test rows go through the
   same two stages. The sklearn wrapper needs labels for ``fit``; a fixed dummy label vector
   is passed and only reaches the discarded in-context stage. The script checks this:
   the embeddings with a second, different dummy label vector must be identical.
3. One ensemble member per run (power normalization, random feature order seeded by
   ``random_state = run``), the run-to-run variation. (TabICL's default 8-member ensemble
   cannot be averaged: its members see differently ordered features.)
4. PCA (fit on the training embeddings) reduces the 512 dims to 16/32/64.

One embedding per split, dim and run serves all targets (standard layout):

    output: output/inductive_competitors/tabicl_lf/<dataset>/split_NN/dim_D/run_RR_{train,test}.csv

The checkpoint is read from ``--checkpoint`` (default: ``$(ws_find pome_eval)/models/tabicl/``).
tabicl 2.2.0, einops and huggingface_hub are installed in ``$(ws_find pome_eval)/pylib``
(``pip install --no-deps --target ...``), which run.sh / submit.sh put on PYTHONPATH.

CPU/GPU:  container/run.sh scripts/embed_tabicl_labelfree_inductive.py [--datasets hancock] [--device cpu]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pome_evaluation.tabular_inputs import (inductive_paths, load_split_tables,  # noqa: E402
                                            split_ids, write_embedding_csv)

METHOD = "tabicl_lf"
CHECKPOINT = "tabicl-classifier-v1.1-20250506.ckpt"
DATASETS = ("hancock", "mimic", "luad_noleak")


def default_checkpoint():
    try:
        ws = subprocess.run(["ws_find", "pome_eval"], capture_output=True, text=True,
                            check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        ws = "/hnvme/workspace/y105dc10-pome_eval"
    return os.path.join(ws, "models", "tabicl", CHECKPOINT)


def as_tabicl_frame(table, cat_cols):
    """Categorical variables as pandas categories (NaN kept), continuous ones as float."""
    df = table.copy()
    for col in cat_cols:
        df[col] = df[col].map(lambda v: np.nan if pd.isna(v) else "c%g" % v).astype("category")
    return df


class RowEmbedder:
    """TabICLClassifier (v1.1) whose row-interaction output is captured for all rows."""

    def __init__(self, checkpoint, run, device):
        from tabicl import TabICLClassifier
        self.clf = TabICLClassifier(
            n_estimators=1, norm_methods="power", feat_shuffle_method="random",
            class_shuffle_method="none", model_path=checkpoint,
            checkpoint_version=os.path.basename(checkpoint), allow_auto_download=False,
            kv_cache=False, device=device, random_state=run, verbose=False)
        self._captured = []

    def _hook(self, module, args, output):
        # copy: the in-context stage adds the label embedding to the training rows of this
        # very tensor in place (on CPU, .float().cpu() would return the same storage)
        self._captured.append(output.detach().float().cpu().clone())

    def embed(self, train, test, dummy_y):
        """(train, test) 512-d row representations; dummy_y only reaches the ICL stage."""
        self.clf.fit(train, dummy_y)
        if getattr(self.clf.model_.col_embedder, "target_aware", False):
            raise RuntimeError("checkpoint has a target-aware column embedding: not label-free")
        handle = self.clf.model_.row_interactor.register_forward_hook(self._hook)
        self._captured = []
        try:
            self.clf.predict_proba(test)
        finally:
            handle.remove()
        H = np.concatenate([h.numpy() for h in self._captured], axis=0)   # (members, T, D)
        if H.shape[0] != 1 or H.shape[1] != len(train) + len(test):
            raise RuntimeError("unexpected captured shape %s (train %d, test %d)"
                               % (H.shape, len(train), len(test)))
        return H[0, :len(train)], H[0, len(train):]


def dummy_labels(n, seed):
    """A fixed balanced 0/1 label vector (any labels give the same embedding; see the check)."""
    y = np.arange(n) % 2
    return np.random.default_rng(seed).permutation(y)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=DATASETS,
                   default=["hancock", "mimic", "luad_noleak"])
    p.add_argument("--splits", nargs="+", type=int, default=None, help="default: all")
    p.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--checkpoint", default=None,
                   help="default: <pome_eval ws>/models/tabicl/" + CHECKPOINT)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    opts = p.parse_args()

    import torch
    device = opts.device if opts.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = opts.checkpoint or default_checkpoint()
    if not os.path.exists(checkpoint):
        raise SystemExit("TabICL checkpoint not found: %s" % checkpoint)
    print("checkpoint %s, device %s" % (checkpoint, device), flush=True)
    embedders = {run: RowEmbedder(checkpoint, run, device) for run in opts.runs}

    checked = False
    for dataset in opts.datasets:
        for split in (opts.splits or split_ids(dataset)):
            todo = [run for run in opts.runs if opts.overwrite or not all(
                os.path.exists(q) for dim in opts.dims
                for q in inductive_paths(METHOD, dataset, split, dim, run))]
            if not todo:
                continue
            train, test, cat_cols = load_split_tables(dataset, split)
            train, test = as_tabicl_frame(train, cat_cols), as_tabicl_frame(test, cat_cols)
            for run in todo:
                t0 = time.time()
                Z_train, Z_test = embedders[run].embed(train, test, dummy_labels(len(train), 0))
                if not checked:     # the labels must not change the embedding
                    Z_train2, Z_test2 = embedders[run].embed(train, test,
                                                             dummy_labels(len(train), 1))
                    diff = max(np.abs(Z_train - Z_train2).max(), np.abs(Z_test - Z_test2).max())
                    if diff > 1e-4:
                        raise RuntimeError("embedding depends on the dummy labels (max diff %g)"
                                           % diff)
                    print("label-independence check passed (max diff %g)" % diff, flush=True)
                    checked = True
                for dim in opts.dims:
                    pca = PCA(n_components=dim, random_state=0).fit(Z_train)
                    train_out, test_out = inductive_paths(METHOD, dataset, split, dim, run)
                    write_embedding_csv(pca.transform(Z_train), train.index, train_out)
                    write_embedding_csv(pca.transform(Z_test), test.index, test_out)
                print("[done %4.0fs] tabicl_lf %s split %02d run %d  train %s test %s"
                      % (time.time() - t0, dataset, split, run, train.shape, test.shape),
                      flush=True)


if __name__ == "__main__":
    main()
