"""Inductive, target-conditioned TabICL embeddings for the linear-probing analysis.

Pre-trained tabular foundation model competitor of POME in the inductive probing
(``scripts/linear_probe_inductive_embeddings.py``). TabPFN >= 2.5 needs a Prior Labs license
login before its weights download, so TabICL (Qu et al.; ``pip install tabicl``; ungated weights
on the Hugging Face Hub, repo ``jingang/TabICL``) is used instead, with its newest checkpoint
``tabicl-classifier-v2-20260212.ckpt`` (TabICLv2).

TabICLv2 is an in-context learner: it reads the labelled training rows as context and
predicts the test rows, and its column embedding is target-aware (``col_target_aware``).
Its sample representations therefore depend on the label. Here they are made per probing
target, in the style of the TabPFN-extensions embedding (``TabPFNEmbedding``), so this
competitor is *supervised*: unlike POME, UMAP, HI-VAE, SAINT and missForest+AE it sees the
target that is probed. Nothing is fit on test samples or test labels:

1. Inputs: the POME-format split files that POME's inductive embeddings use
   (``data/train_test_splits/<dataset>/split_NN_{train,test}_pome.tsv``). Categorical
   variables are passed as pandas categories, so TabICL's own preprocessing ordinal-encodes
   them (missing and unseen categories -> code -1) and mean-imputes the continuous ones
   (training means). Labels are the int-encoded binary targets of the probing script
   (``load_binary_targets``); training rows without a label are not embedded (the probe
   drops them anyway).
2. Embedding = the final in-context-learning hidden state of a query row (the input of
   TabICL's decoder head, after the last layer norm; 4 CLS tokens x 128 = 512 dims).
   * test rows: context = all labelled training rows.
   * training rows: out-of-fold, by stratified 5-fold cross-fitting (context = the other
     4 folds). A training row inside the context sees its own label, so its in-context
     representation would not be comparable to that of a query row.
3. One ensemble member per run: TabICL's default 8-member ensemble mixes class shuffles,
   which permute the label encoding and so the embedding space, so members cannot be
   averaged. Each run uses one member with the power normalization, no class shuffle and a
   random feature order seeded by the run (``random_state = run``) -- the run-to-run variation.
4. PCA (fit on the out-of-fold training embeddings) reduces the 512 dims to 16/32/64, the
   embedding sizes of the other methods.

    output: output/inductive_competitors/tabicl/<dataset>/target_<target>/split_NN/dim_D/
            run_RR_{train,test}.csv

``linear_probe_inductive_embeddings.py`` probes each ``target_<target>`` directory for its own
target only (``TARGET_CONDITIONED_METHODS``).

The checkpoint is read from ``--checkpoint`` (default: ``$(ws_find pome_eval)/models/tabicl/``),
tabicl 2.2.0 + einops + huggingface_hub are installed in ``$(ws_find pome_eval)/pylib``
(``pip install --no-deps --target ...``), which run.sh / submit.sh put on PYTHONPATH.

GPU:  sbatch container/submit.sh scripts/embed_tabicl_inductive.py [--datasets hancock] [--runs 0 1]
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
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from linear_probe_inductive_embeddings import load_binary_targets, target_slug  # noqa: E402
from pome_evaluation.tabular_inputs import (INDUCTIVE_COMPETITOR_ROOT, load_split_tables,  # noqa: E402
                                            split_ids, write_embedding_csv)

METHOD = "tabicl"
CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
DATASETS = ("hancock", "mimic", "luad_noleak", "luad")
N_FOLDS = 5


def target_paths(dataset, label, split, dim, run, root=INDUCTIVE_COMPETITOR_ROOT):
    d = os.path.join(root, METHOD, dataset, "target_" + target_slug(label),
                     "split_%02d" % split, "dim_%d" % dim)
    return (os.path.join(d, "run_%02d_train.csv" % run),
            os.path.join(d, "run_%02d_test.csv" % run))


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


class TabICLEmbedder:
    """TabICLClassifier whose decoder-head input is captured for the query rows."""

    def __init__(self, checkpoint, run, device):
        from tabicl import TabICLClassifier
        self.clf = TabICLClassifier(
            n_estimators=1, norm_methods="power", feat_shuffle_method="random",
            class_shuffle_method="none", model_path=checkpoint,
            checkpoint_version=os.path.basename(checkpoint), allow_auto_download=False,
            kv_cache=False, device=device, random_state=run, verbose=False)
        self._captured = []

    def _hook(self, module, args):
        self._captured.append(args[0].detach().float().cpu())

    def embed(self, X_ctx, y_ctx, X_query):
        """Fit on the context rows, return the (n_query, 512) hidden states of the query rows."""
        self.clf.fit(X_ctx, y_ctx)
        handle = self.clf.model_.icl_predictor.decoder.register_forward_pre_hook(self._hook)
        self._captured = []
        try:
            self.clf.predict_proba(X_query)
        finally:
            handle.remove()
        H = np.concatenate([h.numpy() for h in self._captured], axis=0)   # (members, T, D)
        if H.shape[0] != 1 or H.shape[1] != len(X_ctx) + len(X_query):
            raise RuntimeError("unexpected captured shape %s (context %d, query %d)"
                               % (H.shape, len(X_ctx), len(X_query)))
        return H[0, len(X_ctx):]


def embed_split(embedder, train, test, y_train, run):
    """(out-of-fold train embedding, test embedding) of one split, target and run."""
    Z_train = np.empty((len(train), 0))
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=run).split(train, y_train)
    for ctx, query in folds:
        Z = embedder.embed(train.iloc[ctx], y_train[ctx], train.iloc[query])
        if Z_train.shape[1] == 0:
            Z_train = np.full((len(train), Z.shape[1]), np.nan)
        Z_train[query] = Z
    Z_test = embedder.embed(train, y_train, test)
    return Z_train, Z_test


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=DATASETS,
                   default=["hancock", "mimic", "luad_noleak"])
    p.add_argument("--targets", nargs="+", default=None, help="display labels (default: all)")
    p.add_argument("--splits", nargs="+", type=int, default=None, help="default: all")
    p.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--checkpoint", default=None, help="default: <pome_eval ws>/models/tabicl/"
                   + CHECKPOINT)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    opts = p.parse_args()

    import torch
    device = opts.device if opts.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = opts.checkpoint or default_checkpoint()
    if not os.path.exists(checkpoint):
        raise SystemExit("TabICL checkpoint not found: %s" % checkpoint)
    print("checkpoint %s, device %s" % (checkpoint, device), flush=True)
    embedders = {run: TabICLEmbedder(checkpoint, run, device) for run in opts.runs}

    for dataset in opts.datasets:
        targets = load_binary_targets(dataset)
        labels = [t for t in targets if opts.targets is None or t in opts.targets]
        for split in (opts.splits or split_ids(dataset)):
            tables = None
            for label in labels:
                for run in opts.runs:
                    if not opts.overwrite and all(
                            os.path.exists(q) for dim in opts.dims
                            for q in target_paths(dataset, label, split, dim, run)):
                        continue
                    if tables is None:
                        train, test, cat_cols = load_split_tables(dataset, split)
                        tables = as_tabicl_frame(train, cat_cols), as_tabicl_frame(test, cat_cols)
                    train, test = tables
                    y = targets[label].reindex(train.index)
                    labelled = y.notna().to_numpy()
                    tr, y_tr = train.loc[labelled], y[labelled].astype(int).to_numpy()
                    t0 = time.time()
                    Z_train, Z_test = embed_split(embedders[run], tr, test, y_tr, run)
                    for dim in opts.dims:
                        pca = PCA(n_components=dim, random_state=0).fit(Z_train)
                        train_out, test_out = target_paths(dataset, label, split, dim, run)
                        write_embedding_csv(pca.transform(Z_train), tr.index, train_out)
                        write_embedding_csv(pca.transform(Z_test), test.index, test_out)
                    print("[done %4.0fs] tabicl %s %-24s split %02d run %d  train %s test %s"
                          % (time.time() - t0, dataset, label, split, run, tr.shape,
                             test.shape), flush=True)


if __name__ == "__main__":
    main()
