"""Supervised TabICL reference for the inductive probing analysis.

TabICL (Qu et al.; ``pip install tabicl``, version 2.2.0; ungated weights on the Hugging Face Hub,
repo ``jingang/TabICL``) is a pre-trained tabular foundation model used here as an *upper
reference*, not as an embedding competitor: per split and binary target it is fit with the
classical ``fit(X_train, y_train)`` on the raw training-split table and predicts the held-out
test split (``predict_proba``). No embedding, PCA or probe is involved, so it has no embedding
size and sees the label it predicts, unlike POME and its unsupervised competitors. (TabPFN >= 2.5
needs a Prior Labs license login before its weights download, hence TabICL.)

* Inputs: the POME-format split files of the inductive embeddings
  (``data/train_test_splits/<dataset>/split_NN_{train,test}_pome.tsv``), so the same samples
  and values. Categorical variables are passed as pandas categories (TabICL ordinal-encodes
  them; missing/unseen -> -1), continuous ones as floats (TabICL mean-imputes them with the
  training means).
* Targets and their 0/1 encoding: ``linear_probe_inductive_embeddings.load_binary_targets``.
  Samples without a label are dropped from fit (train) and scoring (test), as in the probes.
* Model: ``TabICLClassifier`` with its default settings (8-member ensemble, default
  checkpoint ``tabicl-classifier-v2-20260212.ckpt``, ``random_state=42``).

Output (one row per dataset/target/split; metrics as in ``probe_one``):

    output/linear_probing/inductive_tabicl_reference.csv

``plot_inductive_linear_probing.py`` draws the median AP over splits as a dashed reference line.

The checkpoint is read from ``--checkpoint`` (default: ``$(ws_find pome_eval)/models/tabicl/``);
tabicl, einops and huggingface_hub are installed in ``$(ws_find pome_eval)/pylib``, which
run.sh / submit.sh put on PYTHONPATH.

    sbatch container/submit.sh scripts/tabicl_inductive_reference.py
    container/run.sh scripts/tabicl_inductive_reference.py --device cpu --datasets hancock
"""

import argparse
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from linear_probe_inductive_embeddings import DATASETS, OUTPUT_DIR, load_binary_targets  # noqa: E402
from pome_evaluation.tabular_inputs import load_split_tables, split_ids  # noqa: E402

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
DEFAULT_OUTPUT = OUTPUT_DIR / "inductive_tabicl_reference.csv"


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
    df.index = df.index.astype(str)
    for col in cat_cols:
        df[col] = df[col].map(lambda v: np.nan if pd.isna(v) else "c%g" % v).astype("category")
    return df


def score(y_te, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "average_precision": float(average_precision_score(y_te, proba)),
        "roc_auc": float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) == 2 else np.nan,
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--splits", nargs="+", type=int, default=None, help="default: all")
    p.add_argument("--checkpoint", default=None,
                   help="default: <pome_eval ws>/models/tabicl/" + CHECKPOINT)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    opts = p.parse_args()
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    import torch
    from tabicl import TabICLClassifier
    device = opts.device if opts.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = opts.checkpoint or default_checkpoint()
    if not os.path.exists(checkpoint):
        raise SystemExit("TabICL checkpoint not found: %s" % checkpoint)
    print("checkpoint %s, device %s" % (checkpoint, device), flush=True)

    rows = []
    for dataset in opts.datasets:
        targets = load_binary_targets(dataset)
        for split in (opts.splits or split_ids(dataset)):
            train, test, cat_cols = load_split_tables(dataset, split)
            train, test = as_tabicl_frame(train, cat_cols), as_tabicl_frame(test, cat_cols)
            for label, y_full in targets.items():
                t0 = time.time()
                y_tr = y_full.reindex(train.index).dropna().astype(int)
                y_te = y_full.reindex(test.index).dropna().astype(int)
                if len(y_te) == 0 or y_tr.nunique() < 2:
                    continue
                clf = TabICLClassifier(model_path=checkpoint, checkpoint_version=CHECKPOINT,
                                       allow_auto_download=False, device=device)
                clf.fit(train.loc[y_tr.index], y_tr.to_numpy())
                classes = list(clf.classes_)
                proba = clf.predict_proba(test.loc[y_te.index])[:, classes.index(1)]
                rows.append({"dataset": dataset, "method": "TABICL", "target": label,
                             "split": split, "n_train": len(y_tr), "n_test": len(y_te),
                             "train_pos_ratio": float(y_tr.mean()),
                             "test_pos_ratio": float(y_te.mean()),
                             **score(y_te.to_numpy(), proba)})
                print("[%4.0fs] %s split %02d %-24s AP %.3f (pos ratio %.3f)"
                      % (time.time() - t0, dataset, split, label,
                         rows[-1]["average_precision"], rows[-1]["test_pos_ratio"]), flush=True)

    out = pd.DataFrame(rows)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(opts.output, index=False)
    print("Wrote %d rows to %s" % (len(out), opts.output))
    print(out.groupby(["dataset", "target"])["average_precision"].median().round(4))


if __name__ == "__main__":
    main()
