"""Inductive linear-probing of the POME / UMAP embeddings produced by
``scripts/generate_inductive_embeddings.py``.

For every computed embedding (dataset x method x split x dim x run) a logistic
regression is *fit on the training-split embedding* and *evaluated on the
held-out test-split embedding*, for each binary held-out target variable of the
dataset. This mirrors the targets and classifier configuration of the
transductive notebooks ``src/pome_evaluation/analyze_<DATASET>_embedding_separability.ipynb``
but uses the inductive train/test structure instead of within-embedding k-fold CV.

Binary targets per dataset (numeric/regression targets from the notebooks are
intentionally excluded -- this script does logistic regression only):

    hancock : recurrence, survival_status, rfs_event
    luad    : Disease Free Status, Disease-specific Survival status,
              Progression Free Status
    mimic   : label_aplasia, label_nf

Object-typed targets are integer-encoded with LabelEncoder fit on the full
target column (reproducing the notebooks; e.g. recurrence no=0/yes=1,
survival_status deceased=0/living=1). Numeric 0/1 targets pass through unchanged.

The classifier matches the notebooks: LogisticRegression(penalty="l2",
solver="liblinear", max_iter=1000), and embeddings are NOT rescaled (use
--standardize to z-score features before fitting).

Run from the project root (any env with pandas + scikit-learn):

    conda run -n torch python scripts/linear_probe_inductive_embeddings.py
    conda run -n torch python scripts/linear_probe_inductive_embeddings.py --datasets hancock --dims 32

Output: one tidy long-format CSV (one row per dataset/method/target/dim/split/run)

    output/linear_probing/inductive_linear_probing_results.csv

with the headline metric ``average_precision`` plus roc_auc / precision /
recall / f1, the per-split sample counts and the naive test-set positive rate
(``test_pos_ratio``) as the AP baseline.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATASETS = PROJECT_ROOT / "data" / "input_datasets"
DEFAULT_EMB_ROOT = PROJECT_ROOT / "output" / "inductive"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "linear_probing" / \
    "inductive_linear_probing_results.csv"

DATASETS = ("hancock", "luad", "mimic")
METHODS = ("pome", "umap")

# Per-dataset target file and the binary targets to probe (raw column ->
# display label, mirroring the notebooks).
DATASET_TARGETS = {
    "hancock": {
        "file": "hancock_targets.csv",
        "targets": {
            "recurrence": "Recurrence",
            "survival_status": "Survival",
            "rfs_event": "RFS Event",
        },
    },
    "luad": {
        "file": "TCGA_LUAD_targets.csv",
        "targets": {
            "Disease Free Status": "Disease Free Status",
            "Disease-specific Survival status": "DSS Status",
            "Progression Free Status": "Progression Free Status",
        },
    },
    "mimic": {
        "file": "mimic_targets.csv",
        "targets": {
            "label_aplasia": "Aplasia",
            "label_nf": "Neutropenic Fever",
        },
    },
}


# --- Target loading ----------------------------------------------------------
def load_binary_targets(dataset: str) -> dict[str, pd.Series]:
    """Return {display_label: int-encoded 0/1 Series indexed by sample id}.

    LabelEncoder is fit on the full non-null column so the 0/1 meaning is
    consistent across all splits (and reproduces the notebooks' encoding).
    """
    spec = DATASET_TARGETS[dataset]
    df = pd.read_csv(INPUT_DATASETS / spec["file"], index_col=0)
    df.index = df.index.astype(str)

    out = {}
    for raw_col, label in spec["targets"].items():
        col = df[raw_col].dropna()
        encoded = pd.Series(LabelEncoder().fit_transform(col), index=col.index)
        n_classes = encoded.nunique()
        if n_classes != 2:
            raise ValueError(
                f"{dataset}:{raw_col} is not binary ({n_classes} classes)")
        out[label] = encoded.astype(int)
    return out


# --- Embedding discovery -----------------------------------------------------
def discover_runs(emb_root: Path, method: str, dataset: str,
                  dims, splits, runs):
    """Yield (split, dim, run, train_csv, test_csv) for existing run pairs."""
    base = emb_root / method / dataset
    if not base.is_dir():
        return
    for split_dir in sorted(base.glob("split_*")):
        split = int(split_dir.name.split("_")[1])
        if splits is not None and split not in splits:
            continue
        for dim_dir in sorted(split_dir.glob("dim_*")):
            dim = int(dim_dir.name.split("_")[1])
            if dims is not None and dim not in dims:
                continue
            for train_csv in sorted(dim_dir.glob("run_*_train.csv")):
                run = int(train_csv.name.split("_")[1])
                if runs is not None and run not in runs:
                    continue
                test_csv = train_csv.with_name(
                    train_csv.name.replace("_train.csv", "_test.csv"))
                if test_csv.exists():
                    yield split, dim, run, train_csv, test_csv


# --- Probing -----------------------------------------------------------------
def probe_one(train_emb: pd.DataFrame, test_emb: pd.DataFrame,
              y_full: pd.Series, standardize: bool) -> dict | None:
    """Fit logreg on train embedding, evaluate on test embedding for one target.

    Returns a metrics dict, or None if the target can't be probed for this
    split (e.g. <2 classes in the training labels).
    """
    y_tr = y_full.reindex(train_emb.index)
    y_te = y_full.reindex(test_emb.index)
    tr_mask, te_mask = y_tr.notna(), y_te.notna()

    X_tr = train_emb.loc[tr_mask].to_numpy()
    y_tr = y_tr.loc[tr_mask].astype(int).to_numpy()
    X_te = test_emb.loc[te_mask].to_numpy()
    y_te = y_te.loc[te_mask].astype(int).to_numpy()

    # A logistic regression cannot consume non-finite embedding coordinates,
    # which the inductive transform can occasionally produce (notably UMAP for a
    # held-out sample with no usable neighbours). Drop those rows from both fit
    # and evaluation and record how many were dropped so it stays visible.
    tr_finite = np.isfinite(X_tr).all(axis=1)
    te_finite = np.isfinite(X_te).all(axis=1)
    n_train_nonfinite = int((~tr_finite).sum())
    n_test_nonfinite = int((~te_finite).sum())
    X_tr, y_tr = X_tr[tr_finite], y_tr[tr_finite]
    X_te, y_te = X_te[te_finite], y_te[te_finite]

    if len(y_tr) == 0 or len(y_te) == 0 or len(np.unique(y_tr)) < 2:
        return None

    if standardize:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    clf = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    test_has_both = len(np.unique(y_te)) == 2
    return {
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "n_train_nonfinite": n_train_nonfinite,
        "n_test_nonfinite": n_test_nonfinite,
        "train_pos_ratio": float(y_tr.mean()),
        "test_pos_ratio": float(y_te.mean()),
        "average_precision": float(average_precision_score(y_te, proba)),
        "roc_auc": float(roc_auc_score(y_te, proba)) if test_has_both else np.nan,
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="dims to probe (default: all found)")
    parser.add_argument("--splits", nargs="+", type=int, default=None,
                        help="split ids to probe (default: all found)")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="run ids to probe (default: all found)")
    parser.add_argument("--emb-root", type=Path, default=DEFAULT_EMB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--standardize", action="store_true",
                        help="z-score embedding features (fit on train) "
                             "before logistic regression (default: off, "
                             "matching the notebooks)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    print(f"Embedding root: {args.emb_root}")
    print(f"Datasets: {args.datasets} | methods: {args.methods} | "
          f"standardize: {args.standardize}")

    rows = []
    for dataset in args.datasets:
        targets = load_binary_targets(dataset)
        print(f"\n[{dataset}] targets: {list(targets)}")
        for method in args.methods:
            run_pairs = list(discover_runs(
                args.emb_root, method, dataset,
                args.dims, args.splits, args.runs))
            if not run_pairs:
                print(f"  [warn] no {method} embeddings found")
                continue
            n_done = 0
            n_nonfinite = 0  # total non-finite embedding rows dropped (train+test)
            for split, dim, run, train_csv, test_csv in run_pairs:
                train_emb = pd.read_csv(train_csv, index_col=0)
                test_emb = pd.read_csv(test_csv, index_col=0)
                train_emb.index = train_emb.index.astype(str)
                test_emb.index = test_emb.index.astype(str)
                for label, y_full in targets.items():
                    metrics = probe_one(train_emb, test_emb, y_full,
                                        args.standardize)
                    if metrics is None:
                        continue
                    n_nonfinite += (metrics["n_train_nonfinite"]
                                    + metrics["n_test_nonfinite"])
                    rows.append({
                        "dataset": dataset, "method": method.upper(),
                        "target": label, "dim": dim, "split": split,
                        "run": run, **metrics,
                    })
                n_done += 1
            msg = (f"  {method:4s}: probed {n_done} embedding pairs "
                   f"x {len(targets)} targets")
            if n_nonfinite:
                msg += (f"  [warn] dropped {n_nonfinite} non-finite embedding "
                        f"rows (see n_*_nonfinite columns)")
            print(msg)

    if not rows:
        print("\nNo results produced -- check --emb-root has embeddings.")
        return

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(result_df)} rows to {args.output}")

    # Compact summary: mean AP across splits x runs per dataset/target/dim/method.
    print("\n=== Mean Average Precision (across splits x runs) ===")
    summary = (result_df
               .groupby(["dataset", "target", "dim", "method"])
               ["average_precision"].mean().round(4)
               .unstack("method"))
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summary)


if __name__ == "__main__":
    main()
