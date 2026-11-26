"""Inductive probing of the sample embeddings of POME and its competitors.

Methods (embeddings fit on the training split, the held-out test split transformed):

    pome       POME 1.4                  scripts/generate_inductive_embeddings.py   output/inductive_pome14
    umap       NA-aware UMAP             scripts/generate_inductive_embeddings.py   output/inductive
    hivae      HI-VAE                    scripts/embed_hivae_inductive.py           output/inductive_competitors
    saint      SAINT (native)            scripts/embed_saint_inductive.py           output/inductive_competitors
    mfae       missForest + autoencoder  scripts/embed_mfae_inductive.py            output/inductive_competitors
    tabicl_lf  TabICL v1.1, label-free   scripts/embed_tabicl_labelfree_inductive.py output/inductive_competitors

Every method is read from its own root (``METHOD_ROOTS``), so one call probes all of them;
``--emb-root`` overrides the root for every method (e.g. for variant embeddings). For every
embedding (dataset x method x split x dim x run) the probe is *fit on the training-split
embedding* and *evaluated on the held-out test-split embedding*, for each binary target:

    hancock     : recurrence, survival_status, rfs_event
    mimic       : label_aplasia, label_nf
    luad_noleak : Disease Free Status, Disease-specific Survival status, Progression Free Status

``luad_noleak`` is TCGA-LUAD without its four outcome/follow-up variables
(``scripts/make_luad_noleak_splits.py``); it is the only LUAD dataset of the inductive analyses.
Object-typed targets are integer-encoded with LabelEncoder fit on the full target column (e.g.
recurrence no=0/yes=1, survival_status deceased=0/living=1); numeric 0/1 targets pass through.

Probes (``--probe``):

    logreg_std : logistic regression on standardized features (z-scored with the training
                 mean/std): LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000).
                 The default.
    mlp        : one hidden layer of 64 ReLU units on standardized features, Adam, early stopping
                 on a 10% validation split of the training data, seed 0. The L2 penalty alpha in
                 {1e-4, 1e-2, 1} is tuned by an inner stratified 3-fold CV (seed 0, scoring =
                 average precision) on the training split only, then refit on the whole split.

Variable-type modes (``--modes``)
---------------------------------
``scripts/generate_inductive_embeddings.py --modes`` can embed a variable-type subset of the
same splits (``numeric_only`` / ``cat_only``) next to the default ``combined`` embeddings. All
discovered modes are probed by default and tagged in the ``mode`` column, so the
data-integration comparison (``scripts/plot_inductive_type_combination.py``) uses the same
targets, probe and splits as the combined probing.

    container/run.sh scripts/linear_probe_inductive_embeddings.py --jobs 32
    container/run.sh scripts/linear_probe_inductive_embeddings.py --probe mlp --jobs 32
    container/run.sh scripts/linear_probe_inductive_embeddings.py --datasets hancock --methods pome --dims 32

Output: one tidy long-format CSV (one row per dataset/method/mode/target/dim/split/run)

    output/linear_probing/inductive_<probe>_probing_results.csv

with the headline metric ``average_precision`` plus roc_auc / precision / recall / f1, the
per-split sample counts, the naive test-set positive rate (``test_pos_ratio``, the AP baseline)
and the probe (``probe``; ``best_params`` for the tuned MLP).
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATASETS = PROJECT_ROOT / "data" / "input_datasets"
OUTPUT_DIR = PROJECT_ROOT / "output" / "linear_probing"

PROBES = ("logreg_std", "mlp")
DATASETS = ("hancock", "mimic", "luad_noleak")
COMPETITOR_ROOT = PROJECT_ROOT / "output" / "inductive_competitors"
METHOD_ROOTS = {
    "pome": PROJECT_ROOT / "output" / "inductive_pome14",
    "umap": PROJECT_ROOT / "output" / "inductive",
    "hivae": COMPETITOR_ROOT,
    "saint": COMPETITOR_ROOT,
    "mfae": COMPETITOR_ROOT,
    "tabicl_lf": COMPETITOR_ROOT,
}
METHODS = tuple(METHOD_ROOTS)
# Variable-type subsets written by generate_inductive_embeddings.py --modes.
# `combined` keeps the original mode-less directory layout.
MODES = ("combined", "numeric_only", "cat_only")

# Per-dataset target file and the binary targets to probe (raw column -> display label).
DATASET_TARGETS = {
    "hancock": {
        "file": "hancock_targets.csv",
        "targets": {
            "recurrence": "Recurrence",
            "survival_status": "Survival",
            "rfs_event": "RFS Event",
        },
    },
    "mimic": {
        "file": "mimic_targets.csv",
        "targets": {
            "label_aplasia": "Aplasia",
            "label_nf": "Neutropenic Fever",
        },
    },
    "luad_noleak": {
        "file": "TCGA_LUAD_targets.csv",
        "targets": {
            "Disease Free Status": "Disease Free Status",
            "Disease-specific Survival status": "DSS Status",
            "Progression Free Status": "Progression Free Status",
        },
    },
}


# --- Target loading ----------------------------------------------------------
def load_binary_targets(dataset: str) -> dict[str, pd.Series]:
    """Return {display_label: int-encoded 0/1 Series indexed by sample id}.

    LabelEncoder is fit on the full non-null column so the 0/1 meaning is
    consistent across all splits.
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
                  dims, splits, runs, mode: str = "combined"):
    """Yield (split, dim, run, train_csv, test_csv) for existing run pairs.

    Mirrors ``generate_inductive_embeddings.run_paths``: ``combined`` lives
    directly under ``{method}/{dataset}``, restricted modes one level deeper.
    Because the mode directories are not named ``split_*`` they never leak into
    the combined listing.
    """
    base = emb_root / method / dataset
    if mode != "combined":
        base = base / mode
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
INNER_CV_FOLDS = 3


def make_classifier(probe: str):
    """The probe estimator; both standardize the features with the training statistics."""
    if probe == "logreg_std":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(penalty="l2", solver="liblinear",
                                                    max_iter=1000))])
    if probe == "mlp":
        est = Pipeline([("scale", StandardScaler()),
                        ("clf", MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True,
                                              validation_fraction=0.1, max_iter=500,
                                              random_state=0))])
        cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=0)
        return GridSearchCV(est, {"clf__alpha": [1e-4, 1e-2, 1.0]},
                            scoring="average_precision", cv=cv, n_jobs=1)
    raise ValueError(probe)


def probe_one(train_emb: pd.DataFrame, test_emb: pd.DataFrame,
              y_full: pd.Series, probe: str = "logreg_std") -> dict | None:
    """Fit the probe on train embedding, evaluate on test embedding for one target.

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

    # A probe cannot consume non-finite embedding coordinates, which the inductive
    # transform can occasionally produce (notably UMAP for a held-out sample with no
    # usable neighbours). Drop those rows from both fit and evaluation and record how
    # many were dropped so it stays visible.
    tr_finite = np.isfinite(X_tr).all(axis=1)
    te_finite = np.isfinite(X_te).all(axis=1)
    n_train_nonfinite = int((~tr_finite).sum())
    n_test_nonfinite = int((~te_finite).sum())
    X_tr, y_tr = X_tr[tr_finite], y_tr[tr_finite]
    X_te, y_te = X_te[te_finite], y_te[te_finite]

    if len(y_tr) == 0 or len(y_te) == 0 or len(np.unique(y_tr)) < 2:
        return None

    clf = make_classifier(probe)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    best = getattr(clf, "best_params_", None)

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
        **({"best_params": repr(best)} if best is not None else {}),
    }


def probe_pair(train_csv, test_csv, targets, probe):
    """Probe every target on one (train, test) embedding pair -> [(label, metrics)]."""
    # set here as well: joblib worker processes do not inherit main()'s filters
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    # sklearn >= 1.8 deprecates penalty="l2"
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*'penalty' was deprecated")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # MLP inner-CV fits
    train_emb = pd.read_csv(train_csv, index_col=0)
    test_emb = pd.read_csv(test_csv, index_col=0)
    train_emb.index = train_emb.index.astype(str)
    test_emb.index = test_emb.index.astype(str)
    out = []
    for label, y_full in targets.items():
        metrics = probe_one(train_emb, test_emb, y_full, probe)
        if metrics is not None:
            out.append((label, metrics))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["combined"],
                        help="variable-type subsets to probe (default: combined; "
                             "modes without embeddings are skipped)")
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="dims to probe (default: all found)")
    parser.add_argument("--splits", nargs="+", type=int, default=None,
                        help="split ids to probe (default: all found)")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="run ids to probe (default: all found)")
    parser.add_argument("--emb-root", type=Path, default=None,
                        help="embedding root for every method (default: METHOD_ROOTS)")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: output/linear_probing/inductive_<probe>_probing_results.csv")
    parser.add_argument("--probe", choices=PROBES, default="logreg_std")
    parser.add_argument("--jobs", type=int, default=1,
                        help="embedding pairs probed in parallel")
    args = parser.parse_args()
    if args.output is None:
        args.output = OUTPUT_DIR / f"inductive_{args.probe}_probing_results.csv"

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    print(f"Datasets: {args.datasets} | methods: {args.methods} | "
          f"modes: {args.modes} | probe: {args.probe}")

    rows = []
    for dataset in args.datasets:
        targets = load_binary_targets(dataset)
        print(f"\n[{dataset}] targets: {list(targets)}")
        for method in args.methods:
            emb_root = args.emb_root or METHOD_ROOTS[method]
            for mode in args.modes:
                run_pairs = list(discover_runs(
                    emb_root, method, dataset,
                    args.dims, args.splits, args.runs, mode))
                if not run_pairs:
                    print(f"  [warn] no {method}/{mode} embeddings found under {emb_root}")
                    continue
                n_nonfinite = 0  # non-finite embedding rows dropped (train+test)
                results = Parallel(n_jobs=args.jobs)(
                    delayed(probe_pair)(train_csv, test_csv, targets, args.probe)
                    for _, _, _, train_csv, test_csv in run_pairs)
                for (split, dim, run, _, _), pair in zip(run_pairs, results):
                    for label, metrics in pair:
                        n_nonfinite += (metrics["n_train_nonfinite"]
                                        + metrics["n_test_nonfinite"])
                        rows.append({
                            "dataset": dataset, "method": method.upper(),
                            "mode": mode, "target": label, "dim": dim,
                            "split": split, "run": run,
                            **metrics, "probe": args.probe,
                        })
                msg = (f"  {method:9s} {mode:12s}: probed {len(run_pairs)} embedding "
                       f"pairs x {len(targets)} targets")
                if n_nonfinite:
                    msg += (f"  [warn] dropped {n_nonfinite} non-finite "
                            f"embedding rows (see n_*_nonfinite columns)")
                print(msg)

    if not rows:
        print("\nNo results produced -- check the embedding roots.")
        return

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(result_df)} rows to {args.output}")

    # Compact summary: mean AP across splits x runs per dataset/target/dim/method.
    print("\n=== Mean Average Precision (across splits x runs) ===")
    group_cols = ["dataset", "target", "dim", "method"]
    if result_df["mode"].nunique() > 1:
        group_cols.insert(2, "mode")
    summary = (result_df
               .groupby(group_cols)
               ["average_precision"].mean().round(4)
               .unstack("method"))
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summary)


if __name__ == "__main__":
    main()
