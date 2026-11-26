"""Inductive linear-probing of the POME / UMAP embeddings produced by
``scripts/generate_inductive_embeddings.py``, of the tabular competitors (HI-VAE, SAINT,
missForest + autoencoder) produced by ``scripts/embed_{hivae,saint,mfae}_inductive.py``, and
of the target-conditioned TabICL embeddings of ``scripts/embed_tabicl_inductive.py`` (one
embedding per target under ``tabicl/<dataset>/target_<target>/``, each probed for its own
target only).

For every computed embedding (dataset x method x split x dim x run) a logistic
regression is *fit on the training-split embedding* and *evaluated on the
held-out test-split embedding*, for each binary held-out target variable of the
dataset. This mirrors the targets and classifier configuration of the transductive
notebooks ``src/pome_evaluation/analyze_<DATASET>_embedding_separability.ipynb``
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

Non-linear probing (``--probe xgboost``)
----------------------------------------
Replaces the logistic regression with a gradient-boosted tree ensemble
(XGBClassifier, see ``XGB_PARAMS``): 300 trees of depth <= 3, learning rate 0.05,
row and column subsampling 0.8, hist tree method. The hyperparameters are fixed
and the same for every dataset, method, target and embedding size (no tuning, so
no test information enters the probe). The seed is fixed (0), and the trees are
deterministic. Everything else -- targets, splits, train/test protocol, metrics
and output schema -- is identical to the linear probe, so
``plot_inductive_linear_probing.py`` draws the result in the same layout. The
default output is then ``inductive_xgboost_probing_results.csv``, and every
output row records its probe in the ``probe`` column. xgboost is not in images
built before this option; install it with
``pip install --no-deps --target $(ws_find pome_eval)/pylib xgboost==2.1.3``
inside the image (run.sh / submit.sh put that directory on PYTHONPATH).

Further non-linear probes (``--probe svm | knn | mlp``)
-------------------------------------------------------
Chosen for the geometry of POME's embeddings, which are L2-normalized (every sample on the unit
sphere, per-dimension std ~0.1, ~6 effective dimensions), and applied identically to every method.
Each one tunes its hyperparameters by an inner stratified 3-fold CV (seed 0, scoring = average
precision) on the *training split only*, then refits on the whole training split:

    svm : RBF-kernel SVC on standardized features (fit on train); C in {0.1, 1, 10, 100},
          gamma = "scale". Test scores = decision function (AP and ROC AUC are rank-based).
    knn : k-nearest neighbours, distance-weighted, unscaled features; metric in
          {cosine, euclidean} x k in {5, 10, 20, 40, 80, 160} (cosine = angular distance on POME's
          sphere).
    mlp : one hidden layer of 64 ReLU units on standardized features, Adam, early stopping
          on a 10% validation split of the training data, seed 0; L2 penalty alpha in
          {1e-4, 1e-2, 1}.

Default outputs: ``inductive_<probe>_probing_results.csv``. ``--standardize`` is ignored by these
probes (their scaling is part of the probe).

Variable-type modes (``--modes``)
---------------------------------
``scripts/generate_inductive_embeddings.py --modes`` can embed a variable-type
subset of the same splits (``numeric_only`` / ``cat_only``) next to the default
``combined`` embeddings. All discovered modes are probed by default and tagged
in the ``mode`` column, so the data-integration comparison (does combining
continuous and categorical variables beat either type alone?) uses the *exact*
same targets, classifier and splits as the headline combined probing --
``scripts/plot_inductive_type_combination.py`` draws it.

Run from the project root (any env with pandas + scikit-learn):

    conda run -n torch python scripts/linear_probe_inductive_embeddings.py
    conda run -n torch python scripts/linear_probe_inductive_embeddings.py --datasets hancock --dims 32
    conda run -n torch python scripts/linear_probe_inductive_embeddings.py --modes combined numeric_only cat_only
    container/run.sh scripts/linear_probe_inductive_embeddings.py --probe xgboost --jobs 32

Output: one tidy long-format CSV (one row per
dataset/method/mode/target/dim/split/run)

    output/linear_probing/inductive_linear_probing_results.csv

with the headline metric ``average_precision`` plus roc_auc / precision /
recall / f1, the per-split sample counts and the naive test-set positive rate
(``test_pos_ratio``) as the AP baseline.
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATASETS = PROJECT_ROOT / "data" / "input_datasets"
DEFAULT_EMB_ROOT = PROJECT_ROOT / "output" / "inductive"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "linear_probing" / \
    "inductive_linear_probing_results.csv"

PROBES = ("logreg", "xgboost", "svm", "knn", "mlp")
# Fixed, untuned XGBoost probe (shallow trees, shrinkage, subsampling: a standard
# small-data configuration). One thread per fit; --jobs parallelizes over embeddings.
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                  reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
                  random_state=0, n_jobs=1)

# luad_noleak = luad without its outcome/follow-up variables (scripts/make_luad_noleak_splits.py)
DATASETS = ("hancock", "luad", "mimic", "luad_noleak")
# hivae / saint / mfae: the tabular competitors of scripts/embed_{hivae,saint,mfae}_inductive.py,
# written under output/inductive_competitors (probe them with --emb-root there).
# tabicl_lf: label-free TabICL v1.1 row embeddings (scripts/embed_tabicl_labelfree_inductive.py).
METHODS = ("pome", "umap", "hivae", "saint", "mfae", "tabicl", "tabicl_lf")
# Target-conditioned methods (scripts/embed_tabicl_inductive.py): one embedding per probing
# target under <method>/<dataset>/target_<slug>/, probed for that target only.
TARGET_CONDITIONED_METHODS = ("tabicl",)
# Variable-type subsets written by generate_inductive_embeddings.py --modes.
# `combined` keeps the original mode-less directory layout.
MODES = ("combined", "numeric_only", "cat_only")

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

DATASET_TARGETS["luad_noleak"] = DATASET_TARGETS["luad"]   # same samples and targets

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
def target_slug(label: str) -> str:
    """Directory name part of a target's display label ('DSS Status' -> 'dss_status')."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def discover_runs(emb_root: Path, method: str, dataset: str,
                  dims, splits, runs, mode: str = "combined", target: str | None = None):
    """Yield (split, dim, run, train_csv, test_csv) for existing run pairs.

    Mirrors ``generate_inductive_embeddings.run_paths``: ``combined`` lives
    directly under ``{method}/{dataset}``, restricted modes one level deeper.
    Because the mode directories are not named ``split_*`` they never leak into
    the combined listing.
    """
    base = emb_root / method / dataset
    if mode != "combined":
        base = base / mode
    if target is not None:      # target-conditioned method: one directory per target
        base = base / ("target_" + target_slug(target))
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
    """The probe estimator; svm / knn / mlp are inner-CV grid searches on the training split."""
    if probe == "logreg":
        return LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
    if probe == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**XGB_PARAMS)
    if probe == "svm":
        est = Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf", gamma="scale"))])
        grid = {"clf__C": [0.1, 1, 10, 100]}
    elif probe == "knn":
        est = KNeighborsClassifier(weights="distance", algorithm="brute")
        grid = {"metric": ["cosine", "euclidean"], "n_neighbors": [5, 10, 20, 40, 80, 160]}
    elif probe == "mlp":
        est = Pipeline([("scale", StandardScaler()),
                        ("clf", MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True,
                                              validation_fraction=0.1, max_iter=500,
                                              random_state=0))])
        grid = {"clf__alpha": [1e-4, 1e-2, 1.0]}
    else:
        raise ValueError(probe)
    cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=0)
    return GridSearchCV(est, grid, scoring="average_precision", cv=cv, n_jobs=1)


def probe_one(train_emb: pd.DataFrame, test_emb: pd.DataFrame,
              y_full: pd.Series, standardize: bool,
              probe: str = "logreg") -> dict | None:
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

    if standardize and probe in ("logreg", "xgboost"):
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    # k-NN needs k < n_train (per inner-CV fold); drop the grid values that do not fit
    clf = make_classifier(probe)
    if probe == "knn":
        k_max = len(y_tr) * (INNER_CV_FOLDS - 1) // INNER_CV_FOLDS - 1
        clf.param_grid["n_neighbors"] = [k for k in clf.param_grid["n_neighbors"] if k <= k_max]
    clf.fit(X_tr, y_tr)
    if probe == "svm":
        proba = clf.decision_function(X_te)        # ranking score; no Platt scaling needed
        pred = clf.predict(X_te).astype(int)
    else:
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


def probe_pair(train_csv, test_csv, targets, standardize, probe):
    """Probe every target on one (train, test) embedding pair -> [(label, metrics)]."""
    # set here as well: joblib worker processes do not inherit main()'s filters
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    # sklearn >= 1.8 deprecates penalty="l2"; kept for the notebooks' exact configuration
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*'penalty' was deprecated")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # MLP inner-CV fits
    train_emb = pd.read_csv(train_csv, index_col=0)
    test_emb = pd.read_csv(test_csv, index_col=0)
    train_emb.index = train_emb.index.astype(str)
    test_emb.index = test_emb.index.astype(str)
    out = []
    for label, y_full in targets.items():
        metrics = probe_one(train_emb, test_emb, y_full, standardize, probe)
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
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=list(MODES),
                        help="variable-type subsets to probe (default: all; "
                             "modes without embeddings are skipped)")
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="dims to probe (default: all found)")
    parser.add_argument("--splits", nargs="+", type=int, default=None,
                        help="split ids to probe (default: all found)")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="run ids to probe (default: all found)")
    parser.add_argument("--emb-root", type=Path, default=DEFAULT_EMB_ROOT)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: inductive_linear_probing_results.csv "
                             "(logreg) / inductive_xgboost_probing_results.csv")
    parser.add_argument("--probe", choices=PROBES, default="logreg",
                        help="logreg (linear, default) or a non-linear probe: xgboost, "
                             "svm (RBF), knn, mlp")
    parser.add_argument("--jobs", type=int, default=1,
                        help="embedding pairs probed in parallel")
    parser.add_argument("--standardize", action="store_true",
                        help="z-score embedding features (fit on train) "
                             "before logistic regression (default: off, "
                             "matching the notebooks)")
    args = parser.parse_args()
    if args.output is None:
        args.output = (DEFAULT_OUTPUT if args.probe == "logreg" else
                       DEFAULT_OUTPUT.with_name(f"inductive_{args.probe}_probing_results.csv"))

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    print(f"Embedding root: {args.emb_root}")
    print(f"Datasets: {args.datasets} | methods: {args.methods} | "
          f"modes: {args.modes} | standardize: {args.standardize} | "
          f"probe: {args.probe}")

    rows = []
    for dataset in args.datasets:
        targets = load_binary_targets(dataset)
        print(f"\n[{dataset}] targets: {list(targets)}")
        for method in args.methods:
            for mode in args.modes:
                # (target directory, targets probed on its embeddings)
                if method in TARGET_CONDITIONED_METHODS:
                    if mode != "combined":
                        continue
                    groups = [(label, {label: y}) for label, y in targets.items()]
                else:
                    groups = [(None, targets)]
                run_pairs, pair_targets = [], []
                for target_dir, group_targets in groups:
                    found = list(discover_runs(
                        args.emb_root, method, dataset,
                        args.dims, args.splits, args.runs, mode, target_dir))
                    run_pairs += found
                    pair_targets += [group_targets] * len(found)
                if not run_pairs:
                    print(f"  [warn] no {method}/{mode} embeddings found")
                    continue
                n_done = 0
                n_nonfinite = 0  # non-finite embedding rows dropped (train+test)
                results = Parallel(n_jobs=args.jobs)(
                    delayed(probe_pair)(train_csv, test_csv, pair_tg,
                                        args.standardize, args.probe)
                    for (_, _, _, train_csv, test_csv), pair_tg
                    in zip(run_pairs, pair_targets))
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
                    n_done += 1
                msg = (f"  {method:4s} {mode:12s}: probed {n_done} embedding "
                       "pairs x "
                       + ("1 own target" if method in TARGET_CONDITIONED_METHODS
                          else f"{len(targets)} targets"))
                if n_nonfinite:
                    msg += (f"  [warn] dropped {n_nonfinite} non-finite "
                            f"embedding rows (see n_*_nonfinite columns)")
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
