"""Linear-probing of the POME / UMAP embeddings trained on the *full* datasets,
evaluated on the *same* train/test splits as
``scripts/linear_probe_inductive_embeddings.py``.

The inductive script fits a logistic regression on a training-split embedding
that was produced *inductively* (encoder fit on the train split, test samples
transformed by the frozen encoder) and evaluates it on the held-out test-split
embedding. This companion script asks the transductive counterpart question:
how well do the embeddings trained on the *whole* dataset separate the same
held-out targets, under the *identical* train/test partitions?

For every full-dataset embedding (dataset x method x dim x run, the files under
``data/embeddings/<DATASET>/embeddings``) and every train/test split under
``data/train_test_splits/<DATASET>/`` we simply *row-subset* the full-dataset
embedding to the split's train / test samples, fit a logistic regression on the
train rows and evaluate on the test rows -- reusing the inductive script's
``probe_one`` so the metrics, target encoding and classifier configuration are
byte-for-byte identical. The only difference from the inductive analysis is the
source of the embedding coordinates (trained-on-all vs trained-on-train-split).

Because the two scripts share the same split ids, targets and output schema, the
full-dataset scores can be drawn as *additional boxes* alongside the inductive
ones by ``scripts/plot_inductive_linear_probing.py``.

Full-dataset embedding files (combined, i.e. all variable types), per run ``R``
in 0..9 and embedding size ``D`` in {16, 32, 64, 128}:

    POME : data/embeddings/<PREFIX>/embeddings/<PREFIX>_samples_<D>_<R>.tsv
    UMAP : data/embeddings/<PREFIX>/embeddings/<PREFIX>_UMAP_<D>_<R>.csv

with ``<PREFIX>`` = HANCOCK / TCGA_LUAD / MIMIC. Train/test sample membership is
read from the split files (identical across the POME/UMAP formats), so the exact
same samples are held out as in the inductive analysis -- including MIMIC's
patient-level grouping, which is already baked into the splits.

The classifier matches the notebooks / inductive script:
LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000); embeddings
are NOT rescaled (use --standardize to z-score features before fitting).

Run from the project root (any env with pandas + scikit-learn):

    python scripts/linear_probe_full_dataset_embeddings.py
    python scripts/linear_probe_full_dataset_embeddings.py --datasets hancock --dims 64

Output: one tidy long-format CSV (one row per dataset/method/target/dim/split/run)

    output/linear_probing/full_dataset_linear_probing_results.csv

with the same columns as the inductive results (headline metric
``average_precision``, plus roc_auc / precision / recall / f1, sample counts and
the test-set positive rate ``test_pos_ratio`` as the AP baseline).
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

# Reuse the inductive script's target definitions/encoding and the probing
# routine so the two analyses are identical apart from the embedding source.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe_inductive_embeddings import (  # noqa: E402
    load_binary_targets, probe_one)

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "embeddings"
SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "linear_probing" / \
    "full_dataset_linear_probing_results.csv"

DATASETS = ("hancock", "luad", "mimic")
METHODS = ("pome", "umap")

# Directory prefix / file stem for each dataset's full-dataset embedding files.
EMB_PREFIX = {"hancock": "HANCOCK", "luad": "TCGA_LUAD", "mimic": "MIMIC"}

_SPLIT_MEMBER_RE = re.compile(r"^split_(\d+)_train_umap\.csv$")


# --- Split membership --------------------------------------------------------
def discover_split_members(dataset: str):
    """Return {split_id: (train_ids, test_ids)} as lists of str sample ids.

    Membership is read from the UMAP split CSVs (sample-indexed); it is identical
    to the POME split files, so the same samples are held out regardless of the
    embedding method being probed.
    """
    members = {}
    split_dir = SPLITS_ROOT / dataset
    if not split_dir.is_dir():
        return members
    for train_csv in sorted(split_dir.glob("split_*_train_umap.csv")):
        m = _SPLIT_MEMBER_RE.match(train_csv.name)
        if not m:
            continue
        split = int(m.group(1))
        test_csv = train_csv.with_name(
            train_csv.name.replace("_train_umap.csv", "_test_umap.csv"))
        if not test_csv.exists():
            continue
        train_ids = pd.read_csv(train_csv, index_col=0).index.astype(str)
        test_ids = pd.read_csv(test_csv, index_col=0).index.astype(str)
        members[split] = (list(train_ids), list(test_ids))
    return members


# --- Embedding discovery -----------------------------------------------------
def emb_dir(dataset: str) -> Path:
    return EMBEDDINGS_ROOT / EMB_PREFIX[dataset] / "embeddings"


def emb_path(dataset: str, method: str, dim: int, run: int) -> Path:
    prefix = EMB_PREFIX[dataset]
    if method == "pome":
        return emb_dir(dataset) / f"{prefix}_samples_{dim}_{run}.tsv"
    return emb_dir(dataset) / f"{prefix}_UMAP_{dim}_{run}.csv"


def discover_dims_runs(dataset: str, method: str):
    """Return {dim: [run, ...]} for the combined full-dataset embedding files.

    Only the combined embeddings are used (``*_cat_only_*`` / ``*_numeric_only_*``
    single-modality variants are excluded).
    """
    prefix = EMB_PREFIX[dataset]
    if method == "pome":
        pattern = re.compile(rf"^{prefix}_samples_(\d+)_(\d+)\.tsv$")
    else:
        pattern = re.compile(rf"^{prefix}_UMAP_(\d+)_(\d+)\.csv$")
    found: dict[int, list[int]] = {}
    d = emb_dir(dataset)
    if not d.is_dir():
        return found
    for p in d.iterdir():
        m = pattern.match(p.name)
        if m:
            dim, run = int(m.group(1)), int(m.group(2))
            found.setdefault(dim, []).append(run)
    return {dim: sorted(runs) for dim, runs in sorted(found.items())}


def load_embedding(dataset: str, method: str, dim: int, run: int) -> pd.DataFrame:
    path = emb_path(dataset, method, dim, run)
    sep = "\t" if method == "pome" else ","
    emb = pd.read_csv(path, sep=sep, index_col=0)
    emb.index = emb.index.astype(str)
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="embedding sizes to probe (default: all found)")
    parser.add_argument("--splits", nargs="+", type=int, default=None,
                        help="split ids to probe (default: all found)")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="run ids to probe (default: all found)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--standardize", action="store_true",
                        help="z-score embedding features (fit on train) before "
                             "logistic regression (default: off)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    print(f"Embeddings root: {EMBEDDINGS_ROOT}")
    print(f"Splits root: {SPLITS_ROOT}")
    print(f"Datasets: {args.datasets} | methods: {args.methods} | "
          f"standardize: {args.standardize}")

    rows = []
    for dataset in args.datasets:
        targets = load_binary_targets(dataset)
        members = discover_split_members(dataset)
        if args.splits is not None:
            members = {s: v for s, v in members.items() if s in args.splits}
        if not members:
            print(f"\n[{dataset}] [warn] no train/test splits found -- skipping")
            continue
        print(f"\n[{dataset}] targets: {list(targets)} | "
              f"splits: {sorted(members)}")

        for method in args.methods:
            dims_runs = discover_dims_runs(dataset, method)
            if not dims_runs:
                print(f"  [warn] no {method} full-dataset embeddings found")
                continue
            n_probed = 0
            n_nonfinite = 0
            for dim, runs in dims_runs.items():
                if args.dims is not None and dim not in args.dims:
                    continue
                for run in runs:
                    if args.runs is not None and run not in args.runs:
                        continue
                    emb = load_embedding(dataset, method, dim, run)
                    for split, (train_ids, test_ids) in members.items():
                        # Row-subset the full-dataset embedding to this split's
                        # train / test samples (the identical partition the
                        # inductive analysis uses).
                        train_emb = emb.loc[emb.index.intersection(train_ids)]
                        test_emb = emb.loc[emb.index.intersection(test_ids)]
                        for label, y_full in targets.items():
                            metrics = probe_one(train_emb, test_emb, y_full,
                                                args.standardize)
                            if metrics is None:
                                continue
                            n_nonfinite += (metrics["n_train_nonfinite"]
                                            + metrics["n_test_nonfinite"])
                            rows.append({
                                "dataset": dataset, "method": method.upper(),
                                # Only the combined embeddings exist in full-
                                # dataset form; the column keeps the schema
                                # identical to the inductive results.
                                "mode": "combined",
                                "target": label, "dim": dim, "split": split,
                                "run": run, **metrics,
                            })
                    n_probed += 1
            msg = (f"  {method:4s}: probed {n_probed} embeddings "
                   f"x {len(members)} splits x {len(targets)} targets")
            if n_nonfinite:
                msg += (f"  [warn] dropped {n_nonfinite} non-finite embedding "
                        f"rows (see n_*_nonfinite columns)")
            print(msg)

    if not rows:
        print("\nNo results produced -- check the embeddings/splits directories.")
        return

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(result_df)} rows to {args.output}")

    print("\n=== Mean Average Precision (across splits x runs) ===")
    summary = (result_df
               .groupby(["dataset", "target", "dim", "method"])
               ["average_precision"].mean().round(4)
               .unstack("method"))
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summary)


if __name__ == "__main__":
    main()
