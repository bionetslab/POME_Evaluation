"""Linear-probe the POME inductive embeddings at each training-epoch snapshot
produced by ``scripts/generate_inductive_epoch_snapshots.py``, to measure how
probing quality evolves with training duration.

For every ``epoch_{E}`` snapshot directory this reuses the exact probing logic of
``scripts/linear_probe_inductive_embeddings.py`` -- a logistic regression fit on
the train-split embedding and evaluated on the held-out test-split embedding, per
binary target -- and stamps each result with its ``epoch``. The snapshot layout
(``.../epoch_{E}/pome/{dataset}/split_{NN}/dim_{D}/run_{R}_{train,test}.csv``) is
structurally identical to ``output/inductive``, so ``discover_runs`` works on each
epoch directory unchanged.

Run from the project root (any env with pandas + scikit-learn):

    conda run -n torch python scripts/linear_probe_inductive_epoch_snapshots.py
    conda run -n torch python scripts/linear_probe_inductive_epoch_snapshots.py \
        --datasets hancock --dims 32

Each row is also stamped with the label-free **RankMe** effective rank (Garrido
et al. 2023) of the train and test embedding matrices (``rankme_train`` /
``rankme_test``) -- a candidate unsupervised early-stopping metric. RankMe is a
property of the embedding, not of the target, so it is identical across the
target rows of a given (epoch, dataset, dim, split, run).

Output: one tidy long-format CSV with an ``epoch`` column (one row per
epoch/dataset/target/dim/split/run), ready to plot quality vs. epochs:

    output/linear_probing/inductive_epoch_snapshots_results.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

# Reuse the probing primitives so the methodology is identical to the
# single-shot inductive probing script.
from linear_probe_inductive_embeddings import (
    DATASETS, discover_runs, load_binary_targets, probe_one)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMB_ROOT = PROJECT_ROOT / "output" / "inductive_epochs"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "linear_probing" / \
    "inductive_epoch_snapshots_results.csv"


def rankme(emb: pd.DataFrame, eps: float = 1e-7) -> float:
    """RankMe effective rank (Garrido et al. 2023): the entropy of the normalized
    singular-value spectrum of the embedding matrix, ``exp(-sum p_k log p_k)`` with
    ``p_k = sigma_k / sum(sigma) + eps``. A label-free proxy for representation
    quality (higher = the embedding uses more of its dimensions). Non-finite rows
    are dropped; returns NaN if fewer than two usable rows remain.
    """
    Z = emb.to_numpy(dtype=np.float64)
    Z = Z[np.isfinite(Z).all(axis=1)]
    if Z.shape[0] < 2:
        return float("nan")
    sigma = np.linalg.svd(Z, compute_uv=False)
    p = sigma / (sigma.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def discover_epoch_dirs(emb_root: Path, epochs) -> list[tuple[int, Path]]:
    """Return sorted (epoch, dir) for each epoch_* snapshot dir under emb_root."""
    out = []
    for d in sorted(emb_root.glob("epoch_*")):
        try:
            e = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if epochs is not None and e not in epochs:
            continue
        out.append((e, d))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--epochs", nargs="+", type=int, default=None,
                        help="epoch snapshots to probe (default: all found)")
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="dims to probe (default: all found)")
    parser.add_argument("--splits", nargs="+", type=int, default=None,
                        help="split ids to probe (default: all found)")
    parser.add_argument("--runs", nargs="+", type=int, default=None,
                        help="run ids to probe (default: all found)")
    parser.add_argument("--emb-root", type=Path, default=DEFAULT_EMB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--standardize", action="store_true",
                        help="z-score embedding features (fit on train) before "
                             "logistic regression (default: off)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    epoch_dirs = discover_epoch_dirs(args.emb_root, args.epochs)
    if not epoch_dirs:
        print(f"No epoch_* snapshot dirs under {args.emb_root}.")
        return

    print(f"Embedding root: {args.emb_root}")
    print(f"Epochs: {[e for e, _ in epoch_dirs]} | datasets: {args.datasets} | "
          f"standardize: {args.standardize}")

    rows = []
    for epoch, epoch_dir in epoch_dirs:
        print(f"\n=== epoch {epoch:04d} ===")
        for dataset in args.datasets:
            targets = load_binary_targets(dataset)
            run_pairs = list(discover_runs(
                epoch_dir, "pome", dataset, args.dims, args.splits, args.runs))
            if not run_pairs:
                print(f"  [warn] no pome embeddings for {dataset}")
                continue
            n_done = 0
            for split, dim, run, train_csv, test_csv in run_pairs:
                train_emb = pd.read_csv(train_csv, index_col=0)
                test_emb = pd.read_csv(test_csv, index_col=0)
                train_emb.index = train_emb.index.astype(str)
                test_emb.index = test_emb.index.astype(str)
                # RankMe is target-independent -> compute once per embedding pair.
                rk_train = rankme(train_emb)
                rk_test = rankme(test_emb)
                for label, y_full in targets.items():
                    metrics = probe_one(train_emb, test_emb, y_full,
                                        args.standardize)
                    if metrics is None:
                        continue
                    rows.append({
                        "epoch": epoch, "dataset": dataset, "method": "POME",
                        "target": label, "dim": dim, "split": split,
                        "run": run, "rankme_train": rk_train,
                        "rankme_test": rk_test, **metrics,
                    })
                n_done += 1
            print(f"  {dataset:8s}: probed {n_done} embedding pairs "
                  f"x {len(targets)} targets")

    if not rows:
        print("\nNo results produced -- check --emb-root has snapshot embeddings.")
        return

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(result_df)} rows to {args.output}")

    # Compact summary: mean AP across splits x runs per dataset/target/dim/epoch.
    print("\n=== Mean Average Precision by epoch (across splits x runs) ===")
    summary = (result_df
               .groupby(["dataset", "target", "dim", "epoch"])
               ["average_precision"].mean().round(4)
               .unstack("epoch"))
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary)


if __name__ == "__main__":
    main()
