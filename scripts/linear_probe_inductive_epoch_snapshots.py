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

Each row is also stamped with label-free representation metrics of the embeddings
(target-independent, so identical across the target rows of a given epoch/dataset/
dim/split/run):
  - ``rankme_train`` / ``rankme_test`` -- RankMe effective rank (Garrido et al.
    2023) of the train / inductive-test embeddings.
  - ``overfit_index`` -- the sample-matched, ceiling-normalized train-test RankMe
    gap (see ``overfit_index``): an N/D-aware overfitting signal that removes the
    sample-count confound of the raw train-test gap.

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


def _rankme_matrix(Z, eps: float = 1e-7) -> float:
    """RankMe effective rank of a raw embedding matrix ``Z`` (n x d): entropy of
    the normalized singular-value spectrum, ``exp(-sum p_k log p_k)``. Non-finite
    rows are dropped; NaN if fewer than two usable rows remain.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z[np.isfinite(Z).all(axis=1)]
    if Z.shape[0] < 2:
        return float("nan")
    sigma = np.linalg.svd(Z, compute_uv=False)
    p = sigma / (sigma.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def rankme(emb: pd.DataFrame, eps: float = 1e-7) -> float:
    """RankMe effective rank (Garrido et al. 2023) of an embedding DataFrame. A
    label-free proxy for representation quality (higher = the embedding uses more
    of its dimensions). Bounded above by ``min(n_samples, dim)``.
    """
    return _rankme_matrix(emb.to_numpy(dtype=np.float64), eps)


def overfit_index(train_emb: pd.DataFrame, test_emb: pd.DataFrame,
                  n_draws: int = 10, seed: int = 0) -> float:
    """Sample-matched, ceiling-normalized train-test RankMe gap.

    The raw gap ``RankMe(train) - RankMe(test)`` is confounded: RankMe is capped by
    ``min(n, dim)`` and biased by ``n``, and ``n_train >> n_test``, so train scores
    higher even with no overfitting. Here both matrices are reduced to the same
    ``n = min(n_train, n_test)`` rows (the larger one subsampled, averaged over
    ``n_draws`` draws) so the cap and finite-sample bias match, and the gap is
    normalized by the shared ceiling ``min(n, dim)``:

        (RankMe_train@n - RankMe_test@n) / min(n, dim)

    The residual reflects genuine geometric divergence (overfitting onset) rather
    than the sample-count asymmetry. Returns NaN if fewer than two usable rows.
    """
    Xtr = train_emb.to_numpy(dtype=np.float64)
    Xtr = Xtr[np.isfinite(Xtr).all(axis=1)]
    Xte = test_emb.to_numpy(dtype=np.float64)
    Xte = Xte[np.isfinite(Xte).all(axis=1)]
    n = min(Xtr.shape[0], Xte.shape[0])
    d = Xtr.shape[1]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)

    def matched(X):
        if X.shape[0] == n:
            return _rankme_matrix(X)
        vals = [_rankme_matrix(X[rng.choice(X.shape[0], n, replace=False)])
                for _ in range(n_draws)]
        return float(np.nanmean(vals))

    rk_train, rk_test = matched(Xtr), matched(Xte)
    ceiling = min(n, d)
    if not (np.isfinite(rk_train) and np.isfinite(rk_test)) or ceiling < 1:
        return float("nan")
    return (rk_train - rk_test) / ceiling


# --- Label-free stopping rules ----------------------------------------------
# All three operate on per-epoch aggregated curves (mean over splits x runs) and
# return the selected epoch. The first two are unsupervised (RankMe only); the
# oracle uses the probe AP and is only for comparison -- you couldn't use it at
# training time.

def stop_rankme_plateau(epochs, rankme_test, rel_tol: float = 0.2) -> int:
    """RankMe-plateau rule: stop when the effective rank stops meaningfully
    growing -- the first epoch whose marginal RankMe gain drops below ``rel_tol``
    times the largest gain observed (the knee of the curve). Returns the last
    epoch if RankMe never flattens (still climbing = the overshoot case).
    """
    e = np.asarray(epochs)
    d = np.diff(np.asarray(rankme_test, dtype=float))
    if d.size == 0 or not np.isfinite(d).any() or np.nanmax(d) <= 0:
        return int(e[-1])
    thresh = rel_tol * np.nanmax(d)
    for i, gain in enumerate(d):
        if np.isfinite(gain) and gain < thresh:
            return int(e[i + 1])
    return int(e[-1])


def stop_traintest_gap(epochs, rankme_train, rankme_test,
                       rel_tol: float = 0.15) -> int:
    """Train-test gap rule (overfitting guard): stop when the (train - test)
    RankMe gap, normalized by test RankMe, first rises ``rel_tol`` above its
    running minimum -- i.e. the training-graph spread starts outstripping the
    inductive spread. Returns the last epoch if the gap never widens.
    """
    e = np.asarray(epochs)
    tr = np.asarray(rankme_train, dtype=float)
    te = np.asarray(rankme_test, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = (tr - te) / te
    run_min = np.inf
    for i, g in enumerate(gap):
        if not np.isfinite(g):
            continue
        if g < run_min:
            run_min = g
        elif g > run_min + rel_tol:
            return int(e[i])
    return int(e[-1])


def stop_overfit_index(epochs, overfit, rel_tol: float = 0.05) -> int:
    """Sample-matched-gap rule: stop when the sample-matched, ceiling-normalized
    train-test RankMe gap (``overfit_index``) first rises ``rel_tol`` above its
    running minimum -- the onset of genuine overfitting, with the sample-count
    confound of the raw gap removed. Returns the last epoch if it never widens.
    """
    e = np.asarray(epochs)
    g = np.asarray(overfit, dtype=float)
    run_min = np.inf
    for i, v in enumerate(g):
        if not np.isfinite(v):
            continue
        if v < run_min:
            run_min = v
        elif v > run_min + rel_tol:
            return int(e[i])
    return int(e[-1])


def oracle_ap_peak(epochs, ap) -> int:
    """Reference (supervised) stop: the epoch of maximum mean probe AP."""
    e = np.asarray(epochs)
    a = np.asarray(ap, dtype=float)
    if not np.isfinite(a).any():
        return int(e[-1])
    return int(e[int(np.nanargmax(a))])


def stopping_rules_table(result_df, rankme_plateau_tol: float = 0.2,
                         gap_tol: float = 0.15, overfit_tol: float = 0.05):
    """Per (dataset, dim) selected epoch for each rule, from the aggregated
    (mean over splits x runs) per-epoch AP / RankMe curves.
    """
    out = []
    for (dataset, dim), sub in result_df.groupby(["dataset", "dim"]):
        epochs = sorted(sub["epoch"].unique())
        ap = sub.groupby("epoch")["average_precision"].mean().reindex(epochs)
        # RankMe / overfit_index are target-independent -> one row per (split,
        # run, epoch).
        rk = sub.drop_duplicates(["split", "run", "epoch"])
        rk_te = rk.groupby("epoch")["rankme_test"].mean().reindex(epochs)
        rk_tr = rk.groupby("epoch")["rankme_train"].mean().reindex(epochs)
        e_plateau = stop_rankme_plateau(epochs, rk_te.to_numpy(),
                                        rankme_plateau_tol)
        e_gap = stop_traintest_gap(epochs, rk_tr.to_numpy(), rk_te.to_numpy(),
                                   gap_tol)
        row = {
            "dataset": dataset, "dim": dim,
            "oracle_ap_peak": oracle_ap_peak(epochs, ap.to_numpy()),
            "rankme_plateau": e_plateau,
            "traintest_gap": e_gap,
        }
        # Sample-matched gap rule (only if the column is present).
        if "overfit_index" in sub.columns:
            ov = rk.groupby("epoch")["overfit_index"].mean().reindex(epochs)
            e_gap_matched = stop_overfit_index(epochs, ov.to_numpy(), overfit_tol)
            row["gap_matched"] = e_gap_matched
            row["combined"] = min(e_plateau, e_gap_matched)  # plateau + matched gap
        else:
            row["combined"] = min(e_plateau, e_gap)
        out.append(row)
    return pd.DataFrame(out).sort_values(["dataset", "dim"]).reset_index(drop=True)


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
    parser.add_argument("--rankme-plateau-tol", type=float, default=0.2,
                        help="RankMe-plateau rule: marginal-gain threshold as a "
                             "fraction of the largest gain (default: 0.2)")
    parser.add_argument("--gap-tol", type=float, default=0.15,
                        help="train-test-gap rule: normalized-gap rise above its "
                             "running minimum that triggers a stop (default: 0.15)")
    parser.add_argument("--overfit-tol", type=float, default=0.05,
                        help="sample-matched-gap rule: rise of the ceiling-"
                             "normalized matched gap above its running minimum "
                             "that triggers a stop (default: 0.05)")
    parser.add_argument("--gap-draws", type=int, default=10,
                        help="subsampling draws for the sample-matched RankMe gap "
                             "(default: 10)")
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
                # RankMe / overfit_index are target-independent -> compute once
                # per embedding pair.
                rk_train = rankme(train_emb)
                rk_test = rankme(test_emb)
                ov_index = overfit_index(train_emb, test_emb,
                                         n_draws=args.gap_draws)
                for label, y_full in targets.items():
                    metrics = probe_one(train_emb, test_emb, y_full,
                                        args.standardize)
                    if metrics is None:
                        continue
                    rows.append({
                        "epoch": epoch, "dataset": dataset, "method": "POME",
                        "target": label, "dim": dim, "split": split,
                        "run": run, "rankme_train": rk_train,
                        "rankme_test": rk_test, "overfit_index": ov_index,
                        **metrics,
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

    # Label-free stopping rules vs. the (supervised) AP-peak oracle, per
    # dataset/dim. Written next to the results CSV for the plot to overlay.
    rules_df = stopping_rules_table(
        result_df, args.rankme_plateau_tol, args.gap_tol, args.overfit_tol)
    rules_path = args.output.with_name(
        args.output.stem.replace("_results", "") + "_stopping_rules.csv")
    rules_df.to_csv(rules_path, index=False)
    print(f"\nWrote stopping-rule epochs to {rules_path}")
    print("\n=== Selected stopping epoch per rule (vs. AP-peak oracle) ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(rules_df.to_string(index=False))


if __name__ == "__main__":
    main()
