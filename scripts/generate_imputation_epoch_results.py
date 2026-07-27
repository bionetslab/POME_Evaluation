"""Compute POME imputation accuracy at several *training-epoch snapshots*.

Produces the per-epoch supplement data consumed by
``scripts/plot_supplement_imputation_epochs.ipynb`` (which renders
``scripts/supplement_imputation_per_epoch.pdf``): how POME's imputation quality
evolves with training duration.

For every masked simulated-missingness file of a dataset
(``data/imputation_data/pome_based/{DATASET}/simulated_data/masked_values_{na}_{run}.tsv``)
a single POME :class:`~pome.gnn_embedding.Embedder` is trained **once** for
``max(--epochs)`` epochs. During that run the model weights are snapshotted in
memory at each requested epoch via POME's lightweight ``epoch_callback`` hook
(clone of ``autoencoder.state_dict()`` ~1 MB each; nothing written mid-training).

After the run, each snapshot's weights are loaded back into the model and the
frozen state is used to impute the masked entries exactly as at the end of a
normal fit: categorical variables via the decoder's nearest-category rule and
continuous variables via POME's regression head re-fit on that epoch's
embeddings. The imputed values are scored against the held-out ground truth in
``data/imputation_groundtruth/{DATASET}`` -- raw MAE on the continuous entries
(``mae_cont``) and accuracy on the categorical entries (``acc_cat``).

Reusing one training run for all snapshots (instead of one fit per epoch count)
keeps compute proportional to the number of masked files, not files x epochs. The
graph topology is epoch-independent, so the completed fit's ``_graph_data`` is
reused for every snapshot; only the weight tensors differ between epochs.

One CSV is written per dataset, holding one row per (masked file x snapshot
epoch):

    data/{DATASET}_imputation_per_epoch.csv
        columns: run, na_ratio, epoch, mae_cont, acc_cat, dataset

The script is resumable: a masked file is retrained only if some of its snapshot
rows are missing from the existing CSV (unless ``--overwrite``).

Run from the project root with the POME-enabled environment (conda env ``torch``):

    conda run -n torch python scripts/generate_imputation_epoch_results.py
    conda run -n torch python scripts/generate_imputation_epoch_results.py --dry-run
    conda run -n torch python scripts/generate_imputation_epoch_results.py \
        --datasets hancock --dim 32 --epochs 100 400 700 1000 1500 2000
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
GROUNDTRUTH_ROOT = DATA_ROOT / "imputation_groundtruth"
POME_IMPUTATION_ROOT = DATA_ROOT / "imputation_data" / "pome_based"

# dataset CLI key -> on-disk dataset label (directory / output-file name).
DATASETS = {
    "hancock": "HANCOCK",
    "mimic": "MIMIC",
    "luad": "TCGA_LUAD",
}

# Per-dataset non-informative NA encoding used to mask simulated missingness
# (mirrors src/pome_evaluation/impute_graph_based.py).
DATASET_NA_ENCODING = {
    "HANCOCK": -99999.0,
    "TCGA_LUAD": -99.0,
    "MIMIC": -99.0,
}

DEFAULT_EPOCHS = (100, 400, 700, 1000, 1500, 2000)
DEFAULT_DIM = 32
DEFAULT_BINS = 15
DEFAULT_DISCRETIZATION = "z"
DEFAULT_SEED = 42


# --- Helpers -----------------------------------------------------------------
def parse_masked_name(filename: str) -> tuple[str, str] | None:
    """Return (na_ratio, run) parsed from ``masked_values_{na}_{run}.tsv``."""
    stem = filename[:-len(".tsv")] if filename.endswith(".tsv") else filename
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "masked" or parts[1] != "values":
        return None
    return parts[2], parts[3]


def groundtruth_path(label: str, na_ratio: str, run: str) -> Path:
    """Ground-truth pickle for a masked file (its run is stored as an int)."""
    run_int = str(int(float(run)))
    return GROUNDTRUTH_ROOT / label / f"masked_values_{na_ratio}_{run_int}.pkl"


def score_imputation(imputed_df: pd.DataFrame, gt_dict: dict) -> tuple[float, float]:
    """Score one imputed matrix (variables x samples) against ground truth.

    Returns (mae_cont, acc_cat): raw mean-absolute-error over the held-out
    continuous entries and accuracy over the categorical entries. POME emits
    exact category codes, so categoricals are compared without rounding.
    """
    gt_cont, pred_cont, gt_cat, pred_cat = [], [], [], []
    for (sample, variable), (gt_value, gt_type) in gt_dict.items():
        pred = imputed_df.loc[variable, sample]
        if gt_type == "cont":
            gt_cont.append(gt_value)
            pred_cont.append(pred)
        else:
            gt_cat.append(gt_value)
            pred_cat.append(pred)

    mae_cont = mean_absolute_error(gt_cont, pred_cont) if gt_cont else np.nan
    acc_cat = accuracy_score(gt_cat, pred_cat) if gt_cat else np.nan
    return float(mae_cont), float(acc_cat)


def impute_at_snapshot(embedder, state_dict, na_encoding: float) -> pd.DataFrame:
    """Load one epoch's weights and impute the masked entries from that state.

    Reconstructs the post-training imputation state on the reloaded weights,
    mirroring what ``Embedder.fit`` does at the end of a normal run: refresh the
    node/variable/bin embeddings and the decoder from the frozen encoder, then
    re-fit the continuous regression head on this epoch's embeddings. Returns
    the imputed matrix (rows = variables, columns = samples).
    """
    import torch

    model = embedder.model
    model.load_state_dict(state_dict)
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        node_emb, variable_emb, bin_emb = model.get_embeddings(
            embedder._graph_data.edge_index.to(device))

    embedder._all_embeddings = node_emb.detach().cpu()
    embedder._variable_embeddings = variable_emb.detach().cpu()
    embedder._bin_embeddings = bin_emb.detach().cpu()
    embedder._fitted_decoder = model.get_decoder().to("cpu")

    if embedder.enable_imputation and embedder._cont_vars:
        embedder._fit_value_regressor()

    return embedder.impute_all(na_value=na_encoding)


def process_masked_file(label: str, graph_path: Path, epochs_list: list[int],
                        dim: int, bins: int, discretization: str, seed: int,
                        device: str) -> list[dict]:
    """Train once and score every snapshot epoch for one masked file."""
    from pome.gnn_embedding import Embedder, make_deterministic

    parsed = parse_masked_name(graph_path.name)
    if parsed is None:
        print(f"    [skip] {graph_path.name}: unrecognised name")
        return []
    na_ratio, run = parsed

    gt_path = groundtruth_path(label, na_ratio, run)
    if not gt_path.exists():
        print(f"    [skip] {graph_path.name}: no ground truth {gt_path.name}")
        return []
    with open(gt_path, "rb") as f:
        gt_dict = pickle.load(f)

    na_encoding = DATASET_NA_ENCODING.get(label, -99.0)
    df = pd.read_csv(graph_path, sep="\t", index_col=0)

    # In-memory weight snapshots: {epoch -> cloned CPU state_dict}. Cloning is
    # essential -- a bare state_dict() aliases the live tensors.
    want = set(epochs_list)
    snaps: dict[int, dict] = {}

    def epoch_callback(epoch: int, autoencoder) -> None:
        if epoch in want:
            snaps[epoch] = {k: v.detach().clone().cpu()
                            for k, v in autoencoder.state_dict().items()}

    make_deterministic(seed)
    embedder = Embedder(
        embedding_dimension=dim,
        epochs=max(epochs_list),
        bins_per_continuous=bins,
        discretization_type=discretization,
        na_encoding=na_encoding,
        enable_imputation=True,
        device=device,
        inductive=False,               # fixed epoch axis, no early stopping
        epoch_callback=epoch_callback,  # lightweight in-memory weight snapshots
    )
    embedder.fit(df)

    rows = []
    for epoch in epochs_list:
        if epoch not in snaps:
            print(f"      [warn] no snapshot captured for epoch {epoch}")
            continue
        # Reset the RNG before each snapshot so the regression head's init is
        # deterministic and independent of snapshot order.
        make_deterministic(seed)
        imputed_df = impute_at_snapshot(embedder, snaps[epoch], na_encoding)
        mae_cont, acc_cat = score_imputation(imputed_df, gt_dict)
        rows.append({
            "run": int(float(run)),
            "na_ratio": na_ratio,
            "epoch": epoch,
            "mae_cont": mae_cont,
            "acc_cat": acc_cat,
            "dataset": label,
        })
        print(f"      [ok]   epoch {epoch:04d}  mae_cont={mae_cont:.3f}  "
              f"acc_cat={acc_cat:.3f}")
    return rows


def load_existing(out_path: Path) -> pd.DataFrame:
    """Load the dataset's existing per-epoch CSV, or an empty frame."""
    if out_path.exists():
        return pd.read_csv(out_path)
    return pd.DataFrame(
        columns=["run", "na_ratio", "epoch", "mae_cont", "acc_cat", "dataset"])


def file_done(existing: pd.DataFrame, na_ratio: str, run: str,
              epochs_list: list[int]) -> bool:
    """True if every snapshot epoch for this masked file is already present."""
    if existing.empty:
        return False
    run_int = int(float(run))
    subset = existing[(existing["na_ratio"].astype(str) == str(na_ratio))
                      & (existing["run"].astype(int) == run_int)]
    return set(epochs_list).issubset(set(subset["epoch"].astype(int)))


# --- Driver ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS),
                        default=list(DATASETS))
    parser.add_argument("--epochs", nargs="+", type=int,
                        default=list(DEFAULT_EPOCHS),
                        help=f"snapshot epochs (default: {list(DEFAULT_EPOCHS)})")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"POME embedding dimension (default: {DEFAULT_DIM})")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help=f"POME bins_per_continuous (default: {DEFAULT_BINS})")
    parser.add_argument("--discretization", default=DEFAULT_DISCRETIZATION,
                        help=f"POME discretization (default: {DEFAULT_DISCRETIZATION})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"deterministic seed (default: {DEFAULT_SEED})")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute all masked files even if rows exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the work to be done and exit")
    args = parser.parse_args()

    epochs_list = sorted(set(args.epochs))

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    print(f"Ground-truth root: {GROUNDTRUTH_ROOT}")
    print(f"POME simulated-data root: {POME_IMPUTATION_ROOT}")
    print(f"Datasets: {args.datasets} | dim: {args.dim} | bins: {args.bins} | "
          f"discretization: {args.discretization}")
    print(f"Snapshot epochs: {epochs_list} | device: {device}")

    warnings.filterwarnings("ignore")
    for key in args.datasets:
        label = DATASETS[key]
        sim_dir = POME_IMPUTATION_ROOT / label / "simulated_data"
        if not sim_dir.is_dir():
            print(f"\n[skip] {label}: no simulated_data ({sim_dir})")
            continue

        masked_files = sorted(p for p in sim_dir.glob("*.tsv"))
        out_path = DATA_ROOT / f"{label}_imputation_per_epoch.csv"
        existing = load_existing(out_path)

        print(f"\n[{label}] {len(masked_files)} masked files -> {out_path.name}")
        if args.dry_run:
            todo = 0
            for graph_path in masked_files:
                parsed = parse_masked_name(graph_path.name)
                if parsed is None:
                    continue
                na_ratio, run = parsed
                if args.overwrite or not file_done(existing, na_ratio, run,
                                                   epochs_list):
                    todo += 1
            print(f"  [dry-run] {todo}/{len(masked_files)} files to train "
                  f"({todo * len(epochs_list)} rows); nothing computed.")
            continue

        for graph_path in masked_files:
            parsed = parse_masked_name(graph_path.name)
            if parsed is None:
                continue
            na_ratio, run = parsed
            if not args.overwrite and file_done(existing, na_ratio, run,
                                                epochs_list):
                print(f"  [skip] {graph_path.name} (all epochs present)")
                continue

            print(f"  [train] {graph_path.name}")
            rows = process_masked_file(
                label, graph_path, epochs_list, args.dim, args.bins,
                args.discretization, args.seed, device)
            if not rows:
                continue

            # Merge with existing rows (drop any stale duplicates for this file),
            # then persist immediately so the run stays resumable.
            new_rows = pd.DataFrame(rows)
            run_int = int(float(run))
            keep = ~((existing["na_ratio"].astype(str) == str(na_ratio))
                     & (existing["run"].astype(int) == run_int)
                     & (existing["epoch"].astype(int).isin(epochs_list))) \
                if not existing.empty else pd.Series(dtype=bool)
            existing = pd.concat(
                [existing[keep] if not existing.empty else existing, new_rows],
                ignore_index=True)
            existing.to_csv(out_path, index=False)

        print(f"  [write] {out_path} ({len(existing)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
