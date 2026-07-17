"""Compute inductive POME sample embeddings at several *training-epoch snapshots*
of a single training run, for every dataset and train/test split under
``data/train_test_splits/``.

Purpose: study how the linear-probing quality of POME's inductive embeddings
evolves with training duration. Rather than five independent fits (more compute),
each ``(dataset, split, dim, run)`` is trained **once** for ``--epochs`` (default
2000). During that single run we snapshot the model at every ``--snapshot-every``
epochs (default 400) via POME's lightweight ``epoch_callback`` hook: the callback
clones ``autoencoder.state_dict()`` into memory (~1 MB) at each snapshot epoch.
Nothing is written to disk mid-training -- in particular we do NOT joblib-dump the
whole ``Embedder`` (which would also serialise the negative-edge sets and graph,
~20x larger and pure dead weight for this purpose).

After the run, for each snapshot we load its weights back into the model and
produce the train/test embedding pair exactly as
``scripts/generate_inductive_embeddings.py`` does at the end of training:

    train : the frozen encoder at that epoch, evaluated on the training graph
            (transductive; the sample rows of ``model.get_embeddings()``)
    test  : the unseen test samples embedded inductively via ``transform()``
            (single frozen forward pass, no retraining)

Both are a deterministic function of ``(weights at epoch E, training-graph
topology)``. The graph topology is epoch-independent, so the completed fit's
``_graph_data`` is reused for every snapshot; only the ~1 MB weight tensors differ
between epochs. (Verified bit-identical to extracting the live model at that
epoch.)

Early stopping is intentionally OFF (``inductive=False``): the whole point is to
sweep a *fixed* epoch axis, not to have POME pick one epoch count.

Run from the project root with the POME-enabled environment, e.g.:

    conda run -n torch python scripts/generate_inductive_epoch_snapshots.py
    conda run -n torch python scripts/generate_inductive_epoch_snapshots.py --dry-run
    conda run -n torch python scripts/generate_inductive_epoch_snapshots.py \
        --datasets hancock --dims 32 --epochs 2000 --snapshot-every 400

Outputs -- one ``_train`` + ``_test`` CSV per snapshot, laid out so each
``epoch_{E}`` directory is structurally identical to ``output/inductive`` and can
be probed directly by ``scripts/linear_probe_inductive_embeddings.py``:

    output/inductive_epochs/epoch_{E:04d}/pome/{dataset}/split_{NN}/dim_{D}/run_{R:02d}_{train,test}.csv

Existing outputs are skipped unless ``--overwrite`` is passed (a config is
retrained only if some of its snapshots are missing), so the script is resumable.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "inductive_epochs"

DATASETS = ("hancock", "luad", "mimic")
DIMENSIONS = (16, 32, 64, 128)
N_RUNS = 10

# POME hyperparameters (mirror generate_inductive_embeddings.py so the snapshots
# are comparable to the main inductive embeddings).
NA_ENCODING = -99.0
DEFAULT_EPOCHS = 2000
DEFAULT_SNAPSHOT_EVERY = 400
DEFAULT_BINS = 15
DEFAULT_DISCRETIZATION = "z"
DEFAULT_SEED = 42


# --- Helpers -----------------------------------------------------------------
def discover_splits(dataset: str) -> list[int]:
    """Return sorted split ids that have a POME training file."""
    paths = (SPLITS_ROOT / dataset).glob("split_*_train_pome.tsv")
    ids = []
    for p in paths:
        stem = p.name[len("split_"):-len("_train_pome.tsv")]
        if stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def load_graph(path: Path) -> pd.DataFrame:
    """Load a graph-format split: rows = variables, cols = samples + 'type'."""
    return pd.read_csv(path, sep="\t", index_col=0)


def standardize_columns(emb: pd.DataFrame) -> pd.DataFrame:
    """Rename embedding columns to dim_0 .. dim_{k-1}, preserving the index."""
    emb = emb.copy()
    emb.columns = [f"dim_{i}" for i in range(emb.shape[1])]
    return emb


def snapshot_epochs(epochs: int, every: int) -> list[int]:
    """Epoch counts at which snapshots are taken: every, 2*every, ..., <= epochs."""
    return list(range(every, epochs + 1, every))


def run_paths(out_root: Path, epoch: int, dataset: str, split_id: int,
              dim: int, run: int) -> tuple[Path, Path]:
    """(train_csv, test_csv) for one snapshot, mirroring output/inductive layout."""
    d = (out_root / f"epoch_{epoch:04d}" / "pome" / dataset
         / f"split_{split_id:02d}" / f"dim_{dim}")
    return d / f"run_{run:02d}_train.csv", d / f"run_{run:02d}_test.csv"


def config_done(out_root: Path, epochs_list, dataset, split_id, dim, run) -> bool:
    """True if every snapshot's train+test CSV already exists for this config."""
    return all(tr.exists() and te.exists()
               for e in epochs_list
               for tr, te in [run_paths(out_root, e, dataset, split_id, dim, run)])


# --- Snapshot extraction -----------------------------------------------------
def extract_from_weights(embedder, state_dict, test_df, dim: int):
    """Load one epoch's weights into the model and return (train_emb, test_emb).

    Reuses the completed fit's model object (which carries the epoch-independent
    ``node_to_embeddings`` / graph attributes) and just swaps in the snapshot
    weights. Train embeddings are the sample rows of the frozen encoder's
    eval-mode forward over the training graph (matching ``get_embeddings()``);
    test embeddings come from the inductive ``transform()`` (also eval mode).
    """
    import torch

    embedder.model.load_state_dict(state_dict)
    embedder.model.eval()

    # `_graph_data` is left on CPU by fit(), but the model may be on GPU; align the
    # edge_index to the model's device before the forward pass. (transform() does
    # this alignment internally, so the test side needs no such handling.)
    device = next(embedder.model.parameters()).device
    with torch.no_grad():
        node_emb, _, _ = embedder.model.get_embeddings(
            embedder._graph_data.edge_index.to(device))
    sample_rows = list(embedder._sample_node_dict.values())
    train_arr = node_emb[sample_rows].detach().cpu().numpy()
    train_emb = pd.DataFrame(
        train_arr, index=list(embedder._X.columns),
        columns=[f"dim_{i}" for i in range(dim)])

    test_emb = embedder.transform(test_df)  # inductive; eval/no_grad internally
    return train_emb, test_emb


def process_config(dataset: str, split_id: int, dim: int, run: int,
                   epochs: int, every: int, bins: int, discretization: str,
                   seed_base: int, device: str, out_root: Path,
                   overwrite: bool) -> None:
    """Train one (dataset, split, dim, run) once and write all epoch snapshots."""
    from pome.gnn_embedding import Embedder, make_deterministic

    epochs_list = snapshot_epochs(epochs, every)
    if not overwrite and config_done(out_root, epochs_list, dataset, split_id,
                                      dim, run):
        print(f"    [skip] {dataset} split {split_id:02d} dim {dim} "
              f"run {run:02d} (all {len(epochs_list)} snapshots present)")
        return

    split_dir = SPLITS_ROOT / dataset
    train_df = load_graph(split_dir / f"split_{split_id:02d}_train_pome.tsv")
    test_df = load_graph(split_dir / f"split_{split_id:02d}_test_pome.tsv")

    # In-memory snapshot store: {epoch -> cloned CPU state_dict (~1 MB each)}.
    # Cloning is essential -- a bare state_dict() aliases the live tensors, so
    # every snapshot would otherwise end up holding the final-epoch weights.
    want = set(epochs_list)
    snaps: dict[int, dict] = {}

    def epoch_callback(epoch: int, autoencoder) -> None:
        if epoch in want:
            snaps[epoch] = {k: v.detach().clone().cpu()
                            for k, v in autoencoder.state_dict().items()}

    make_deterministic(seed_base + run)
    embedder = Embedder(
        embedding_dimension=dim,
        epochs=epochs,
        bins_per_continuous=bins,
        discretization_type=discretization,
        na_encoding=NA_ENCODING,
        device=device,
        inductive=False,               # fixed epoch axis, no early stopping
        epoch_callback=epoch_callback,  # lightweight in-memory weight snapshots
    )
    embedder.fit(train_df)

    for e in epochs_list:
        train_out, test_out = run_paths(
            out_root, e, dataset, split_id, dim, run)
        if not overwrite and train_out.exists() and test_out.exists():
            print(f"      [skip] epoch {e:04d}")
            continue
        if e not in snaps:
            print(f"      [warn] no snapshot captured for epoch {e}")
            continue
        train_emb, test_emb = extract_from_weights(
            embedder, snaps[e], test_df, dim)
        train_out.parent.mkdir(parents=True, exist_ok=True)
        standardize_columns(train_emb).to_csv(train_out)
        standardize_columns(test_emb).to_csv(test_out)
        print(f"      [ok]   epoch {e:04d}  train {train_emb.shape} "
              f"test {test_emb.shape}")


# --- Driver ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--dims", nargs="+", type=int, default=list(DIMENSIONS))
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--splits", nargs="+", type=int, default=None,
                        help="split ids to process (default: all discovered)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="total POME training epochs (default: 2000)")
    parser.add_argument("--snapshot-every", type=int,
                        default=DEFAULT_SNAPSHOT_EVERY,
                        help="snapshot the embeddings every N epochs (default: 400)")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help="POME bins_per_continuous (default: 15)")
    parser.add_argument("--discretization", default=DEFAULT_DISCRETIZATION,
                        help="POME discretization_type (default: z)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="base seed; run r uses seed+r (default: 42)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute and overwrite existing snapshot files")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the work to be done and exit")
    args = parser.parse_args()

    if args.epochs % args.snapshot_every != 0:
        print(f"[warn] --epochs ({args.epochs}) is not a multiple of "
              f"--snapshot-every ({args.snapshot_every}); the last snapshot will "
              f"be at {snapshot_epochs(args.epochs, args.snapshot_every)[-1]}.")
    epochs_list = snapshot_epochs(args.epochs, args.snapshot_every)

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    print(f"Splits root: {SPLITS_ROOT}")
    print(f"Output root: {args.output_dir}")
    print(f"Datasets: {args.datasets} | dims: {args.dims} | runs: {args.runs}")
    print(f"POME: epochs={args.epochs} snapshot_every={args.snapshot_every} "
          f"-> snapshots at {epochs_list}")
    print(f"      bins={args.bins} discretization={args.discretization} "
          f"device={device}")

    plan = []  # (dataset, [split_ids])
    for dataset in args.datasets:
        found = discover_splits(dataset)
        ids = found if args.splits is None else [s for s in args.splits
                                                 if s in found]
        if not ids:
            print(f"  [warn] no POME splits for {dataset}")
            continue
        plan.append((dataset, ids))
        n_cfg = len(ids) * len(args.dims) * args.runs
        print(f"  {dataset:8s}: splits {ids} -> {n_cfg} training runs, "
              f"{n_cfg * len(epochs_list)} snapshots "
              f"({n_cfg * len(epochs_list) * 2} files)")

    if args.dry_run:
        total_cfg = sum(len(ids) * len(args.dims) * args.runs
                        for _, ids in plan)
        print(f"\n[dry-run] {total_cfg} training runs, "
              f"{total_cfg * len(epochs_list)} snapshots; nothing computed.")
        return

    warnings.filterwarnings("ignore")
    for dataset, ids in plan:
        for split_id in ids:
            for dim in args.dims:
                for run in range(args.runs):
                    print(f"\n[{dataset} | split {split_id:02d} | dim {dim} | "
                          f"run {run:02d}]")
                    process_config(
                        dataset, split_id, dim, run, args.epochs,
                        args.snapshot_every, args.bins, args.discretization,
                        args.seed, device, args.output_dir, args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    main()
