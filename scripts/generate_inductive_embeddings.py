"""Compute inductive (train-fit, test-transform) embeddings for every dataset
and train/test split under ``data/train_test_splits/``.

For each split we produce embeddings with two methods:

POME  (``*_train_pome.tsv`` / ``*_test_pome.tsv``, graph format)
    Fit an ``Embedder`` on the training split, read off the transductive
    embeddings for the training samples via ``get_embeddings()``, and embed the
    unseen test samples inductively via ``transform()`` (single forward pass of
    the frozen trained encoder, no retraining).

UMAP  (``*_train_umap.csv`` / ``*_test_umap.csv``, sample format)
    Fit UMAP on the training split and apply it to the unseen test samples.
    Mirrors ``src/pome_evaluation/embed_UMAP_combined.py`` but inductively:

    - HANCOCK / TCGA-LUAD: a numeric UMAP (euclidean, on RobustScaler-scaled
      continuous features) and a categorical UMAP (hamming) are fit *separately*,
      each at the full ``dim`` components, then combined by intersecting their
      fuzzy graphs (``numeric_mapper * cat_mapper``) -- exactly as the original
      transductive pipeline. The train embedding is the intersected model's
      ``embedding_``; the test embedding is produced by ``umap.transform_combined``,
      which rebuilds that intersection for unseen samples (per-modality bipartite
      graphs -> graph intersection -> optimise against the joint embedding). Train
      and test therefore share the same ``dim``-column intersected space.
    - MIMIC IV: numeric-only (its UMAP data has no categorical columns), so a
      single numeric mapper is used at full ``dim``; the test set is embedded with
      the mapper's ordinary inductive ``transform()``. => output has ``dim`` columns.

For each split and method, embeddings are computed for dimensions 16, 32, 64,
and for each dimension over 10 independent runs (distinct random seeds).

Run from the project root with the POME-enabled environment, e.g.:

    conda run -n torch python scripts/generate_inductive_embeddings.py
    conda run -n torch python scripts/generate_inductive_embeddings.py --dry-run
    conda run -n torch python scripts/generate_inductive_embeddings.py \
        --datasets hancock --methods umap --epochs 200

Outputs (each run writes a ``_train`` and a ``_test`` CSV, sample-indexed):

    output/inductive_embeddings/{method}/{dataset}/split_{NN}/dim_{D}/run_{R}_{train,test}.csv

Existing outputs are skipped unless ``--overwrite`` is passed, so the script is
resumable.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
INPUT_DATASETS = PROJECT_ROOT / "data" / "input_datasets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "inductive_embeddings"

DATASETS = ("hancock", "luad", "mimic")
METHODS = ("pome", "umap")
DIMENSIONS = (16, 32, 64)
N_RUNS = 10

# Per-dataset categorical-variable lists (column `cat_var`), used to split the
# UMAP sample matrices into numeric vs categorical feature blocks.
CAT_VAR_FILES = {
    "hancock": "hancock_cat_variables.csv",
    "luad": "TCGA_LUAD_cat_vars.csv",
    "mimic": "mimic_aggregated_cat_variables.csv",
}
# Datasets whose categorical UMAP is deactivated (numeric-only), per the
# original transductive pipeline.
NUMERIC_ONLY_DATASETS = {"mimic"}

# POME hyperparameters (mirrors evaluate_inductive_ood.py in the POME repo).
NA_ENCODING = -99.0
DEFAULT_EPOCHS = 1000
DEFAULT_BINS = 15
DEFAULT_DISCRETIZATION = "z"
DEFAULT_SEED = 42

_SPLIT_RE = re.compile(r"split_(\d+)_train_(pome|umap)\.(tsv|csv)$")


# --- Helpers -----------------------------------------------------------------
def discover_splits(dataset: str, method: str) -> list[int]:
    """Return sorted split ids that have a training file for ``method``."""
    ext = "tsv" if method == "pome" else "csv"
    paths = (SPLITS_ROOT / dataset).glob(f"split_*_train_{method}.{ext}")
    ids = []
    for p in paths:
        m = _SPLIT_RE.search(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def standardize_columns(emb: pd.DataFrame) -> pd.DataFrame:
    """Rename embedding columns to dim_0 .. dim_{k-1}, preserving the index."""
    emb = emb.copy()
    emb.columns = [f"dim_{i}" for i in range(emb.shape[1])]
    return emb


def run_paths(out_root: Path, method: str, dataset: str,
              split_id: int, dim: int, run: int) -> tuple[Path, Path]:
    """Return (train_csv, test_csv) output paths for one run."""
    d = out_root / method / dataset / f"split_{split_id:02d}" / f"dim_{dim}"
    return d / f"run_{run:02d}_train.csv", d / f"run_{run:02d}_test.csv"


# --- POME --------------------------------------------------------------------
def load_graph(path: Path) -> pd.DataFrame:
    """Load a graph-format split: rows = variables, cols = samples + 'type'."""
    return pd.read_csv(path, sep="\t", index_col=0)


def embed_pome_split(dataset: str, split_id: int, dims, n_runs: int,
                     epochs: int, bins: int, discretization: str,
                     seed_base: int, device: str, out_root: Path,
                     overwrite: bool) -> None:
    from pome.gnn_embedding import Embedder, make_deterministic

    split_dir = SPLITS_ROOT / dataset
    train_df = load_graph(split_dir / f"split_{split_id:02d}_train_pome.tsv")
    test_df = load_graph(split_dir / f"split_{split_id:02d}_test_pome.tsv")

    for dim in dims:
        for run in range(n_runs):
            train_out, test_out = run_paths(
                out_root, "pome", dataset, split_id, dim, run)
            if not overwrite and train_out.exists() and test_out.exists():
                print(f"    [skip] pome {dataset} split {split_id:02d} "
                      f"dim {dim} run {run:02d}")
                continue

            make_deterministic(seed_base + run)
            embedder = Embedder(
                embedding_dimension=dim,
                epochs=epochs,
                bins_per_continuous=bins,
                discretization_type=discretization,
                na_encoding=NA_ENCODING,
                device=device,
            )
            embedder.fit(train_df)
            train_emb, *_ = embedder.get_embeddings()  # transductive (train)
            test_emb = embedder.transform(test_df)     # inductive (test)

            train_out.parent.mkdir(parents=True, exist_ok=True)
            standardize_columns(train_emb).to_csv(train_out)
            standardize_columns(test_emb).to_csv(test_out)
            print(f"    [ok]   pome {dataset} split {split_id:02d} dim {dim} "
                  f"run {run:02d}  train {train_emb.shape} test {test_emb.shape}")


# --- UMAP --------------------------------------------------------------------
def split_numeric_categorical(df: pd.DataFrame, dataset: str):
    """Split a UMAP sample matrix into (continuous, categorical) column blocks.

    For numeric-only datasets the categorical block is empty regardless of the
    cat-var file.
    """
    if dataset in NUMERIC_ONLY_DATASETS:
        cat_cols = []
    else:
        cat_path = INPUT_DATASETS / CAT_VAR_FILES[dataset]
        cat_vars = set(pd.read_csv(cat_path)["cat_var"])
        cat_cols = [c for c in df.columns if c in cat_vars]
    cont_cols = [c for c in df.columns if c not in set(cat_cols)]
    return cont_cols, cat_cols


def embed_umap_split(dataset: str, split_id: int, dims, n_runs: int,
                     out_root: Path, overwrite: bool) -> None:
    import umap

    split_dir = SPLITS_ROOT / dataset
    train_df = pd.read_csv(
        split_dir / f"split_{split_id:02d}_train_umap.csv", index_col=0)
    test_df = pd.read_csv(
        split_dir / f"split_{split_id:02d}_test_umap.csv", index_col=0)
    # Align test columns to the training feature order.
    test_df = test_df[train_df.columns]

    cont_cols, cat_cols = split_numeric_categorical(train_df, dataset)
    use_cat = len(cat_cols) > 0

    # RobustScaler is fit on the training continuous features only, then applied
    # to both train and test (proper inductive scaling).
    scaler = RobustScaler().fit(train_df[cont_cols].to_numpy())
    train_num = scaler.transform(train_df[cont_cols].to_numpy())
    test_num = scaler.transform(test_df[cont_cols].to_numpy())
    if use_cat:
        train_cat = train_df[cat_cols].to_numpy()
        test_cat = test_df[cat_cols].to_numpy()

    for dim in dims:
        # Both modalities are fit at the full `dim`; the fuzzy-graph intersection
        # produces a single `dim`-column joint embedding (np.min of the two
        # n_components), so no dimensionality splitting is needed.
        for run in range(n_runs):
            train_out, test_out = run_paths(
                out_root, "umap", dataset, split_id, dim, run)
            if not overwrite and train_out.exists() and test_out.exists():
                print(f"    [skip] umap {dataset} split {split_id:02d} "
                      f"dim {dim} run {run:02d}")
                continue

            # Numeric UMAP (euclidean) at the full target dimensionality.
            num_mapper = umap.UMAP(n_components=dim, random_state=run).fit(
                train_num.copy(), ensure_all_finite="allow-nan")

            if use_cat:
                # Categorical UMAP (hamming), also at full `dim`, then combine the
                # two fuzzy graphs by intersection (matching the transductive
                # pipeline). transform_combined() reproduces that intersection for
                # unseen test samples, so train and test share the same joint space.
                cat_mapper = umap.UMAP(
                    n_components=dim, metric="hamming", random_state=run).fit(
                    train_cat.copy(), ensure_all_finite="allow-nan")
                combined = num_mapper * cat_mapper
                train_arr = combined.embedding_
                test_arr = umap.transform_combined(
                    [num_mapper, cat_mapper], combined,
                    [test_num.copy(), test_cat.copy()],
                    ensure_all_finite="allow-nan")
            else:
                # Numeric-only dataset: single mapper, plain inductive transform.
                train_arr = num_mapper.embedding_
                test_arr = num_mapper.transform(
                    test_num.copy(), ensure_all_finite="allow-nan")

            train_emb = pd.DataFrame(train_arr, index=train_df.index)
            test_emb = pd.DataFrame(test_arr, index=test_df.index)

            train_out.parent.mkdir(parents=True, exist_ok=True)
            standardize_columns(train_emb).to_csv(train_out)
            standardize_columns(test_emb).to_csv(test_out)
            print(f"    [ok]   umap {dataset} split {split_id:02d} dim {dim} "
                  f"run {run:02d}  train {train_emb.shape} test {test_emb.shape}")


# --- Driver ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--dims", nargs="+", type=int, default=list(DIMENSIONS))
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--splits", nargs="+", type=int, default=None,
                        help="split ids to process (default: all discovered)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="POME training epochs (default: 1000)")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help="POME bins_per_continuous (default: 15)")
    parser.add_argument("--discretization", default=DEFAULT_DISCRETIZATION,
                        help="POME discretization_type (default: z)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="base seed; run r uses seed+r (default: 42)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute and overwrite existing embedding files")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the work to be done and exit")
    args = parser.parse_args()

    device = "cpu"
    if "pome" in args.methods:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

    print(f"Splits root: {SPLITS_ROOT}")
    print(f"Output root: {args.output_dir}")
    print(f"Datasets: {args.datasets} | methods: {args.methods} | "
          f"dims: {args.dims} | runs: {args.runs}")
    if "pome" in args.methods:
        print(f"POME: epochs={args.epochs} bins={args.bins} "
              f"discretization={args.discretization} device={device}")

    plan = []  # (method, dataset, [split_ids])
    for method in args.methods:
        for dataset in args.datasets:
            found = discover_splits(dataset, method)
            ids = found if args.splits is None else [s for s in args.splits
                                                     if s in found]
            if not ids:
                print(f"  [warn] no {method} splits for {dataset}")
                continue
            plan.append((method, dataset, ids))
            n = len(ids) * len(args.dims) * args.runs
            print(f"  {method:4s} {dataset:8s}: splits {ids} -> {n} runs "
                  f"({n * 2} files)")

    if args.dry_run:
        total = sum(len(ids) * len(args.dims) * args.runs
                    for _, _, ids in plan)
        print(f"\n[dry-run] {total} runs total; nothing computed.")
        return

    warnings.filterwarnings("ignore")
    for method, dataset, ids in plan:
        for split_id in ids:
            print(f"\n[{method} | {dataset} | split {split_id:02d}]")
            if method == "pome":
                embed_pome_split(
                    dataset, split_id, args.dims, args.runs, args.epochs,
                    args.bins, args.discretization, args.seed, device,
                    args.output_dir, args.overwrite)
            else:
                embed_umap_split(
                    dataset, split_id, args.dims, args.runs,
                    args.output_dir, args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    main()
