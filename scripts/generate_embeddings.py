"""Generate transductive POME embeddings for the HANCOCK, MIMIC and LUAD datasets.

For each dataset the graph-format input (rows = variables, columns = samples plus a
final ``type`` column) is fed to a :class:`pome.gnn_embedding.Embedder`. Every
(dimension, run) combination is trained from scratch with a deterministic seed and
the resulting sample / variable / bin embeddings are written next to the previously
generated embeddings, using the ``{LABEL}_{kind}_{dim}_{run}.tsv`` naming convention.

Run from the project root with the POME-enabled environment (conda env ``torch``):

    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --datasets hancock luad --dims 16 32
"""

import argparse
import os
from pathlib import Path

import pandas as pd

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATASETS = PROJECT_ROOT / "data" / "input_datasets"
EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "embeddings"

# POME embedder hyperparameters (mirror the reference transductive pipeline).
NA_ENCODING = -99.0
INFORMATIVE_NAS = []
DEFAULT_EPOCHS = 2000
DEFAULT_DIMS = (16,)
DEFAULT_RUNS = 10

# dataset key -> (input graph file, output label, output directory name)
DATASETS = {
    "hancock": ("hancock_wo_targets_graph.tsv", "HANCOCK", "HANCOCK"),
    "mimic": ("mimic_aggregated_wo_targets.tsv", "MIMIC", "MIMIC"),
    "luad": ("TCGA_LUAD_wo_targets_graph.tsv", "TCGA_LUAD", "TCGA_LUAD"),
}


def resolve_device(requested: str) -> str:
    """Return a usable torch device, falling back to CPU when CUDA is absent."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    print("  [warn] CUDA unavailable, falling back to CPU")
    return "cpu"


def embed_dataset(key: str, dims, n_runs: int, epochs: int, device: str,
                  overwrite: bool) -> None:
    from pome.gnn_embedding import Embedder, make_deterministic

    file_name, label, out_name = DATASETS[key]
    in_path = INPUT_DATASETS / file_name
    out_dir = EMBEDDINGS_ROOT / out_name / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, index_col=0, sep="\t")
    print(f"[{label}] loaded {in_path.name}  (variables={df.shape[0]}, "
          f"columns={df.shape[1]})")

    for dim in dims:
        for run in range(n_runs):
            paths = {
                kind: out_dir / f"{label}_{kind}_{dim}_{run}.tsv"
                for kind in ("samples", "variables", "bins")
            }
            if not overwrite and all(p.exists() for p in paths.values()):
                print(f"  [skip] {label} dim {dim} run {run}")
                continue

            print(f"  [run]  {label} dim {dim} run {run}")
            make_deterministic(run)
            embedder = Embedder(
                embedding_dimension=dim,
                epochs=epochs,
                na_encoding=NA_ENCODING,
                informative_nas=INFORMATIVE_NAS,
                device=device,
            )
            embedder.fit(df.copy())

            sample_df, var_df, bin_df, _ = embedder.get_embeddings()
            sample_df.to_csv(paths["samples"], sep="\t", index=True)
            var_df.to_csv(paths["variables"], sep="\t", index=True)
            bin_df.to_csv(paths["bins"], sep="\t", index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS),
                        default=list(DATASETS))
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS),
                        help=f"embedding dimensions (default: {list(DEFAULT_DIMS)})")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"runs per dimension, seeded 0..runs-1 "
                             f"(default: {DEFAULT_RUNS})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="compute device (default: cuda, falls back to cpu)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute and overwrite existing embedding files")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Input datasets: {INPUT_DATASETS}")
    print(f"Embeddings root: {EMBEDDINGS_ROOT}")
    print(f"Datasets: {args.datasets} | dims: {args.dims} | runs: {args.runs} | "
          f"epochs: {args.epochs} | device: {device}")

    for key in args.datasets:
        embed_dataset(key, args.dims, args.runs, args.epochs, device,
                      args.overwrite)


if __name__ == "__main__":
    main()
