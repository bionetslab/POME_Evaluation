"""Impute simulated-missingness datasets with POME's graph-based embedder.

For every graph-format ``.tsv`` under ``input_directory`` an
:class:`pome.gnn_embedding.Embedder` is fitted with imputation enabled and the
missing entries (encoded by ``na_encoding``) are filled in via
``impute_all()``. Continuous variables are imputed by POME's regression head (the
only continuous-imputation path in the current implementation): a head trained on
the frozen embeddings predicts the value directly. Categorical variables use the
decoder's nearest-category rule.

Outputs mirror the input file names: ``{name}.tsv`` (imputed matrix) plus a
``{name}.pkl`` sidecar storing the run's hyperparameters and AP score.

Run from the project root with the POME-enabled environment (conda env ``torch``):

    conda run -n torch python src/pome_evaluation/impute_graph_based.py
"""
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd

from pome.gnn_embedding import Embedder, make_deterministic

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POME_IMPUTATION_ROOT = PROJECT_ROOT / "data" / "imputation_data" / "pome_based"

# Per-dataset non-informative NA encoding used to mask simulated missingness.
DATASET_NA_ENCODING = {
    "HANCOCK": -99999.0,
    "TCGA_LUAD": -99.0,
    "MIMIC": -99.0,
}


def impute_graph_based(input_directory: str,
                       output_directory: str,
                       num_epochs: int,
                       na_encoding: float,
                       informative_nas: list,
                       num_dimensions: int,
                       num_bins: int,
                       discretization_type: str,
                       device: str = "cuda"):

    # Iterate over all files in the directory.
    for filename in os.listdir(input_directory):
        if not filename.endswith('.tsv'):
            continue

        file_path = os.path.join(input_directory, filename)
        name_without_ext = os.path.splitext(filename)[0]

        imputed_outfile_path = os.path.join(output_directory,
                                            name_without_ext + '.tsv')
        if os.path.exists(imputed_outfile_path):
            continue

        print("Processing: ", file_path, "...")

        # Load graph-format dataframe (rows = variables, cols = samples + type).
        df = pd.read_csv(file_path, sep='\t', index_col=0)

        # Fit the embedder with imputation enabled. Continuous variables are
        # filled in by POME's regression head (the only path in current POME).
        embedder_params = {
            'na_encoding': na_encoding,
            'informative_nas': informative_nas,
            'device': device,
            'epochs': num_epochs,
            'bins_per_continuous': num_bins,
            'discretization_type': discretization_type,
            'enable_imputation': True,
        }
        embedder = Embedder(**embedder_params,
                            embedding_dimension=num_dimensions)
        embedder.fit(df)
        ap = embedder.return_ap_score()

        # Impute every entry flagged with the non-informative NA encoding.
        imputed_df = embedder.impute_all(na_value=na_encoding)
        imputed_df.to_csv(imputed_outfile_path, sep='\t', index=True)

        # Persist the run's parameters and AP score alongside the imputed file.
        params_record = dict(embedder_params)
        params_record['embedding_dimension'] = num_dimensions
        params_record['ap'] = ap
        params_outfile = os.path.join(output_directory,
                                      f"{name_without_ext}.pkl")
        with open(params_outfile, 'wb') as f:
            pickle.dump(params_record, f)

        del embedder


def main():
    """Run the POME imputation grid, scoped by CLI flags.

    The full grid is ``{z,nonlinear} x {7,11,15} x {16,32,64}`` per dataset.
    To parallelise across Slurm jobs (each capped at 24h), submit one job per
    ``(discretization, bins)`` cell -- 6 jobs cover the whole grid, each doing
    all dims x datasets for its cell:

        for disc in z nonlinear; do for b in 7 11 15; do
            STAGE_OUT="data/imputation_data/pome_based" sbatch --time=24:00:00 \\
                container/submit.sh src/pome_evaluation/impute_graph_based.py \\
                --discretizations $disc --bins $b
        done; done

    Existing imputed ``.tsv`` files are skipped, so jobs are resumable.
    """
    parser = argparse.ArgumentParser(
        description="POME graph-based imputation over a (discretization, bins, "
                    "dim) grid; scope one cell per Slurm job with the flags below.")
    parser.add_argument("--discretizations", nargs="+", default=["z", "nonlinear"],
                        choices=["z", "nonlinear"],
                        help="discretization strategies (default: both)")
    parser.add_argument("--bins", nargs="+", type=int, default=[7, 11, 15],
                        help="bin counts per continuous variable (default: 7 11 15)")
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64],
                        help="POME embedding dimensions (default: 16 32 64)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NA_ENCODING),
                        choices=list(DATASET_NA_ENCODING),
                        help="datasets to impute (default: all with simulated_data)")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="training epochs per fit (default: 2000)")
    args = parser.parse_args()

    informative_na = []
    print(f"Datasets: {args.datasets} | discretizations: {args.discretizations} "
          f"| bins: {args.bins} | dims: {args.dims} | epochs: {args.epochs}")

    for dataset in args.datasets:
        na_encoding = DATASET_NA_ENCODING[dataset]
        input_directory = POME_IMPUTATION_ROOT / dataset / "simulated_data"
        if not input_directory.is_dir():
            print(f"[skip] no simulated_data for {dataset} ({input_directory})")
            continue

        for discretization in args.discretizations:
            for bin_count in args.bins:
                for dim in args.dims:
                    make_deterministic(42)
                    output_directory = (POME_IMPUTATION_ROOT / dataset /
                                        f"imputed_{discretization}_{bin_count}_{dim}")
                    output_directory.mkdir(parents=True, exist_ok=True)

                    impute_graph_based(
                        str(input_directory),
                        str(output_directory),
                        args.epochs,
                        na_encoding,
                        informative_na,
                        num_dimensions=dim,
                        num_bins=bin_count,
                        discretization_type=discretization,
                    )
        print(f"Finished {dataset} imputation!")


if __name__ == "__main__":
    main()
