"""Impute simulated-missingness datasets with POME's graph-based embedder.

For every graph-format ``.tsv`` under ``input_directory`` an
:class:`pome.gnn_embedding.Embedder` is fitted with imputation enabled and the
missing entries (encoded by ``na_encoding``) are filled in via
``impute_all()``. Continuous variables are imputed with
``numeric_imputation="regression"`` (a regression head trained on the frozen
embeddings predicts the value directly, instead of falling back to the selected
bin's mean); categorical variables use the decoder's nearest-category rule.

Outputs mirror the input file names: ``{name}.tsv`` (imputed matrix) plus a
``{name}.pkl`` sidecar storing the run's hyperparameters and AP score.

Run from the project root with the POME-enabled environment (conda env ``torch``):

    conda run -n torch python src/pome_evaluation/impute_graph_based.py
"""
import os
import pickle
from pathlib import Path

import pandas as pd

from pome.gnn_embedding import Embedder, make_deterministic

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POME_IMPUTATION_ROOT = PROJECT_ROOT / "data" / "imputation_data" / "pome_based"

# Continuous imputation strategy exposed by the current POME implementation.
# "regression" trains a regression head on the frozen embeddings; the
# alternative is "bin_mean".
NUMERIC_IMPUTATION = "regression"

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
                       numeric_imputation: str = NUMERIC_IMPUTATION,
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

        # Fit the embedder with imputation enabled. numeric_imputation selects
        # how continuous variables are filled in (here: "regression").
        embedder_params = {
            'na_encoding': na_encoding,
            'informative_nas': informative_nas,
            'device': device,
            'epochs': num_epochs,
            'bins_per_continuous': num_bins,
            'discretization_type': discretization_type,
            'enable_imputation': True,
            'numeric_imputation': numeric_imputation,
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


if __name__ == "__main__":

    dims = [16, 32, 64]
    bins = [15]
    discretizations = ["z"]
    number_of_epochs = 2000
    informative_na = []

    for dataset, na_encoding in DATASET_NA_ENCODING.items():
        input_directory = POME_IMPUTATION_ROOT / dataset / "simulated_data"
        if not input_directory.is_dir():
            print(f"[skip] no simulated_data for {dataset} ({input_directory})")
            continue

        for discretization in discretizations:
            for bin_count in bins:
                for dim in dims:
                    make_deterministic(42)
                    output_directory = (POME_IMPUTATION_ROOT / dataset /
                                        f"imputed_{discretization}_{bin_count}_{dim}_regression")
                    output_directory.mkdir(parents=True, exist_ok=True)

                    impute_graph_based(
                        str(input_directory),
                        str(output_directory),
                        number_of_epochs,
                        na_encoding,
                        informative_na,
                        num_dimensions=dim,
                        num_bins=bin_count,
                        discretization_type=discretization,
                    )
        print(f"Finished {dataset} imputation!")
