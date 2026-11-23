"""Samples x variables view of the POME input graphs, for the tabular embedding competitors.

The competitors of the unsupervised clustering analysis (HI-VAE, SAINT, missForest +
autoencoder; see ``scripts/embed_hivae.py``, ``scripts/embed_saint.py``,
``scripts/impute_full_missforest.py`` and ``scripts/embed_autoencoder.py``) read exactly the data POME is trained on: the graph-format
inputs of ``scripts/generate_embeddings.py`` (rows = variables, columns = samples plus a
final ``type`` column, ``-99`` = missing). The variable types come from that ``type``
column (``cat`` -> categorical, everything else -> continuous); for HANCOCK and TCGA_LUAD
they coincide with the ``*_cat_variables`` lists, MIMIC is all-continuous.

Kept Python-3.7 compatible: it is also imported inside the HI-VAE image (TF 1.15).
"""

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DATASETS = os.path.join(PROJECT_ROOT, "data", "input_datasets")
COMPETITOR_ROOT = os.path.join(PROJECT_ROOT, "data", "embeddings_competitors")
NA_ENCODING = -99.0

# output label -> graph file (same as generate_embeddings.DATASETS)
GRAPH_FILES = {
    "HANCOCK": "hancock_wo_targets_graph.tsv",
    "MIMIC": "mimic_aggregated_wo_targets.tsv",
    "TCGA_LUAD": "TCGA_LUAD_wo_targets_graph.tsv",
}
DATASETS = list(GRAPH_FILES)


def load_table(dataset):
    """Return (samples x variables float DataFrame with NaN = missing, categorical columns)."""
    graph = pd.read_csv(os.path.join(INPUT_DATASETS, GRAPH_FILES[dataset]), sep="\t",
                        index_col=0)
    types = graph.pop("type")
    table = graph.T.astype(float).replace(NA_ENCODING, np.nan)
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    cat_cols = [str(v) for v in types.index[types == "cat"]]
    return table, cat_cols


def embedding_path(dataset, method, dim, run, root=COMPETITOR_ROOT):
    """``<root>/<DATASET>/embeddings/<DATASET>_<METHOD>_<dim>_<run>.tsv``."""
    return os.path.join(root, dataset, "embeddings",
                        "%s_%s_%d_%d.tsv" % (dataset, method, dim, run))


def imputed_path(dataset, root=COMPETITOR_ROOT):
    """missForest-imputed full table (``scripts/impute_full_missforest.py``)."""
    return os.path.join(root, dataset, "imputed", "%s_missforest.tsv" % dataset)
