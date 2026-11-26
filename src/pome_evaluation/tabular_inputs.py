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


# --- inductive (train/test split) inputs, for the linear-probing analysis ----------------
SPLITS_ROOT = os.path.join(PROJECT_ROOT, "data", "train_test_splits")
INDUCTIVE_COMPETITOR_ROOT = os.path.join(PROJECT_ROOT, "output", "inductive_competitors")
SPLIT_DATASETS = ("hancock", "luad", "mimic", "luad_noleak")   # directory names under SPLITS_ROOT


def _graph_to_table(path):
    graph = pd.read_csv(path, sep="\t", index_col=0)
    types = graph.pop("type")
    table = graph.T.astype(float).replace(NA_ENCODING, np.nan)
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    return table, [str(v) for v in types.index[types == "cat"]]


def split_ids(dataset):
    """Sorted ids of the train/test splits of ``dataset`` (``hancock`` / ``luad`` / ``mimic``)."""
    names = os.listdir(os.path.join(SPLITS_ROOT, dataset))
    return sorted(int(n.split("_")[1]) for n in names if n.endswith("_train_pome.tsv"))


def load_split_tables(dataset, split):
    """Return (train table, test table, categorical columns) of one split.

    Read from the POME-format split files that POME's own inductive embeddings are built
    from (``data/train_test_splits/<dataset>/split_NN_{train,test}_pome.tsv``), so the
    competitors see exactly the same samples and values; NaN = missing, test columns in
    training order.
    """
    base = os.path.join(SPLITS_ROOT, dataset, "split_%02d_%%s_pome.tsv" % split)
    train, cat_cols = _graph_to_table(base % "train")
    test, _ = _graph_to_table(base % "test")
    return train, test[train.columns], cat_cols


def inductive_paths(method, dataset, split, dim, run, root=INDUCTIVE_COMPETITOR_ROOT):
    """(train csv, test csv) in the layout of ``generate_inductive_embeddings.run_paths``.

    ``<root>/<method>/<dataset>/split_NN/dim_D/run_RR_{train,test}.csv``, which
    ``linear_probe_inductive_embeddings.py --emb-root <root>`` discovers.
    """
    d = os.path.join(root, method, dataset, "split_%02d" % split, "dim_%d" % dim)
    return (os.path.join(d, "run_%02d_train.csv" % run),
            os.path.join(d, "run_%02d_test.csv" % run))


def write_embedding_csv(Z, index, path):
    """Write a samples x dim embedding (columns dim_0..) atomically."""
    if not np.isfinite(Z).all():
        raise RuntimeError("non-finite embedding for %s" % path)
    df = pd.DataFrame(Z, index=index, columns=["dim_%d" % i for i in range(Z.shape[1])])
    os.makedirs(os.path.dirname(path), exist_ok=True)       # parallel workers share dirs
    df.to_csv(path + ".tmp")
    os.replace(path + ".tmp", path)
