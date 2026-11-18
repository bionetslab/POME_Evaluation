"""Model-free reference for the in-distribution check of POME's inductive embeddings (CPU part).

``evaluate_inductive_ood.py`` tests whether POME's inductive embeddings of the held-out test
samples are distributed like the training samples' embeddings. A significant shift there can be
a model effect or a genuine difference between the training and test samples of a split. This
script writes the same train/test comparison for two representations that involve no trained
model, so that ``evaluate_inductive_ood.py`` reports the shift already present in the data:

  raw_onehot     binary sample x value-node indicator matrix: exactly the graph POME is trained
                 on (z-discretization, 15 bins, NA -99; test samples mapped to the training value
                 nodes as in transform()). Squared Euclidean distance = number of differing values.
  spectral       unit-length spectral sample coordinates of the bipartite graph (POME's spectral
                 initialization, d = 32 / 64 components); test samples are placed by the Nystrom
                 extension, which reproduces a training sample's coordinates exactly.

Files follow the layout of ``generate_inductive_ood_embeddings.py`` (``train_transductive``,
``test_inductive``, ``train_inductive``), so the evaluation script runs unchanged; for these
representations ``train_inductive`` equals ``train_transductive`` (the control is trivial).

    container/run.sh scripts/generate_model_free_ood_reference.py
    $WORK/envs/scxmatch/bin/python scripts/evaluate_inductive_ood.py

Outputs: output/inductive_ood/{raw_onehot,spectral}/{dataset}/split_{NN}/dim_{D}/*.csv
         (dim 0 for raw_onehot)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
OUT_ROOT = PROJECT_ROOT / "output" / "inductive_ood"
NA_ENCODING = -99.0


def load_graph(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def write(out_dir: Path, train: np.ndarray, test: np.ndarray, train_names, test_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [f"dim_{i}" for i in range(train.shape[1])]
    tr = pd.DataFrame(train, index=train_names, columns=cols)
    tr.to_csv(out_dir / "train_transductive.csv")
    tr.to_csv(out_dir / "train_inductive.csv")
    pd.DataFrame(test, index=test_names, columns=cols).to_csv(out_dir / "test_inductive.csv")


def run_one(dataset, split, dims, args):
    from pome.gnn_embedding import Embedder
    from pome.utils import spectral_nystrom_projection

    split_dir = SPLITS_ROOT / dataset
    train_df = load_graph(split_dir / f"split_{split:02d}_train_pome.tsv")
    test_df = load_graph(split_dir / f"split_{split:02d}_test_pome.tsv")
    # Only the graph construction is needed (value-node vocabulary, binning); 1 epoch.
    emb = Embedder(embedding_dimension=8, epochs=1, bins_per_continuous=15,
                   discretization_type="z", na_encoding=NA_ENCODING, device="cpu",
                   inductive=False)
    emb.fit(train_df)
    g = emb._graph_data
    ns, nv = g.num_samples, g.num_nodes - g.num_samples

    def value_sets(df):
        data = df.drop(columns=[emb.type_column])
        names = [str(c) for c in data.columns]
        sets = [sorted({v for _, v in emb._sample_value_node_pairs(data, s)})
                for s in data.columns]
        return names, sets

    tr_names, tr_sets = value_sets(train_df)
    te_names, te_sets = value_sets(test_df)
    te_keep = [i for i, s in enumerate(te_sets) if s]  # edgeless samples have no representation

    def onehot(sets):
        X = np.zeros((len(sets), nv))
        for i, s in enumerate(sets):
            X[i, np.asarray(s, dtype=int) - ns] = 1.0
        return X

    base = args.out_root
    write(base / "raw_onehot" / dataset / f"split_{split:02d}" / "dim_0",
          onehot(tr_sets), onehot([te_sets[i] for i in te_keep]),
          tr_names, [te_names[i] for i in te_keep])

    for d in dims:
        P = spectral_nystrom_projection(g.unique_edges, ns, g.num_nodes, d)

        def coords(sets):
            X = np.stack([P[np.asarray(s, dtype=int) - ns].sum(0) for s in sets])
            return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

        write(base / "spectral" / dataset / f"split_{split:02d}" / f"dim_{d}",
              coords(tr_sets), coords([te_sets[i] for i in te_keep]),
              tr_names, [te_names[i] for i in te_keep])
    print(f"[ok] {dataset} split {split:02d}  train {len(tr_sets)} test {len(te_keep)} "
          f"value nodes {nv}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=["hancock", "luad", "mimic"])
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    for dataset in args.datasets:
        for split in args.splits:
            run_one(dataset, split, args.dims, args)


if __name__ == "__main__":
    main()
