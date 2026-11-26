"""
Generate 10 random 80/20 train/test splits of the Hancock dataset.

Both graph format (vars x samples, hancock_wo_targets_graph.tsv) and sample format
(samples x vars, hancock_wo_targets.csv) are saved for each split.

Output layout:
  data/splits/
    split_00_train_graph.tsv   split_00_test_graph.tsv
    split_00_train_samples.csv split_00_test_samples.csv
    split_01_...
    ...
"""

import argparse
import os
import numpy as np
import pandas as pd

TYPE_COL = "type"
N_SPLITS = 10
TRAIN_FRAC = 0.8
SEED = 42


def make_splits(graph_df, sample_df, n_splits=N_SPLITS, train_frac=TRAIN_FRAC, seed=SEED):
    """Yield (i, train graph, test graph, train samples, test samples) of the random splits.

    graph_df: variables x samples (+ ``type`` column); sample_df: samples x variables.
    """
    all_samples = np.array([c for c in graph_df.columns if c != TYPE_COL])
    n_train = round(len(all_samples) * train_frac)
    rng = np.random.default_rng(seed)
    for i in range(n_splits):
        perm = rng.permutation(len(all_samples))
        train = all_samples[perm[:n_train]].tolist()
        test = all_samples[perm[n_train:]].tolist()
        yield (i, graph_df[train + [TYPE_COL]], graph_df[test + [TYPE_COL]],
               sample_df.loc[train], sample_df.loc[test])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", default="data/hancock_wo_targets_graph.tsv")
    parser.add_argument("--samples", default="data/hancock_wo_targets.csv")
    parser.add_argument("--outdir", default="data/splits/hancock")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    graph_df = pd.read_csv(args.graph, sep="\t", index_col=0)
    sample_df = pd.read_csv(args.samples, index_col=0)
    sample_df.index = sample_df.index.astype(str)

    all_samples = np.array([c for c in graph_df.columns if c != TYPE_COL])
    n_train = round(len(all_samples) * args.train_frac)

    print(f"Dataset: {len(all_samples)} samples, {len(graph_df.index)} variables")
    print(f"Generating {args.n_splits} splits  "
          f"(train={n_train}, test={len(all_samples) - n_train})")
    print(f"Output directory: {args.outdir}\n")

    for i, g_train, g_test, s_train, s_test in make_splits(
            graph_df, sample_df, args.n_splits, args.train_frac, args.seed):
        prefix = os.path.join(args.outdir, f"split_{i:02d}")
        g_train.to_csv(f"{prefix}_train_pome.tsv", sep="\t")
        g_test.to_csv(f"{prefix}_test_pome.tsv", sep="\t")
        s_train.to_csv(f"{prefix}_train_umap.csv")
        s_test.to_csv(f"{prefix}_test_umap.csv")
        print(f"Split {i:02d}: train={len(s_train)}  test={len(s_test)}")

    print(f"\nDone. Files written to {args.outdir}/")


if __name__ == "__main__":
    main()
