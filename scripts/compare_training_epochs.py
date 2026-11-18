"""How do POME 1.4's imputation and clustering results change with fewer training epochs?

The POME 1.4 results use 2000 epochs. This compares them with re-runs at 250, 500 and 1000 epochs
that are otherwise identical (same data, seeds and settings; outputs tagged ``_pome14_e{N}``):

  Imputation  d = 64, z-discretization with 15 bins: normalized MAE (numeric) and accuracy
              (categorical) of POME per missingness ratio, mean over the masking runs.
  Clustering  Leiden on the kNN graph of the embeddings (10 seeds x d = 16/32/64): silhouette and
              Davies-Bouldin (POME's squared Euclidean ``db_custom``), median and SD over seeds.

Inputs:
  data/{DATASET}_imputation_z_bins_15_pome14[_e{N}].csv          (generate_imputation_results.py)
  output/unsupervised/{DATASET}_leiden_knn_clustering_scores_pome14[_e{N}].csv
                                                                  (generate_unsupervised_figure.py)
Outputs: output/training_epochs/{imputation,clustering}_by_epochs.csv and
         training_epochs.{pdf,png}

Pipeline (N in 250 500 1000; the 2000-epoch results are the POME 1.4 defaults):
  sbatch container/submit.sh src/pome_evaluation/impute_graph_based.py --discretizations z \\
      --bins 15 --dims 64 --datasets $DS --epochs $N --output-suffix _pome14_e$N
  sbatch container/submit.sh scripts/generate_embeddings.py --dims 16 32 64 --runs 10 \\
      --epochs $N --embeddings-root data/embeddings_pome14_e$N
  container/run.sh scripts/generate_imputation_results.py --discretization z --bins 15 \\
      --dims 64 --pome-suffix _pome14_e$N
  container/run.sh scripts/generate_unsupervised_figure.py \\
      --embeddings-root data/embeddings_pome14_e$N --tag _pome14_e$N --no-plot
  container/run.sh scripts/compare_training_epochs.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
UNSUP_DIR = PROJECT_ROOT / "output" / "unsupervised"
OUT_DIR = PROJECT_ROOT / "output" / "training_epochs"

EPOCHS = [250, 500, 1000, 2000]
DATASETS = [("HANCOCK", "HANCOCK"), ("TCGA_LUAD", "TCGA-LUAD"), ("MIMIC", "MIMIC-IV")]
# Ordinal blue ramp (light -> dark) for ordered series (missingness ratio, dimension).
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, INK_2 = "#1f1f1e", "#5f5e5a"


def tag(epochs):
    return "_pome14" if epochs == 2000 else f"_pome14_e{epochs}"


def load_imputation():
    rows = []
    for label, _ in DATASETS:
        for ep in EPOCHS:
            path = DATA_ROOT / f"{label}_imputation_z_bins_15{tag(ep)}.csv"
            if not path.exists():
                print(f"[warn] missing {path.name}")
                continue
            df = pd.read_csv(path, index_col=0)
            rows.append(df[df["tool"] == "POME"].assign(dataset=label, epochs=ep))
    df = pd.concat(rows, ignore_index=True)
    return (df.groupby(["dataset", "na_ratio", "epochs"])
            .agg(nmae_mean=("nmae_cont", "mean"), nmae_sd=("nmae_cont", "std"),
                 acc_mean=("acc_cat", "mean"), acc_sd=("acc_cat", "std"),
                 runs=("run", "nunique"))
            .reset_index())


def load_clustering():
    rows = []
    for label, _ in DATASETS:
        for ep in EPOCHS:
            path = UNSUP_DIR / f"{label}_leiden_knn_clustering_scores{tag(ep)}.csv"
            if not path.exists():
                print(f"[warn] missing {path.name}")
                continue
            df = pd.read_csv(path)
            rows.append(df[df["method"] == "POME_KNN_Leiden"].assign(dataset=label, epochs=ep))
    df = pd.concat(rows, ignore_index=True)
    return (df.groupby(["dataset", "dim", "epochs"])
            .agg(silhouette_median=("silhouette", "median"), silhouette_sd=("silhouette", "std"),
                 db_median=("db_custom", "median"), db_sd=("db_custom", "std"),
                 k_median=("k", "median"), seeds=("run", "nunique"))
            .reset_index())


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xscale("log", base=2)
    ax.set_xticks(EPOCHS, [str(e) for e in EPOCHS])
    ax.minorticks_off()


def series(ax, df, key, value, labeller):
    levels = sorted(df[key].unique())
    ramp = RAMP[-len(levels):] if len(levels) <= len(RAMP) else RAMP
    for lvl, color in zip(levels, ramp):
        g = df[df[key] == lvl].sort_values("epochs")
        ax.plot(g["epochs"], g[value], color=color, lw=2, marker="o", markersize=4,
                label=labeller(lvl))


def plot(imp, clu, path):
    panels = [(imp, "na_ratio", "nmae_mean", "Numeric imputation\nnormalized MAE",
               lambda r: f"{r:.0%} missing"),
              (imp, "na_ratio", "acc_mean", "Categorical imputation\naccuracy",
               lambda r: f"{r:.0%} missing"),
              (clu, "dim", "silhouette_median", "Clustering\nsilhouette (median of seeds)",
               lambda d: f"d = {d}"),
              (clu, "dim", "db_median", "Clustering\nDavies-Bouldin (median of seeds)",
               lambda d: f"d = {d}")]
    fig, axes = plt.subplots(len(panels), 3, figsize=(10.5, 10), constrained_layout=True,
                             sharex=True)
    for row, (df, key, value, ylabel, labeller) in enumerate(panels):
        for col, (label, title) in enumerate(DATASETS):
            ax = axes[row, col]
            style(ax)
            series(ax, df[df["dataset"] == label], key, value, labeller)
            if row == 0:
                ax.set_title(title, fontsize=10, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=INK)
            if row == len(panels) - 1:
                ax.set_xlabel("Training epochs", fontsize=9, color=INK)
            if col == 2 and row in (0, 2):
                ax.legend(fontsize=7, frameon=False, loc="best")
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print(f"wrote {path.with_suffix('.pdf')}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    imp, clu = load_imputation(), load_clustering()
    imp.to_csv(args.out_dir / "imputation_by_epochs.csv", index=False)
    clu.to_csv(args.out_dir / "clustering_by_epochs.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(imp.round(4).to_string(index=False))
        print(clu.round(4).to_string(index=False))
    plot(imp, clu, args.out_dir / "training_epochs")


if __name__ == "__main__":
    main()
