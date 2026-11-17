"""Unsupervised clustering / 2D-visualization figure (manuscript Fig. 3, panels a-j).

Scriptable port of the legacy chain
    src/pome_evaluation/analyze_unsupervised_clustering.py      (panels a-g scores)
    src/pome_evaluation/project_embeddings_to_2D.ipynb          (2D projections)
    src/pome_evaluation/analyze_distance_preservation_2D.ipynb  (panel h)
    src/pome_evaluation/analyze_cluster_preservation_2D.ipynb   (panels i, j)
    scripts/plot_fig4_unsupervised.ipynb                         (plot, panels a-j)
with the same metrics and settings, parameterized by the POME embedding root so that a
re-trained POME variant (e.g. ``generate_embeddings.py --pome-config stable
--embeddings-root data/embeddings_stable``) can be evaluated without touching the published
results. Panels k-m (HANCOCK case study) are not part of this script.

Steps (each cached; rerun with --overwrite to recompute):
  1. Clustering scores. Per dataset x dim (16, 32, 64) x run (0-9): Leiden (RBConfiguration)
     on the kNN graph (k = 5 % of samples, squared Euclidean) of the POME sample embedding;
     Silhouette (squared Euclidean) and Davies-Bouldin (custom, squared Euclidean). The UMAP
     rows are copied from the published score tables (output/unsupervised/
     {DATASET}_leiden_knn_clustering_scores.csv): the UMAP embeddings are unchanged and the
     MIMIC ones are not stored in this repo.
     -> output/unsupervised/{DATASET}_leiden_knn_clustering_scores{tag}.csv
  2. 2D projections (PCA, UMAP, t-SNE of the standardized embedding; random_state 42).
     -> {embeddings-root}/{DATASET}/visualization/{DATASET}_samples_{dim}_{run}.tsv
  3. Distance preservation: mean Jaccard overlap of each sample's K nearest neighbours
     (K = 10, 30, 50) in the embedding vs. the projection; cluster preservation: Silhouette /
     Davies-Bouldin of the step-1 Leiden labels computed in the 2D projection.
     -> output/unsupervised/visualization_{distance,cluster}_preservation_results{tag}.csv
  4. Figure -> output/unsupervised/unsupervised_clustering_{tag-or-published}.pdf

Difference to the legacy code: Leiden is seeded (seed = run) so that the scores are
reproducible, and step 3 reuses the step-1 labels instead of re-running unseeded Leiden.

CPU:  container/run.sh scripts/generate_unsupervised_figure.py \\
          --embeddings-root data/embeddings_stable --tag _stable
"""

import argparse
import sys
import warnings
from pathlib import Path

import igraph as ig
import leidenalg as la
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as transforms  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.neighbors import kneighbors_graph  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pome_evaluation.analyze_unsupervised_clustering import (  # noqa: E402
    davies_bouldin_score_custom)

DEFAULT_EMB_ROOT = PROJECT_ROOT / "data" / "embeddings"
OUT_DIR = PROJECT_ROOT / "output" / "unsupervised"
DATASETS = ["HANCOCK", "MIMIC", "TCGA_LUAD"]
DISPLAY = {"HANCOCK": "HANCOCK", "MIMIC": "MIMIC-IV", "TCGA_LUAD": "TCGA-LUAD"}
DIMS = [16, 32, 64]
N_RUNS = 10
K_LIST = [10, 30, 50]
PROJECTIONS = ["UMAP", "PCA", "tSNE"]


# --- 1. clustering ------------------------------------------------------------

def leiden_labels(X, seed):
    n_neighbors = int(0.05 * len(X))
    adj = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity",
                           include_self=False, metric="sqeuclidean")
    src, dst = adj.nonzero()
    graph = ig.Graph(n=adj.shape[0], edges=list(zip(src, dst)), directed=False)
    part = la.find_partition(graph, la.RBConfigurationVertexPartition, seed=seed)
    return np.array(part.membership)


def sample_path(root, dataset, dim, run):
    return root / dataset / "embeddings" / f"{dataset}_samples_{dim}_{run}.tsv"


def load_embedding(root, dataset, dim, run):
    df = pd.read_csv(sample_path(root, dataset, dim, run), sep="\t", index_col=0)
    df.index = df.index.astype(str)
    return df


def clustering_scores(root, dataset, tag, overwrite):
    out = OUT_DIR / f"{dataset}_leiden_knn_clustering_scores{tag}.csv"
    labels_out = OUT_DIR / f"{dataset}_leiden_knn_labels{tag}.csv"
    if out.exists() and labels_out.exists() and not overwrite:
        return
    rows, labels = [], []
    for dim in DIMS:
        for run in range(N_RUNS):
            emb = load_embedding(root, dataset, dim, run)
            X = emb.to_numpy()
            lab = leiden_labels(X, seed=run)
            rows.append(dict(run=run, k=len(np.unique(lab)),
                             silhouette=silhouette_score(X, lab, metric="sqeuclidean"),
                             db_custom=davies_bouldin_score_custom(X, lab, metric="sqeuclidean"),
                             method="POME_KNN_Leiden", dim=dim))
            labels.append(pd.DataFrame(dict(sample=emb.index, dim=dim, run=run, label=lab)))
            print(f"  [cluster] {dataset} dim {dim} run {run}: k={rows[-1]['k']} "
                  f"sil={rows[-1]['silhouette']:.3f}", flush=True)
    pome = pd.DataFrame(rows)
    published = pd.read_csv(OUT_DIR / f"{dataset}_leiden_knn_clustering_scores.csv")
    umap = published[published["method"].str.startswith("UMAP")]
    pd.concat([pome, umap], ignore_index=True).to_csv(out, index=False)
    pd.concat(labels, ignore_index=True).to_csv(labels_out, index=False)


# --- 2. projections -------------------------------------------------------------

def project(root, dataset, overwrite):
    import umap.umap_ as umap
    vis_dir = root / dataset / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for dim in DIMS:
        for run in range(N_RUNS):
            path = vis_dir / f"{dataset}_samples_{dim}_{run}.tsv"
            if path.exists() and not overwrite:
                continue
            emb = load_embedding(root, dataset, dim, run)
            X = StandardScaler().fit_transform(emb.to_numpy())
            parts = {
                "PCA": PCA(n_components=2, random_state=42).fit_transform(X),
                "UMAP": umap.UMAP(n_components=2, metric="euclidean",
                                  random_state=42).fit_transform(X),
                "tSNE": TSNE(n_components=2, init="pca", random_state=42).fit_transform(X),
            }
            df = pd.concat([pd.DataFrame(v, index=emb.index, columns=[f"{k}_1", f"{k}_2"])
                            for k, v in parts.items()], axis=1)
            df.to_csv(path, sep="\t")
            print(f"  [project] {dataset} dim {dim} run {run}", flush=True)


# --- 3. preservation ------------------------------------------------------------

def knn_order(X):
    return np.argsort(cdist(X, X, metric="sqeuclidean"), axis=1)


def jaccard(order_a, order_b, k):
    a, b = order_a[:, 1:k + 1], order_b[:, 1:k + 1]
    return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]))


def preservation(root, tag, overwrite, datasets=DATASETS):
    dist_out = OUT_DIR / f"visualization_distance_preservation_results{tag}.csv"
    clust_out = OUT_DIR / f"visualization_cluster_preservation_results{tag}.csv"
    if dist_out.exists() and clust_out.exists() and not overwrite:
        return
    dist_rows, clust_rows = [], []
    for dataset in datasets:
        labels = pd.read_csv(OUT_DIR / f"{dataset}_leiden_knn_labels{tag}.csv",
                             dtype={"sample": str})
        for dim in DIMS:
            for run in range(N_RUNS):
                emb = load_embedding(root, dataset, dim, run)
                vis = pd.read_csv(root / dataset / "visualization" /
                                  f"{dataset}_samples_{dim}_{run}.tsv", sep="\t", index_col=0)
                vis.index = vis.index.astype(str)
                vis = vis.loc[emb.index]
                lab = labels[(labels["dim"] == dim) & (labels["run"] == run)]
                lab = lab.set_index("sample").loc[emb.index, "label"].to_numpy()
                emb_order = knn_order(emb.to_numpy())
                for method in PROJECTIONS:
                    P = vis[[f"{method}_1", f"{method}_2"]].to_numpy()
                    p_order = knn_order(P)
                    for k in K_LIST:
                        dist_rows.append(dict(run=run, dim=dim, method=method, K=k,
                                              jaccard_index=jaccard(emb_order, p_order, k),
                                              dataset=dataset))
                    clust_rows.append(dict(run=run, dim=dim, method=method,
                                           silhouette=silhouette_score(P, lab),
                                           davies_bouldin=davies_bouldin_score_custom(P, lab),
                                           dataset=dataset))
                print(f"  [preserve] {dataset} dim {dim} run {run}", flush=True)
    pd.DataFrame(dist_rows).to_csv(dist_out, index=False)
    pd.DataFrame(clust_rows).to_csv(clust_out, index=False)


# --- 4. figure --------------------------------------------------------------------

def load_scores(tag):
    frames = []
    for dataset in DATASETS:
        df = pd.read_csv(OUT_DIR / f"{dataset}_leiden_knn_clustering_scores{tag}.csv")
        df["Dataset"] = DISPLAY[dataset]
        frames.append(df)
    scores = pd.concat(frames, ignore_index=True)
    scores["Method"] = scores["method"].str.split("_").str[0]
    # POME: custom (squared-Euclidean) DB index; UMAP: sklearn DB index (as published).
    scores["db_final"] = np.where(scores["Method"] == "POME", scores["db_custom"],
                                  scores["davies_bouldin"])
    return scores


def best_counts(scores):
    medians = scores.groupby(["Dataset", "dim", "Method"])[["silhouette", "db_final"]] \
        .median().reset_index()
    best = medians.groupby(["Dataset", "Method"]).agg(
        db_final=("db_final", "min"), silhouette=("silhouette", "max")).reset_index()
    best["best_sil"] = best.groupby("Dataset")["silhouette"].transform("max") == best["silhouette"]
    best["best_db"] = best.groupby("Dataset")["db_final"].transform("min") == best["db_final"]
    out = []
    for col, name in (("best_sil", "Silhouette"), ("best_db", "Davies-Bouldin")):
        res = best[best[col]].groupby("Method").size().reindex(["POME", "UMAP"], fill_value=0)
        out.append(pd.DataFrame(dict(Method=res.index, max_on_dataset=res.values, score=name)))
    return pd.concat(out, ignore_index=True)


def preservation_frames(tag):
    dist = pd.read_csv(OUT_DIR / f"visualization_distance_preservation_results{tag}.csv")
    clust = pd.read_csv(OUT_DIR / f"visualization_cluster_preservation_results{tag}.csv")
    dist = dist[dist["K"] == 10]
    mean_dist = dist.groupby(["dataset", "method", "run"], as_index=False)["jaccard_index"].mean()
    sil = clust.groupby(["dataset", "method", "run"], as_index=False)["silhouette"].mean()
    db = clust.groupby(["dataset", "run", "method"], as_index=False)["davies_bouldin"].mean()
    names = {"TCGA_LUAD": "TCGA-LUAD", "MIMIC": "MIMIC-IV"}
    return [f.replace(names) for f in (mean_dist, sil, db)]


def ylim_covering(ax, lo, hi, values):
    values = np.asarray(values, dtype=float)
    pad = 0.05 * (hi - lo)
    ax.set_ylim(min(lo, np.nanmin(values) - pad), max(hi, np.nanmax(values) + pad))


def plot(tag, out_path):
    scores = load_scores(tag)
    agg = best_counts(scores)
    mean_dist, sil, db = preservation_frames(tag)

    mosaic = [["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"]]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(16, 9), layout="constrained")
    label_fs, title_fs, tick_fs, legend_fs, legend_title_fs = 19, 19, 15, 15, 17
    offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)
    set2 = sns.color_palette("Set2")
    order = ["POME", "UMAP"]

    def letter(key, size=label_fs):
        axes[key].text(0.0, 1.0, key, transform=axes[key].transAxes + offset,
                       fontsize=size, fontweight="bold")

    def style(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.set_title(title, fontsize=title_fs)

    sns.barplot(data=agg, y="score", x="max_on_dataset", hue="Method", hue_order=order,
                ax=axes["a"], palette=set2)
    style(axes["a"], "Number of datasets with \nbest median score", "", "")
    axes["a"].set_title("Results aggregated \n over datasets and \n embedding sizes",
                        fontsize=title_fs - 1)
    letter("a")
    handles, labels = axes["a"].get_legend_handles_labels()
    axes["a"].legend_.remove()
    legend1 = fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.34, 1.04),
                         ncol=len(labels), fontsize=legend_fs, frameon=True, fancybox=False,
                         title="Embedding of POM data", title_fontsize=legend_title_fs)

    for key, ds in zip("bcd", ["HANCOCK", "MIMIC-IV", "TCGA-LUAD"]):
        data = scores[scores["Dataset"] == ds]
        sns.boxplot(data=data, x="dim", y="silhouette", hue="Method", hue_order=order,
                    ax=axes[key], dodge=True, gap=.2, legend=False, palette=set2)
        style(axes[key], "Embedding size", "Silhouette score", ds)
        axes[key].set_yticks([0.15, 0.3, 0.45, 0.6])
        ylim_covering(axes[key], 0.1, 0.65, data["silhouette"])
        letter(key)
    for key, ds in zip("efg", ["HANCOCK", "MIMIC-IV", "TCGA-LUAD"]):
        data = scores[scores["Dataset"] == ds]
        sns.boxplot(data=data, x="dim", y="db_final", hue="Method", hue_order=order,
                    ax=axes[key], legend=False, palette=set2)
        style(axes[key], "Embedding size", "Davies-Bouldin index", ds)
        axes[key].set_yticks([0.5, 1.3, 2.1, 2.9])
        ylim_covering(axes[key], 0.25, 3.1, data["db_final"])
        letter(key)

    husl = sns.color_palette("husl", 3)
    ds_order = ["HANCOCK", "MIMIC-IV", "TCGA-LUAD"]
    sns.boxplot(data=mean_dist, x="jaccard_index", y="dataset", hue="method",
                hue_order=PROJECTIONS, order=ds_order, ax=axes["h"], palette=husl)
    style(axes["h"], "Mean Jaccard index", "", "Distance preservation")
    letter("h")
    handles2, labels2 = axes["h"].get_legend_handles_labels()
    axes["h"].legend_.remove()
    legend2 = fig.legend(handles2, labels2, loc="center", bbox_to_anchor=(0.69, 1.04),
                         ncol=len(labels2), fontsize=legend_fs, frameon=True, fancybox=False,
                         title="2D projection of embeddings", title_fontsize=legend_title_fs)
    sns.boxplot(data=sil, x="silhouette", y="dataset", hue="method", hue_order=PROJECTIONS,
                order=ds_order, ax=axes["i"], legend=False, palette=husl, showfliers=False)
    style(axes["i"], "Silhouette score", "", "Cluster preservation")
    letter("i")
    sns.boxplot(data=db, x="davies_bouldin", y="dataset", hue="method", hue_order=PROJECTIONS,
                order=ds_order, ax=axes["j"], legend=False, palette=husl, showfliers=False)
    style(axes["j"], "Davies-Bouldin index", "", "Cluster preservation")
    letter("j", label_fs + 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", bbox_extra_artists=[legend1, legend2])
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMB_ROOT,
                   help="POME embedding root (<root>/<DATASET>/embeddings/...)")
    p.add_argument("--tag", default="_stable",
                   help="suffix of the result CSVs / figure (default: _stable)")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--plot-only", action="store_true", help="skip steps 1-3")
    p.add_argument("--no-plot", action="store_true",
                   help="skip step 4 (e.g. when only a subset of datasets was computed)")
    p.add_argument("--overwrite", action="store_true", help="recompute cached steps")
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    if not args.plot_only:
        for dataset in args.datasets:
            clustering_scores(args.embeddings_root, dataset, args.tag, args.overwrite)
            project(args.embeddings_root, dataset, args.overwrite)
        preservation(args.embeddings_root, args.tag, args.overwrite, args.datasets)
    if not args.no_plot:
        plot(args.tag, OUT_DIR / f"unsupervised_clustering{args.tag}.pdf")


if __name__ == "__main__":
    main()
