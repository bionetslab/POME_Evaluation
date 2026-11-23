"""Unsupervised clustering / 2D-visualization figure (manuscript Fig. 3, panels a-j).

Scriptable port of the legacy chain
    src/pome_evaluation/analyze_unsupervised_clustering.py      (panels a-g scores)
    src/pome_evaluation/project_embeddings_to_2D.ipynb          (2D projections)
    src/pome_evaluation/analyze_distance_preservation_2D.ipynb  (panel h)
    src/pome_evaluation/analyze_cluster_preservation_2D.ipynb   (panels i, j)
    scripts/plot_fig4_unsupervised.ipynb                         (plot, panels a-j)
with the same metrics and settings, parameterized by the POME embedding root so that
re-trained POME embeddings (by default the POME 1.4 embeddings of ``generate_embeddings.py``
under ``data/embeddings_pome14``) can be evaluated without touching the published results.
Panels k-m (HANCOCK case study) are not part of this script.

Steps (each cached; rerun with --overwrite to recompute):
  1. Clustering scores. Per dataset x dim (16, 32, 64) x run (0-9): Leiden (RBConfiguration)
     on the kNN graph (k = 5 % of samples, squared Euclidean) of the POME sample embedding;
     Silhouette (squared Euclidean) and Davies-Bouldin (custom, squared Euclidean). The UMAP
     rows are copied from the published score tables (output/unsupervised/
     {DATASET}_leiden_knn_clustering_scores.csv): the UMAP embeddings are unchanged and the
     MIMIC ones are not stored in this repo. Per-method metrics, as published: POME is
     clustered and scored in squared Euclidean (= cosine; unit-length embeddings), UMAP in
     Euclidean (sklearn davies_bouldin_score). The published UMAP db_custom column (a separate
     clustering of L2-normalized UMAP embeddings) is not used and is left empty here.
     -> output/unsupervised/{DATASET}_leiden_knn_clustering_scores{tag}.csv
  1b. Competitor scores (independent of the POME tag). Embeddings of the same sizes and runs
     from data/embeddings_competitors/{DATASET}/embeddings/{DATASET}_{METHOD}_{dim}_{run}.tsv:
     HIVAE (scripts/embed_hivae.py), SAINT (SAINT's own missing-value masking,
     scripts/embed_saint.py) and MFAE = missForest + autoencoder
     (scripts/impute_full_missforest.py, scripts/embed_autoencoder.py). Scored like UMAP: Leiden
     (RBConfiguration, seed = run) on the Euclidean kNN graph (k = 5 % of samples), Euclidean
     Silhouette, sklearn Davies-Bouldin. A method is skipped while any of its embeddings is
     missing.
     -> output/unsupervised/{DATASET}_leiden_knn_clustering_scores_competitors.csv
        output/unsupervised/{DATASET}_leiden_knn_labels_competitors.csv
  2. 2D projections (PCA, UMAP, t-SNE of the standardized embedding; random_state 42).
     -> {embeddings-root}/{DATASET}/visualization/{DATASET}_samples_{dim}_{run}.tsv
  3. Distance preservation: mean Jaccard overlap of each sample's K nearest neighbours
     (K = 10, 30, 50) in the embedding vs. the projection; cluster preservation: Silhouette /
     Davies-Bouldin of the step-1 Leiden labels computed in the 2D projection.
     -> output/unsupervised/visualization_{distance,cluster}_preservation_results{tag}.csv
     The same for the competitors (projections of their embeddings, their step-1b labels;
     parallel with --jobs):
     -> data/embeddings_competitors/{DATASET}/visualization/{DATASET}_{METHOD}_{dim}_{run}.tsv
        output/unsupervised/{DATASET}_visualization_{distance,cluster}_preservation_competitors.csv
     Panels h-j show POME per dataset when it is the only embedding with projections
     (--methods POME UMAP: UMAP's MIMIC embeddings are not stored), otherwise one row per
     embedding method (boxes over datasets x runs, means over embedding sizes).
  4. Figure -> output/unsupervised/unsupervised_clustering{tag}.pdf

Difference to the legacy code: Leiden is seeded (seed = run) so that the scores are
reproducible, and step 3 reuses the step-1 labels instead of re-running unseeded Leiden.

CPU:  container/run.sh scripts/generate_unsupervised_figure.py
      (= --embeddings-root data/embeddings_pome14 --tag _pome14)
"""

import argparse
import os
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
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms as transforms  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import davies_bouldin_score, silhouette_score  # noqa: E402
from sklearn.neighbors import kneighbors_graph  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pome_evaluation.analyze_unsupervised_clustering import (  # noqa: E402
    davies_bouldin_score_custom)
from pome_evaluation.tabular_inputs import COMPETITOR_ROOT, embedding_path  # noqa: E402

DEFAULT_EMB_ROOT = PROJECT_ROOT / "data" / "embeddings_pome14"
OUT_DIR = PROJECT_ROOT / "output" / "unsupervised"
DATASETS = ["HANCOCK", "MIMIC", "TCGA_LUAD"]
DISPLAY = {"HANCOCK": "HANCOCK", "MIMIC": "MIMIC-IV", "TCGA_LUAD": "TCGA-LUAD"}
DIMS = [16, 32, 64]
N_RUNS = 10
K_LIST = [10, 30, 50]
PROJECTIONS = ["UMAP", "PCA", "tSNE"]
# file label of a competitor embedding -> name in the figure
COMPETITORS = {"HIVAE": "HI-VAE", "SAINT": "SAINT", "MFAE": "missForest+AE"}
DISPLAY_METHOD = {"POME": "POME", "UMAP": "UMAP", **COMPETITORS}
DEFAULT_METHODS = ["POME", "UMAP", "HI-VAE", "SAINT", "missForest+AE"]


# --- 1. clustering ------------------------------------------------------------

def leiden_labels(X, seed, metric="sqeuclidean"):
    n_neighbors = int(0.05 * len(X))
    adj = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity",
                           include_self=False, metric=metric)
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
    # The published UMAP db_custom is a different measurement (Euclidean DB of a separate Leiden
    # run on L2-normalized UMAP embeddings); it is not UMAP's score in this figure (that is
    # davies_bouldin), so it is blanked to avoid reading it as a common-metric UMAP value.
    umap = published[published["method"].str.startswith("UMAP")].assign(db_custom=np.nan)
    pd.concat([pome, umap], ignore_index=True).to_csv(out, index=False)
    pd.concat(labels, ignore_index=True).to_csv(labels_out, index=False)


def competitor_scores(dataset, overwrite):
    out = OUT_DIR / f"{dataset}_leiden_knn_clustering_scores_competitors.csv"
    labels_out = OUT_DIR / f"{dataset}_leiden_knn_labels_competitors.csv"
    cached = out.exists() and labels_out.exists() and not overwrite
    old = pd.read_csv(out) if cached else pd.DataFrame(columns=["method"])
    old_labels = pd.read_csv(labels_out, dtype={"sample": str}) if cached \
        else pd.DataFrame(columns=["method"])
    frames, label_frames = [], []
    for method in COMPETITORS:
        tag = f"{method}_KNN_Leiden"
        paths = {(dim, run): Path(embedding_path(dataset, method, dim, run))
                 for dim in DIMS for run in range(N_RUNS)}
        if (old["method"] == tag).sum() == len(paths) and (old_labels["method"] == method).any():
            frames.append(old[old["method"] == tag])
            label_frames.append(old_labels[old_labels["method"] == method])
            continue
        missing = [p for p in paths.values() if not p.exists()]
        if missing:
            print(f"  [skip] {dataset} {method}: {len(missing)} of {len(paths)} embeddings missing")
            continue
        rows = []
        for (dim, run), path in paths.items():
            emb = pd.read_csv(path, sep="\t", index_col=0)
            X = emb.to_numpy()
            lab = leiden_labels(X, seed=run, metric="euclidean")
            rows.append(dict(run=run, k=len(np.unique(lab)),
                             silhouette=silhouette_score(X, lab),
                             davies_bouldin=davies_bouldin_score(X, lab),
                             method=tag, dim=dim))
            label_frames.append(pd.DataFrame(dict(sample=emb.index.astype(str), dim=dim,
                                                  run=run, method=method, label=lab)))
            print(f"  [cluster] {dataset} {method} dim {dim} run {run}: k={rows[-1]['k']} "
                  f"sil={rows[-1]['silhouette']:.3f}", flush=True)
        frames.append(pd.DataFrame(rows))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        pd.concat(label_frames, ignore_index=True).to_csv(labels_out, index=False)


# --- 2. projections -------------------------------------------------------------

def project_matrix(emb):
    """PCA / UMAP / t-SNE (2D) of the standardized embedding (DataFrame, samples as index)."""
    import umap.umap_ as umap
    X = StandardScaler().fit_transform(emb.to_numpy())
    parts = {
        "PCA": PCA(n_components=2, random_state=42).fit_transform(X),
        "UMAP": umap.UMAP(n_components=2, metric="euclidean",
                          random_state=42).fit_transform(X),
        "tSNE": TSNE(n_components=2, init="pca", random_state=42).fit_transform(X),
    }
    return pd.concat([pd.DataFrame(v, index=emb.index, columns=[f"{k}_1", f"{k}_2"])
                      for k, v in parts.items()], axis=1)


def project(root, dataset, overwrite):
    vis_dir = root / dataset / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for dim in DIMS:
        for run in range(N_RUNS):
            path = vis_dir / f"{dataset}_samples_{dim}_{run}.tsv"
            if path.exists() and not overwrite:
                continue
            project_matrix(load_embedding(root, dataset, dim, run)).to_csv(path, sep="\t")
            print(f"  [project] {dataset} dim {dim} run {run}", flush=True)


def competitor_vis_path(dataset, method, dim, run):
    return Path(COMPETITOR_ROOT) / dataset / "visualization" / f"{dataset}_{method}_{dim}_{run}.tsv"


def load_competitor(dataset, method, dim, run):
    emb = pd.read_csv(embedding_path(dataset, method, dim, run), sep="\t", index_col=0)
    emb.index = emb.index.astype(str)
    return emb


def _project_competitor(task):
    dataset, method, dim, run = task
    path = competitor_vis_path(dataset, method, dim, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    project_matrix(load_competitor(dataset, method, dim, run)).to_csv(path, sep="\t")
    return task


def project_competitors(dataset, overwrite, jobs):
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(dataset, m, dim, run) for m in COMPETITORS for dim in DIMS for run in range(N_RUNS)
             if Path(embedding_path(dataset, m, dim, run)).exists()
             and (overwrite or not competitor_vis_path(dataset, m, dim, run).exists())]
    if not tasks:
        return
    # Spawned (not forked) single-threaded workers: forking after the clustering step has
    # started OpenMP/numba thread pools deadlocks the children.
    import multiprocessing
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    with ProcessPoolExecutor(max_workers=max(1, jobs),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for task in pool.map(_project_competitor, tasks):
            print("  [project] %s %s dim %d run %d" % task, flush=True)


# --- 3. preservation ------------------------------------------------------------

def knn_order(X):
    return np.argsort(cdist(X, X, metric="sqeuclidean"), axis=1)


def jaccard(order_a, order_b, k):
    a, b = order_a[:, 1:k + 1], order_b[:, 1:k + 1]
    return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]))


def preservation_rows(emb, vis, lab, base):
    """Distance (K-NN Jaccard) and cluster (Silhouette / DB of `lab`) preservation rows."""
    dist_rows, clust_rows = [], []
    vis = vis.loc[emb.index]
    emb_order = knn_order(emb.to_numpy())
    for method in PROJECTIONS:
        P = vis[[f"{method}_1", f"{method}_2"]].to_numpy()
        p_order = knn_order(P)
        for k in K_LIST:
            dist_rows.append(dict(base, method=method, K=k,
                                  jaccard_index=jaccard(emb_order, p_order, k)))
        clust_rows.append(dict(base, method=method, silhouette=silhouette_score(P, lab),
                               davies_bouldin=davies_bouldin_score_custom(P, lab)))
    return dist_rows, clust_rows


def read_vis(path):
    vis = pd.read_csv(path, sep="\t", index_col=0)
    vis.index = vis.index.astype(str)
    return vis


def labels_for(labels, index, dim, run):
    lab = labels[(labels["dim"] == dim) & (labels["run"] == run)]
    return lab.set_index("sample").loc[index, "label"].to_numpy()


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
                vis = read_vis(root / dataset / "visualization" /
                               f"{dataset}_samples_{dim}_{run}.tsv")
                d, c = preservation_rows(emb, vis, labels_for(labels, emb.index, dim, run),
                                         dict(run=run, dim=dim, dataset=dataset))
                dist_rows += d
                clust_rows += c
                print(f"  [preserve] {dataset} dim {dim} run {run}", flush=True)
    pd.DataFrame(dist_rows).to_csv(dist_out, index=False)
    pd.DataFrame(clust_rows).to_csv(clust_out, index=False)


def competitor_preservation(dataset, overwrite):
    """Same as `preservation`, for the competitors (each with its own step-1b Leiden labels)."""
    dist_out = OUT_DIR / f"{dataset}_visualization_distance_preservation_competitors.csv"
    clust_out = OUT_DIR / f"{dataset}_visualization_cluster_preservation_competitors.csv"
    labels_path = OUT_DIR / f"{dataset}_leiden_knn_labels_competitors.csv"
    if not labels_path.exists():
        return
    labels = pd.read_csv(labels_path, dtype={"sample": str})
    done = set()
    if dist_out.exists() and clust_out.exists() and not overwrite:
        old_d, old_c = pd.read_csv(dist_out), pd.read_csv(clust_out)
        done = set(old_c["embedding"])
    else:
        old_d, old_c = pd.DataFrame(), pd.DataFrame()
    dist_rows, clust_rows = [], []
    for method in COMPETITORS:
        if method in done:
            continue
        vis_paths = [competitor_vis_path(dataset, method, dim, run)
                     for dim in DIMS for run in range(N_RUNS)]
        if not (labels["method"] == method).any() or not all(p.exists() for p in vis_paths):
            continue
        lab_m = labels[labels["method"] == method]
        for dim in DIMS:
            for run in range(N_RUNS):
                emb = load_competitor(dataset, method, dim, run)
                vis = read_vis(competitor_vis_path(dataset, method, dim, run))
                d, c = preservation_rows(emb, vis, labels_for(lab_m, emb.index, dim, run),
                                         dict(run=run, dim=dim, dataset=dataset,
                                              embedding=method))
                dist_rows += d
                clust_rows += c
            print(f"  [preserve] {dataset} {method} dim {dim}", flush=True)
    if dist_rows:
        old_d = old_d[~old_d.get("embedding", pd.Series(dtype=str)).isin(
            {r["embedding"] for r in clust_rows})] if len(old_d) else old_d
        old_c = old_c[~old_c.get("embedding", pd.Series(dtype=str)).isin(
            {r["embedding"] for r in clust_rows})] if len(old_c) else old_c
        pd.concat([old_d, pd.DataFrame(dist_rows)], ignore_index=True).to_csv(dist_out, index=False)
        pd.concat([old_c, pd.DataFrame(clust_rows)], ignore_index=True).to_csv(clust_out,
                                                                                index=False)


# --- 4. figure --------------------------------------------------------------------

def load_scores(tag, methods):
    frames = []
    for dataset in DATASETS:
        df = pd.read_csv(OUT_DIR / f"{dataset}_leiden_knn_clustering_scores{tag}.csv")
        comp = OUT_DIR / f"{dataset}_leiden_knn_clustering_scores_competitors.csv"
        if comp.exists():
            df = pd.concat([df, pd.read_csv(comp)], ignore_index=True)
        df["Dataset"] = DISPLAY[dataset]
        frames.append(df)
    scores = pd.concat(frames, ignore_index=True)
    scores["Method"] = scores["method"].str.split("_").str[0].map(DISPLAY_METHOD)
    # POME: custom (squared-Euclidean) DB index; others: sklearn DB index (UMAP as published).
    scores["db_final"] = np.where(scores["Method"] == "POME", scores["db_custom"],
                                  scores["davies_bouldin"])
    absent = [m for m in methods if m not in set(scores["Method"])]
    if absent:
        raise SystemExit(f"no scores for {absent}; compute them first (step 1b)")
    return scores[scores["Method"].isin(methods)]


def best_counts(scores, order):
    medians = scores.groupby(["Dataset", "dim", "Method"])[["silhouette", "db_final"]] \
        .median().reset_index()
    best = medians.groupby(["Dataset", "Method"]).agg(
        db_final=("db_final", "min"), silhouette=("silhouette", "max")).reset_index()
    best["best_sil"] = best.groupby("Dataset")["silhouette"].transform("max") == best["silhouette"]
    best["best_db"] = best.groupby("Dataset")["db_final"].transform("min") == best["db_final"]
    out = []
    for col, name in (("best_sil", "Silhouette"), ("best_db", "Davies-Bouldin")):
        res = best[best[col]].groupby("Method").size().reindex(order, fill_value=0)
        out.append(pd.DataFrame(dict(Method=res.index, max_on_dataset=res.values, score=name)))
    return pd.concat(out, ignore_index=True)


def preservation_frames(tag, methods):
    """Per (dataset, embedding, projection, run) means over embedding sizes; K = 10."""
    dist = [pd.read_csv(OUT_DIR / f"visualization_distance_preservation_results{tag}.csv")
            .assign(embedding="POME")]
    clust = [pd.read_csv(OUT_DIR / f"visualization_cluster_preservation_results{tag}.csv")
             .assign(embedding="POME")]
    for dataset in DATASETS:
        for kind, frames in (("distance", dist), ("cluster", clust)):
            path = OUT_DIR / f"{dataset}_visualization_{kind}_preservation_competitors.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    dist, clust = pd.concat(dist, ignore_index=True), pd.concat(clust, ignore_index=True)
    for df in (dist, clust):
        df["embedding"] = df["embedding"].map(DISPLAY_METHOD)
    dist = dist[(dist["K"] == 10) & dist["embedding"].isin(methods)]
    clust = clust[clust["embedding"].isin(methods)]
    keys = ["dataset", "embedding", "method", "run"]
    mean_dist = dist.groupby(keys, as_index=False)["jaccard_index"].mean()
    sil = clust.groupby(keys, as_index=False)["silhouette"].mean()
    db = clust.groupby(keys, as_index=False)["davies_bouldin"].mean()
    names = {"TCGA_LUAD": "TCGA-LUAD", "MIMIC": "MIMIC-IV"}
    return [f.replace(names) for f in (mean_dist, sil, db)]


def ylim_covering(ax, lo, hi, values):
    values = np.asarray(values, dtype=float)
    pad = 0.05 * (hi - lo)
    ax.set_ylim(min(lo, np.nanmin(values) - pad), max(hi, np.nanmax(values) + pad))
    if np.nanmin(values) < lo or np.nanmax(values) > hi:     # fixed ticks would not cover it
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))


def plot(tag, out_path, methods):
    order = list(methods)
    scores = load_scores(tag, order)
    agg = best_counts(scores, order)
    mean_dist, sil, db = preservation_frames(tag, order)
    # h-j: every embedding with stored projections (UMAP's MIMIC embeddings are not stored)
    hj_order = [m for m in order if m in set(sil["embedding"])]

    mosaic = [["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"]]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(16, 9), layout="constrained")
    label_fs, title_fs, tick_fs, legend_fs, legend_title_fs = 19, 19, 15, 15, 17
    offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)
    set2 = sns.color_palette("Set2")

    def letter(key, size=label_fs, title_lines=1):
        # raised by one title line height per extra title line
        shift = transforms.ScaledTranslation(0, (title_lines - 1) * 1.2 * title_fs / 72,
                                             fig.dpi_scale_trans)
        axes[key].text(0.0, 1.0, key, transform=axes[key].transAxes + offset + shift,
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
    legend1 = fig.legend(handles, labels, loc="center",
                         bbox_to_anchor=(0.34, 1.04 if len(labels) <= 3 else 1.075),
                         ncol=len(labels) if len(labels) <= 3 else 2, fontsize=legend_fs,
                         frameon=True, fancybox=False,
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
    if hj_order == ["POME"]:
        # published layout: POME only, one row per dataset
        hj_y, hj_rows, hj_suffix = "dataset", ds_order, ""
    else:
        # one row per embedding method, boxes over datasets x runs (means over sizes)
        hj_y, hj_rows, hj_suffix = "embedding", hj_order, "\n(all datasets)"
    sns.boxplot(data=mean_dist, x="jaccard_index", y=hj_y, hue="method",
                hue_order=PROJECTIONS, order=hj_rows, ax=axes["h"], palette=husl)
    style(axes["h"], "Mean Jaccard index", "", "Distance preservation" + hj_suffix)
    letter("h", title_lines=hj_suffix.count("\n") + 1)
    handles2, labels2 = axes["h"].get_legend_handles_labels()
    axes["h"].legend_.remove()
    legend2 = fig.legend(handles2, labels2, loc="center", bbox_to_anchor=(0.69, 1.04),
                         ncol=len(labels2), fontsize=legend_fs, frameon=True, fancybox=False,
                         title="2D projection of embeddings", title_fontsize=legend_title_fs)
    sns.boxplot(data=sil, x="silhouette", y=hj_y, hue="method", hue_order=PROJECTIONS,
                order=hj_rows, ax=axes["i"], legend=False, palette=husl, showfliers=False)
    style(axes["i"], "Silhouette score", "", "Cluster preservation" + hj_suffix)
    letter("i", title_lines=hj_suffix.count("\n") + 1)
    sns.boxplot(data=db, x="davies_bouldin", y=hj_y, hue="method", hue_order=PROJECTIONS,
                order=hj_rows, ax=axes["j"], legend=False, palette=husl, showfliers=False)
    style(axes["j"], "Davies-Bouldin index", "", "Cluster preservation" + hj_suffix)
    letter("j", label_fs + 2, title_lines=hj_suffix.count("\n") + 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", bbox_extra_artists=[legend1, legend2])
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMB_ROOT,
                   help="POME embedding root (<root>/<DATASET>/embeddings/...)")
    p.add_argument("--tag", default="_pome14",
                   help="suffix of the result CSVs / figure (default: _pome14)")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   choices=list(DISPLAY_METHOD.values()),
                   help="methods in panels a-g, in this order (default: %(default)s; "
                        "POME UMAP = the published figure)")
    p.add_argument("--plot-only", action="store_true", help="skip steps 1-3")
    p.add_argument("--no-plot", action="store_true",
                   help="skip step 4 (e.g. when only a subset of datasets was computed)")
    p.add_argument("--overwrite", action="store_true", help="recompute cached steps")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel processes for the competitor 2D projections")
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    if not args.plot_only:
        for dataset in args.datasets:
            clustering_scores(args.embeddings_root, dataset, args.tag, args.overwrite)
            competitor_scores(dataset, args.overwrite)
            project(args.embeddings_root, dataset, args.overwrite)
            project_competitors(dataset, args.overwrite, args.jobs)
            competitor_preservation(dataset, args.overwrite)
        preservation(args.embeddings_root, args.tag, args.overwrite, args.datasets)
    if not args.no_plot:
        suffix = "" if args.methods == ["POME", "UMAP"] else "_competitors"
        plot(args.tag, OUT_DIR / f"unsupervised_clustering{args.tag}{suffix}.pdf", args.methods)


if __name__ == "__main__":
    main()
