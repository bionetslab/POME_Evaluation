"""Downstream effect of a joint (shared) sample encoding in POME: probing, clustering, imputation.

Reads the repeated-seed runs of ``generate_shared_sample_encoding_runs.py`` (configs
``individual``, ``individual_latent``, ``shared_learnable``, ``shared_fixed``) and computes,
per run,

  * linear probing     test AP of the inductive-probing logreg (``probe_one``, standardized
                       features) per binary target, ``probe_ap`` = mean over targets;
  * clustering         Leiden on the kNN graph of the train embeddings, as in the paper's
                       unsupervised analysis (k = 5 % of samples, sqeuclidean, RBConfiguration,
                       seed 0): number of clusters, silhouette (sqeuclidean), Davies-Bouldin,
                       Calinski-Harabasz, normalized cluster entropy; plus the effective rank
                       (RankMe) of the train embedding as a dimensional-collapse check;
  * imputation         numeric NRMSE (error / variable SD of the observed training values) and
                       categorical accuracy on the fixed 10 % cell mask;

and the seed-to-seed agreement measures of the training-noise analysis (CKA, 10-NN overlap,
k-means ARI, per-cell imputation jitter), reusing ``analyze_training_noise_variance``.
A ``mean_imputation`` reference row gives the NRMSE of imputing every masked numeric cell with
the observed training mean of its variable.

Each config is compared with ``individual`` (POME 1.2) per dataset x epochs: difference of the
means over seeds and a two-sided Mann-Whitney U p-value (seeds are independent runs).

CPU:  container/run.sh scripts/analyze_shared_sample_encoding.py [--root DIR]
      (follow-up: --root output/shared_sample_encoding_followup)
Outputs: {root}/per_seed_metrics.csv     one row per run
         {root}/summary.csv              mean / SD over seeds + agreement measures per config
         {root}/comparison.csv           config vs individual: delta of means, Mann-Whitney p
"""

import argparse
import sys
import warnings
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import analyze_training_noise_variance as tn  # noqa: E402
from linear_probe_inductive_embeddings import load_binary_targets  # noqa: E402
from pome_evaluation.analyze_unsupervised_clustering import normalized_cluster_entropy  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "output" / "shared_sample_encoding"
CONFIG_ORDER = ["individual", "individual_latent", "shared_learnable", "shared_fixed",
                "individual_sdrop50", "individual_mask30", "shared_mask30", "shared_heads4",
                "shared_mask30_heads4"]
METRICS = ["probe_ap", "imp_nrmse", "imp_acc", "leiden_k", "silhouette", "davies_bouldin",
           "calinski_harabasz", "entropy", "eff_rank"]
LOWER_IS_BETTER = {"imp_nrmse", "davies_bouldin"}
# Descriptive only (no better/worse direction): cluster count, balance, effective rank.
UNDIRECTED = {"leiden_k", "entropy", "eff_rank"}


def effective_rank(X):
    """RankMe: exp of the entropy of the normalized singular values of the centred embedding."""
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def clustering_metrics(X):
    """Leiden-on-kNN clustering of one embedding, as in analyze_unsupervised_clustering."""
    adj = kneighbors_graph(X, n_neighbors=int(0.05 * len(X)), mode="connectivity",
                           include_self=False, metric="sqeuclidean")
    src, dst = adj.nonzero()
    g = ig.Graph(n=adj.shape[0], edges=list(zip(src, dst)), directed=False)
    labels = np.array(la.find_partition(g, la.RBConfigurationVertexPartition, seed=0).membership)
    out = {"leiden_k": len(np.unique(labels)), "entropy": normalized_cluster_entropy(labels),
           "eff_rank": effective_rank(X)}
    if out["leiden_k"] > 1:
        out.update(silhouette=float(silhouette_score(X, labels, metric="sqeuclidean")),
                   davies_bouldin=float(davies_bouldin_score(X, labels)),
                   calinski_harabasz=float(calinski_harabasz_score(X, labels)))
    return out


def mean_imputation_nrmse(dataset, split, mask, sd):
    """NRMSE of imputing each masked numeric cell with its variable's observed training mean."""
    df = pd.read_csv(tn.SPLITS_ROOT / dataset / f"split_{split:02d}_train_pome.tsv",
                     sep="\t", index_col=0)
    vals = df.drop(columns=["type"])
    vals.columns = vals.columns.astype(str)
    masked = set(zip(mask["variable"], mask["sample"]))
    num = mask[(mask["type"] == "cont") & mask["variable"].isin(sd.index)]
    means = {}
    for var in num["variable"].unique():
        row = vals.loc[var]
        means[var] = row[[s for s in row.index
                          if row[s] != -99.0 and (var, s) not in masked]].mean()
    z = (num["variable"].map(means) - num["true"]) / num["variable"].map(sd)
    return float(np.sqrt(np.mean(z ** 2)))


def compare(per_seed):
    """Each config vs individual: delta of the seed means and Mann-Whitney U p-value."""
    rows = []
    for (dataset, epochs), grp in per_seed.groupby(["dataset", "epochs"]):
        ref = grp[grp["config"] == "individual"]
        for config in [c for c in CONFIG_ORDER[1:] if c in set(grp["config"])]:
            cur = grp[grp["config"] == config]
            for m in METRICS + [c for c in grp.columns if c.startswith("ap::")]:
                a, b = ref[m].dropna(), cur[m].dropna()
                if len(a) < 2 or len(b) < 2:
                    continue
                p = mannwhitneyu(b, a, alternative="two-sided").pvalue
                rows.append(dict(dataset=dataset, epochs=epochs, config=config, metric=m,
                                 individual=a.mean(), value=b.mean(), delta=b.mean() - a.mean(),
                                 better=(None if m in UNDIRECTED or b.mean() == a.mean()
                                         else (b.mean() < a.mean()) == (m in LOWER_IS_BETTER)),
                                 p_mwu=p))
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--split", type=int, default=0)
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    summaries, per_seed = [], []
    for ds_dir in sorted(d for d in args.root.iterdir() if d.is_dir()):
        dataset = ds_dir.name
        targets = load_binary_targets(dataset)
        mask = pd.read_csv(ds_dir / "imputation_mask.csv", dtype={"variable": str, "sample": str})
        sd = tn.numeric_sd(dataset, args.split, mask)
        summaries.append(dict(dataset=dataset, config="mean_imputation",
                              imp_nrmse_mean=mean_imputation_nrmse(dataset, args.split, mask, sd)))
        for ep_dir in sorted(ds_dir.glob("epochs_*"), key=lambda d: int(d.name[7:])):
            for config in [c for c in CONFIG_ORDER if (ep_dir / c).is_dir()]:
                cdir = ep_dir / config
                seeds = sorted(int(t.name.split("_")[1]) for t in cdir.glob("seed_*_imputed.csv"))
                if len(seeds) < 2:
                    continue
                paths = {s: (cdir / f"seed_{s:02d}_train.csv", cdir / f"seed_{s:02d}_test.csv")
                         for s in seeds}
                imps = {s: pd.read_csv(cdir / f"seed_{s:02d}_imputed.csv",
                                       dtype={"variable": str, "sample": str}) for s in seeds}
                runs = tn.load_runs(paths)
                meta = dict(dataset=dataset, epochs=int(ep_dir.name[7:]), config=config)
                summary, rows = tn.analyze_group(meta, runs, targets, imps, sd)
                for row in rows:
                    row.update(clustering_metrics(runs[row["run"]][0].to_numpy()))
                rows_df = pd.DataFrame(rows)
                for m in METRICS:
                    if m in rows_df:
                        summary[f"{m}_mean"] = float(rows_df[m].mean())
                        summary[f"{m}_sd"] = float(rows_df[m].std(ddof=1))
                summaries.append(summary)
                per_seed += rows
                print(f"{dataset} {ep_dir.name} {config}: {len(seeds)} seeds", flush=True)

    per_seed = pd.DataFrame(per_seed)
    summary = pd.DataFrame(summaries)
    comparison = compare(per_seed)
    per_seed.to_csv(args.root / "per_seed_metrics.csv", index=False)
    summary.to_csv(args.root / "summary.csv", index=False)
    comparison.to_csv(args.root / "comparison.csv", index=False)

    cols = ["dataset", "epochs", "config"] + [f"{m}_mean" for m in METRICS] + [
        "imp_cell_sd", "cka_test", "knn_train"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
        main_metrics = comparison[comparison["metric"].isin(METRICS)]
        print(main_metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
