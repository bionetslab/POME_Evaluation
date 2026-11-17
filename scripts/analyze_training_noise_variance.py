"""Run-to-run variation of POME's downstream results under LR decay / all-negatives scoring.

Reads the repeated-seed runs of ``generate_training_noise_runs.py`` and, per dataset x epoch
budget x configuration (baseline, lr_decay, all_negatives, both, each with random init,
spectral init, spectral init + table anchoring ``anchor{w}_*``, and spectral init + output
Gram anchor ``gram{mu}_*`` / ``anchor{w}_gram{mu}_*``), quantifies how much the
results differ between seeds. Inductive UMAP (10 runs) and the published inductive POME runs
(5 runs, CV-tuned epochs, constant lr, sampled negatives) on the same split / dim are added as
references where they exist.

Agreement between seeds (mean over all seed pairs; higher = more stable):
  * cka_train / cka_test   linear CKA of the train / test sample embeddings
                           (invariant to rotation and isotropic scaling);
  * knn_train              mean Jaccard overlap of each train sample's 10 nearest neighbours;
  * kmeans_ari             ARI between k-means partitions (k = 5, n_init = 10, fixed seed) of
                           the train embeddings.
Spread across seeds (SD over seeds; lower = more stable):
  * probe_ap_sd            SD of the test average precision of the linear probe (the
                           inductive-probing logreg ``probe_one``, standardized features),
                           averaged over the dataset's binary targets; probe_ap_mean = mean AP;
  * imp_nrmse_sd / imp_acc_sd   SD of the numeric NRMSE (error / variable SD over the observed
                           training values; constant variables excluded) and categorical accuracy of the masked-cell
                           imputations; *_mean = level;
  * imp_cell_sd            per masked numeric cell, SD of the imputed value across seeds in
                           variable-SD units, averaged over cells (seed-to-seed jitter of the
                           individual imputations).
Cluster structure (as in Fig. 3; median over seeds, compare with the ``baseline`` config =
published POME settings):
  * leiden_k / leiden_sil / leiden_db   number of seeded Leiden clusters on the train embedding,
                           their silhouette (higher = better separated; leiden_sil_sd = SD over
                           seeds) and Davies-Bouldin index (lower = better), sqeuclidean;
  * eff_dim                participation ratio of the embedding's PCA spectrum (mean over seeds).

CPU:  container/run.sh scripts/analyze_training_noise_variance.py [--root DIR]
Outputs: {root}/per_seed_metrics{tag}.csv, {root}/variance_summary{tag}.csv
"""

import argparse
import itertools
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe_inductive_embeddings import load_binary_targets, probe_one  # noqa: E402
from generate_unsupervised_figure import davies_bouldin_score_custom, leiden_labels  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "output" / "training_noise"
INDUCTIVE_ROOT = PROJECT_ROOT / "output" / "inductive"
SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
CONFIG_ORDER = ["baseline", "lr_decay", "all_negatives", "both"]
CONFIG_ORDER += [f"spectral_{c}" for c in CONFIG_ORDER]
K_NN, K_MEANS = 10, 5


def config_key(config):
    """Sort anchored configs by (gram weight, table anchor weight, base config)."""
    def weight(name):
        m = re.search(rf"(?:^|_){name}([^_]+)_", config)
        return float(m.group(1)) if m else -1.0
    base = re.sub(r"^(anchor[^_]+_)?(gram[^_]+_)?(k\d+_)?", "", config)
    k = re.search(r"_k(\d+)_", config)
    return (weight("gram"), weight("anchor"), int(k.group(1)) if k else 10**6,
            CONFIG_ORDER.index(base) if base in CONFIG_ORDER else 99)


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2
                 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def knn_sets(X):
    idx = NearestNeighbors(n_neighbors=K_NN + 1).fit(X).kneighbors(X, return_distance=False)
    return [set(row[1:]) for row in idx]


def knn_jaccard(a, b):
    return float(np.mean([len(x & y) / len(x | y) for x, y in zip(a, b)]))


def kmeans_labels(X):
    return KMeans(n_clusters=K_MEANS, n_init=10, random_state=0).fit_predict(X)


def pair_mean(items, fn):
    keys = sorted(items)
    return float(np.mean([fn(items[a], items[b]) for a, b in itertools.combinations(keys, 2)]))


def load_runs(paths):
    """{run_id: (train_df, test_df)} with sample-aligned indices."""
    runs = {}
    for rid, (tr, te) in paths.items():
        train = pd.read_csv(tr, index_col=0)
        test = pd.read_csv(te, index_col=0)
        train.index, test.index = train.index.astype(str), test.index.astype(str)
        runs[rid] = (train, test)
    ref_tr, ref_te = runs[min(runs)][0].index, runs[min(runs)][1].index
    return {k: (tr.loc[ref_tr], te.loc[ref_te]) for k, (tr, te) in runs.items()}


def imputation_metrics(df, sd):
    """Numeric NRMSE and categorical accuracy of one run's masked-cell imputations."""
    num = df[(df["type"] == "cont") & df["variable"].isin(sd.index)]
    cat = df[df["type"] != "cont"]
    z = (num["imputed"] - num["true"]) / num["variable"].map(sd)
    return {"imp_nrmse": float(np.sqrt(np.mean(z ** 2))) if len(num) else np.nan,
            "imp_acc": float(np.mean(np.isclose(cat["imputed"], cat["true"]))) if len(cat) else np.nan}


def numeric_sd(dataset, split, mask):
    """Per-variable SD of the observed training values (masked cells excluded)."""
    df = pd.read_csv(SPLITS_ROOT / dataset / f"split_{split:02d}_train_pome.tsv",
                     sep="\t", index_col=0)
    vals = df.drop(columns=["type"])
    vals.columns = vals.columns.astype(str)
    masked = set(zip(mask["variable"], mask["sample"]))
    out = {}
    for var in df.index[df["type"] == "cont"]:
        row = vals.loc[var]
        keep = [s for s in row.index if row[s] != -99.0 and (var, s) not in masked]
        out[var] = float(row[keep].std())
    out = pd.Series(out)
    return out[out > 0]  # constant variables have no scale to normalize errors by


def analyze_group(meta, runs, targets, imputations=None, sd=None):
    """Return (summary_row, per_seed_rows) for one group of runs."""
    per_seed = []
    for rid, (train, test) in runs.items():
        row = dict(meta, run=rid)
        aps = []
        for label, y in targets.items():
            res = probe_one(train, test, y, standardize=True)
            if res is not None:
                row[f"ap::{label}"] = res["average_precision"]
                aps.append(res["average_precision"])
        row["probe_ap"] = float(np.mean(aps)) if aps else np.nan
        # Fig. 3-style cluster structure of the train embedding (seeded Leiden on a kNN graph,
        # k = 5 % of samples; silhouette and Davies-Bouldin in squared Euclidean distance).
        X = train.to_numpy()
        lab = leiden_labels(X, seed=0)
        row["leiden_k"] = int(lab.max() + 1)
        row["leiden_sil"] = float(silhouette_score(X, lab, metric="sqeuclidean"))
        row["leiden_db"] = float(davies_bouldin_score_custom(X, lab, metric="sqeuclidean"))
        # Effective dimension (participation ratio of the PCA spectrum).
        ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
        row["eff_dim"] = float(ev.sum() ** 2 / (ev ** 2).sum())
        if imputations is not None:
            row.update(imputation_metrics(imputations[rid], sd))
        per_seed.append(row)
    ps = pd.DataFrame(per_seed)

    trains = {k: v[0].to_numpy() for k, v in runs.items()}
    tests = {k: v[1].to_numpy() for k, v in runs.items()}
    knns = {k: knn_sets(X) for k, X in trains.items()}
    kms = {k: kmeans_labels(X) for k, X in trains.items()}
    ap_cols = [c for c in ps.columns if c.startswith("ap::")]
    summary = dict(meta, n_runs=len(runs),
                   cka_train=pair_mean(trains, linear_cka),
                   cka_test=pair_mean(tests, linear_cka),
                   knn_train=pair_mean(knns, knn_jaccard),
                   kmeans_ari=pair_mean(kms, adjusted_rand_score),
                   probe_ap_mean=float(ps["probe_ap"].mean()),
                   probe_ap_sd=float(ps[ap_cols].std(ddof=1).mean()),
                   leiden_k=float(ps["leiden_k"].median()),
                   leiden_sil=float(ps["leiden_sil"].median()),
                   leiden_sil_sd=float(ps["leiden_sil"].std(ddof=1)),
                   leiden_db=float(ps["leiden_db"].median()),
                   eff_dim=float(ps["eff_dim"].mean()))
    if imputations is not None:
        summary.update(imp_nrmse_mean=float(ps["imp_nrmse"].mean()),
                       imp_nrmse_sd=float(ps["imp_nrmse"].std(ddof=1)),
                       imp_acc_mean=float(ps["imp_acc"].mean()),
                       imp_acc_sd=float(ps["imp_acc"].std(ddof=1)))
        num = {k: v[(v["type"] == "cont") & v["variable"].isin(sd.index)].set_index(["variable", "sample"])["imputed"]
               for k, v in imputations.items()}
        mat = pd.concat(num, axis=1)
        var_sd = mat.index.get_level_values("variable").map(sd).to_numpy()
        summary["imp_cell_sd"] = float(np.nanmean(mat.std(axis=1, ddof=1).to_numpy() / var_sd))
    return summary, per_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--epochs", type=int, nargs="+", default=None,
                   help="only these epoch budgets (default: all)")
    p.add_argument("--configs", default=None,
                   help="regex; only configs matching it (default: all)")
    p.add_argument("--datasets", nargs="+", default=None, help="default: all under --root")
    p.add_argument("--tag", default="",
                   help="suffix of the output files, e.g. _topk (variance_summary{tag}.csv)")
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    summaries, per_seed = [], []
    for ds_dir in sorted(d for d in args.root.iterdir() if d.is_dir()):
        dataset = ds_dir.name
        if args.datasets and dataset not in args.datasets:
            continue
        targets = load_binary_targets(dataset)
        mask = pd.read_csv(ds_dir / "imputation_mask.csv", dtype={"variable": str, "sample": str})
        sd = numeric_sd(dataset, args.split, mask)
        for ep_dir in sorted(ds_dir.glob("epochs_*"), key=lambda d: int(d.name[7:])):
            if args.epochs and int(ep_dir.name[7:]) not in args.epochs:
                continue
            present = [d.name for d in ep_dir.iterdir() if d.is_dir()
                       and (args.configs is None or re.search(args.configs, d.name))]
            anchored = sorted((c for c in present if c not in CONFIG_ORDER), key=config_key)
            for config in [c for c in CONFIG_ORDER if c in present] + anchored:
                cdir = ep_dir / config
                trains = sorted(cdir.glob("seed_*_train.csv"))
                if len(trains) < 2:
                    continue
                seeds = [int(t.name.split("_")[1]) for t in trains]
                paths = {s: (cdir / f"seed_{s:02d}_train.csv", cdir / f"seed_{s:02d}_test.csv")
                         for s in seeds}
                imps = {s: pd.read_csv(cdir / f"seed_{s:02d}_imputed.csv",
                                       dtype={"variable": str, "sample": str}) for s in seeds}
                meta = dict(dataset=dataset, method="pome", epochs=int(ep_dir.name[7:]),
                            config=config)
                s, rows = analyze_group(meta, load_runs(paths), targets, imps, sd)
                summaries.append(s)
                per_seed += rows
                print(f"{dataset} {ep_dir.name} {config}: {len(seeds)} seeds", flush=True)
        # References on the same split / dim.
        for method, config in (("umap", "reference"), ("pome", "published_inductive")):
            rdir = INDUCTIVE_ROOT / method / dataset / f"split_{args.split:02d}" / f"dim_{args.dim}"
            trains = sorted(rdir.glob("run_*_train.csv"))
            if len(trains) < 2:
                continue
            paths = {int(t.name.split("_")[1]): (t, t.with_name(t.name.replace("_train", "_test")))
                     for t in trains}
            meta = dict(dataset=dataset, method=method, epochs=np.nan, config=config)
            s, rows = analyze_group(meta, load_runs(paths), targets)
            summaries.append(s)
            per_seed += rows

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.root / f"variance_summary{args.tag}.csv", index=False)
    pd.DataFrame(per_seed).to_csv(args.root / f"per_seed_metrics{args.tag}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
