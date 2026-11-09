"""Cross-seed stability of candidate MIMIC variable representations (manuscript Sec. 2.6 revision).

Reads the per-run representations written by ``generate_variable_representations.py`` and, per
representation option x embedding size, quantifies

  * target profile stability: Spearman correlation of the target-to-lab cosine profiles
    (aplasia / neutropenic fever vs. the 100 lab variables) between all pairs of seeds;
  * lab-lab structure stability: Spearman correlation of the lab-lab cosine matrices (upper
    triangle) between all pairs of seeds;
  * cluster stability: Leiden clustering exactly as in analyze_variable_embeddings.ipynb (cosine
    kNN graph with k = 5% of variables, RBConfiguration, seed 42); adjusted Rand index between
    all pairs of seeds and, per lab variable, the fraction of seeds in which it shares the
    aplasia cluster;
  * target vector norms (small centroid norms -> unstable cosines);
  * optionally, if --importance-dir holds feature_ranks_aplasia.csv, feature_ranks_NF.csv and
    d_labitems.csv: Spearman correlation between CatBoost importance and target-to-lab cosine for
    every run, with a per-run permutation p-value. For contrast vectors the absolute cosine is
    used, since importances are unsigned while the contrast direction is signed.

CPU only:  container/run.sh scripts/analyze_variable_representation_stability.py [--importance-dir DIR]
"""

import argparse
import itertools
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_ROOT = PROJECT_ROOT / "data" / "embeddings" / "MIMIC" / "variable_representations"
OUT_DIR = PROJECT_ROOT / "output" / "variable_embeddings" / "representation_stability"
TARGETS = {"label_aplasia": "Aplasia", "label_nf": "NF"}
OPTIONS = ("raw", "centroid_input", "centroid_latent", "contrast_latent", "numeric_concept")
N_PERM = 1000


def load_runs(dim):
    runs = {}
    for run_dir in sorted((IN_ROOT / f"dim_{dim}").glob("run_*"), key=lambda p: int(p.name[4:])):
        if not (run_dir / "norms.tsv").exists():
            continue
        reps = {o: pd.read_csv(run_dir / f"{o}.tsv", sep="\t", index_col=0) for o in OPTIONS}
        for df in reps.values():
            df.index = df.index.astype(str)
        norms = pd.read_csv(run_dir / "norms.tsv", sep="\t", index_col=0)
        norms.index = norms.index.astype(str)
        runs[int(run_dir.name[4:])] = (reps, norms)
    return runs


def unit(df):
    X = df.to_numpy()
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def leiden_labels(df):
    """Leiden clustering as in analyze_variable_embeddings.ipynb."""
    adj = kneighbors_graph(df.to_numpy(), n_neighbors=int(0.05 * len(df)), mode="connectivity",
                           include_self=False, metric="cosine")
    src, dst = adj.nonzero()
    graph = ig.Graph(n=adj.shape[0], edges=list(zip(src, dst)), directed=False)
    part = la.find_partition(graph, la.RBConfigurationVertexPartition, seed=42)
    return pd.Series(part.membership, index=df.index)


def pairwise(values, fn):
    return [fn(values[a], values[b]) for a, b in itertools.combinations(sorted(values), 2)]


def summarize(xs):
    xs = np.asarray(xs, dtype=float)
    return dict(mean=np.nanmean(xs), sd=np.nanstd(xs), min=np.nanmin(xs), n=len(xs))


RANK_ALIASES = {"Monocytes (%)": "Monocytes"}   # feature-rank name -> d_labitems label


def load_importances(directory, lab_ids):
    """itemid -> rank name for the lab variables, and the per-target importance tables.

    Names that occur for several lab variables (e.g. blood and urine Glucose / pH) cannot be
    assigned to one itemid and are excluded instead of being duplicated.
    """
    path = directory / "d_labitems.csv"
    labels = pd.read_csv(path if path.exists() else path.with_suffix(".csv.gz"))
    name = pd.Series(dict(zip(labels["itemid"].astype(str), labels["label"]))).reindex(lab_ids)
    ambiguous = set(name[name.duplicated(keep=False)])
    imps = {}
    for target, fname in [("label_aplasia", "feature_ranks_aplasia.csv"),
                          ("label_nf", "feature_ranks_NF.csv")]:
        imp = pd.read_csv(directory / fname, index_col=0).rename(index=RANK_ALIASES)
        imps[target] = imp[~imp.index.isin(ambiguous)]
        print(f"{fname}: {imps[target].index.isin(name).sum()} of {len(imp)} features matched "
              f"(excluded ambiguous: {sorted(ambiguous)}; unmatched: "
              f"{sorted(set(imp.index) - set(name) - ambiguous)})")
    label_of = {i: n for i, n in name.items() if n not in ambiguous}
    return label_of, imps


def importance_correlation(sims, imp, label_of, rng):
    """Spearman(imp_mean, similarity) after joining on lab labels, plus permutation p-value."""
    s = pd.Series(sims.to_numpy(), index=sims.index.map(lambda i: label_of.get(i)), name="sim")
    s = s[s.index.notna()]
    joint = imp[["imp_mean"]].join(s, how="inner").dropna()
    rho = spearmanr(joint["imp_mean"], joint["sim"])[0]
    null = np.array([spearmanr(rng.permutation(joint["imp_mean"].to_numpy()), joint["sim"])[0]
                     for _ in range(N_PERM)])
    p = (1 + np.sum(null >= rho)) / (1 + N_PERM)
    return rho, p, len(joint)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--importance-dir", type=Path, default=None)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    importances = None
    rng = np.random.default_rng(0)

    stab_rows, cocluster_rows, norm_rows, imp_rows, profile_rows = [], [], [], [], []
    for dim in args.dims:
        runs = load_runs(dim)
        if len(runs) < 2:
            print(f"[skip] dim {dim}: {len(runs)} completed runs")
            continue
        print(f"dim {dim}: {len(runs)} runs", flush=True)
        if args.importance_dir and importances is None:
            ids = next(iter(runs.values()))[0]["raw"].index
            importances = load_importances(args.importance_dir, [i for i in ids if i not in TARGETS])
        for option in OPTIONS:
            reps = {r: rep[option].dropna() for r, (rep, _) in runs.items()}
            labs = [v for v in next(iter(reps.values())).index if v not in TARGETS]
            base = dict(dim=dim, option=option, n_runs=len(reps))

            # lab-lab structure
            iu = np.triu_indices(len(labs), k=1)
            lab_cos = {r: (unit(df.loc[labs]) @ unit(df.loc[labs]).T)[iu] for r, df in reps.items()}
            stab_rows.append({**base, "metric": "lablab_spearman",
                              **summarize(pairwise(lab_cos, lambda a, b: spearmanr(a, b)[0]))})

            # clustering
            clusters = {r: leiden_labels(df) for r, df in reps.items()}
            stab_rows.append({**base, "metric": "leiden_ari",
                              **summarize(pairwise(clusters, adjusted_rand_score))})
            stab_rows.append({**base, "metric": "leiden_n_clusters",
                              **summarize([c.nunique() for c in clusters.values()])})

            has_targets = all(t in reps[r].index for t in TARGETS for r in reps)
            if not has_targets:
                continue
            for target, tname in TARGETS.items():
                profiles = {}
                for r, df in reps.items():
                    U = pd.DataFrame(unit(df), index=df.index)
                    profiles[r] = pd.Series(U.loc[labs].to_numpy() @ U.loc[target].to_numpy(),
                                            index=labs)
                    profile_rows.append(pd.DataFrame(dict(dim=dim, option=option, run=r,
                                                          target=tname, lab=labs,
                                                          cosine=profiles[r].to_numpy())))
                    if importances is not None:
                        label_of, imps = importances
                        sims = profiles[r].abs() if option == "contrast_latent" else profiles[r]
                        rho, p, n_joint = importance_correlation(sims, imps[target], label_of, rng)
                        imp_rows.append({**base, "target": tname, "run": r, "spearman": rho,
                                         "perm_p": p, "n_features": n_joint})
                stab_rows.append({**base, "target": tname, "metric": "target_profile_spearman",
                                  **summarize(pairwise(profiles, lambda a, b: spearmanr(a, b)[0]))})
                norm_rows += [{**base, "target": tname, "run": r, "norm": runs[r][1].loc[target, option]}
                              for r in runs]
            # co-clustering with aplasia
            co = pd.DataFrame({r: (c.loc[labs] == c.loc["label_aplasia"]) for r, c in clusters.items()})
            cocluster_rows.append(pd.DataFrame(dict(dim=dim, option=option, lab=labs,
                                                    frac_runs_with_aplasia=co.mean(axis=1).to_numpy())))
            stab_rows.append({**base, "target": "Aplasia", "metric": "aplasia_cluster_size",
                              **summarize([(c == c.loc["label_aplasia"]).sum() for c in clusters.values()])})

    stab = pd.DataFrame(stab_rows)
    stab.to_csv(OUT_DIR / "stability_summary.csv", index=False)
    pd.concat(profile_rows).to_csv(OUT_DIR / "target_profiles.csv", index=False)
    pd.concat(cocluster_rows).to_csv(OUT_DIR / "aplasia_coclustering.csv", index=False)
    pd.DataFrame(norm_rows).to_csv(OUT_DIR / "target_norms.csv", index=False)
    if imp_rows:
        pd.DataFrame(imp_rows).to_csv(OUT_DIR / "importance_correlation.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(stab.pivot_table(index=["option", "target"], columns=["metric", "dim"],
                               values="mean", dropna=False).round(3).to_string())
    if imp_rows:
        imp = pd.DataFrame(imp_rows)
        summary = imp.groupby(["target", "option", "dim"]).agg(
            rho_mean=("spearman", "mean"), rho_sd=("spearman", "std"),
            rho_min=("spearman", "min"), rho_max=("spearman", "max"),
            frac_p05=("perm_p", lambda p: (p < 0.05).mean()), n_features=("n_features", "first"))
        summary.to_csv(OUT_DIR / "importance_correlation_summary.csv")
        print(summary.round(3).to_string())
    print(f"Wrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
