"""Numeric imputation with the current vs. an invariant regression-head input (revision test).

POME's numeric imputation head predicts the standardized value of (sample i, variable j) from

    raw        [ 2 e_i , e_j ]           current POME: raw pre-encoder sample encoding and raw
                                         numeric variable encoding (not invariant: the model only
                                         sees normalize(2 e_i), so each e_i has an arbitrary scale)
    invariant  [ h_i , c_j ]             h_i = e_i / ||e_i|| (the sample node's model input) and
                                         c_j = numeric concept encoding (e_j centred over numeric
                                         variables, scale-normalized), both invariant to POME's
                                         parameter symmetries

For every masked benchmark file the Embedder is fitted ONCE; both heads are then trained on the
same frozen model with the same seed, architecture and hyperparameters as POME's own head
(``Embedder._fit_value_regressor``), so the only difference is the input features. The head
POME fitted itself during ``fit()`` is scored as well (``package``, reference for ``raw``).

Metric: per-variable range-normalized MAE on the masked continuous entries (``nmae_cont``),
computed exactly as in generate_imputation_results.py.

GPU (resumable; finished files are skipped):
    sbatch --time=12:00:00 container/submit.sh scripts/compare_imputation_regressor_features.py
Plot / summary only (CPU):
    container/run.sh scripts/compare_imputation_regressor_features.py --plot-only
Output: output/imputation_regressor_features/{results.csv, summary.csv, nmae_difference.{pdf,png}}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_imputation_results import _load_variable_range, _macro_nmae  # noqa: E402

POME_ROOT = PROJECT_ROOT / "data" / "imputation_data" / "pome_based"
GT_ROOT = PROJECT_ROOT / "data" / "imputation_groundtruth"
OUT_DIR = PROJECT_ROOT / "output" / "imputation_regressor_features"
RESULTS = OUT_DIR / "results.csv"
NA_ENCODING = {"HANCOCK": -99999.0, "TCGA_LUAD": -99.0, "MIMIC": -99.0}
VARIANTS = ("package", "raw", "invariant")
HEAD_SEED = 0


# ----------------------------------------------------------------------------- features
def head_features(embedder, variant):
    """(sample feature matrix indexed like _sample_node_dict, {numeric var: feature vector})."""
    import torch
    model = embedder.model
    W = model.node_embeddings.weight.detach().cpu()
    sample_rows = torch.tensor(list(embedder._sample_node_dict.values()))
    sample_raw = 2 * W[sample_rows]                       # e_i + e_i, as in POME's head
    if variant == "raw":
        var_raw = embedder._variable_embeddings.detach().cpu()
        var_feat = {v: var_raw[k] for k, v in enumerate(embedder._variable_names)
                    if v in embedder._cont_vars}
        return sample_raw, var_feat
    if variant == "invariant":
        names, concepts, _ = embedder._numeric_concepts_and_bins()
        var_feat = {v: torch.tensor(concepts[k], dtype=W.dtype) for k, v in enumerate(names)}
        return sample_raw / sample_raw.norm(dim=1, keepdim=True), var_feat
    raise ValueError(variant)


def fit_head(embedder, variant):
    """Re-implementation of Embedder._fit_value_regressor with exchangeable input features."""
    import torch
    import torch.nn.functional as F
    from pome.models import ValueRegressor

    device = next(embedder.model.parameters()).device
    sample_feat, var_feat = head_features(embedder, variant)
    sample_pos = {s: k for k, s in enumerate(embedder._sample_node_dict)}
    stats, xs, vs, ys = {}, [], [], []
    for var in sorted(embedder._cont_vars):
        if not embedder._var_value_dict[var] or var not in var_feat:
            continue
        row = embedder._X.loc[var]
        obs = [s for s in row.index[embedder._observed_continuous_mask(row)] if s in sample_pos]
        if not obs:
            continue
        mean, std = embedder._reg_target_standardization(var)
        stats[var] = (mean, std)
        xs.append(sample_feat[[sample_pos[s] for s in obs]])
        vs.append(var_feat[var].unsqueeze(0).expand(len(obs), -1))
        ys.append(torch.tensor([(float(row[s]) - mean) / std for s in obs], dtype=sample_feat.dtype))
    X, V, y = (torch.cat(t).to(device) for t in (xs, vs, ys))

    torch.manual_seed(HEAD_SEED)
    head = ValueRegressor(embedder.embedding_dimension).to(device)
    lr = embedder.regressor_lr if embedder.regressor_lr is not None else embedder.lr
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=embedder.regressor_weight_decay)
    head.train()
    for _ in range(embedder.regressor_epochs):
        opt.zero_grad()
        F.l1_loss(head(X, V), y).backward()
        opt.step()
    head.eval().cpu()

    def predict(sample, var):
        if var not in stats:
            return embedder._observed_mean(var)
        with torch.no_grad():
            z = head(sample_feat[sample_pos[sample]].unsqueeze(0), var_feat[var].unsqueeze(0))
        mean, std = stats[var]
        return z.item() * std + mean
    return predict


def package_predictor(embedder):
    def predict(sample, var):
        if hasattr(embedder, "_value_regressor") and var in embedder._reg_target_stats:
            return embedder._regress_value(var, sample)
        return embedder._observed_mean(var)
    return predict


# ----------------------------------------------------------------------------- benchmark
def score(dataset, gt_path, predictors):
    import pickle
    gt = pickle.load(open(gt_path, "rb"))
    cont = [(s, v, val) for (s, v), (val, kind) in gt.items() if kind == "cont"]
    truth = np.array([c[2] for c in cont], dtype=float)
    variables = np.array([c[1] for c in cont])
    scales = _load_variable_range(dataset)
    return {name: _macro_nmae(truth, np.array([p(s, v) for s, v, _ in cont], dtype=float),
                              variables, scales)
            for name, p in predictors.items()}


def run(args):
    import torch
    from pome import Embedder, make_deterministic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    done = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(
        columns=["dataset", "dim", "na_ratio", "run"])
    done_keys = set(map(tuple, done[["dataset", "dim", "na_ratio", "run"]].astype(str).values))
    for dataset in args.datasets:
        for dim in args.dims:
            for ratio in args.na_ratios:
                for r in args.runs:
                    key = (dataset, str(dim), str(ratio), str(r))
                    masked = POME_ROOT / dataset / "simulated_data" / f"masked_values_{ratio}_{float(r)}.tsv"
                    gt_path = GT_ROOT / dataset / f"masked_values_{ratio}_{r}.pkl"
                    if key in done_keys or not masked.exists() or not gt_path.exists():
                        continue
                    print(f"[fit] {dataset} dim {dim} ratio {ratio} run {r}", flush=True)
                    make_deterministic(42)                 # as in impute_graph_based.py
                    embedder = Embedder(na_encoding=NA_ENCODING[dataset], informative_nas=[],
                                        device=device, epochs=args.epochs,
                                        bins_per_continuous=args.bins,
                                        discretization_type=args.discretization,
                                        enable_imputation=True, embedding_dimension=dim)
                    embedder.fit(pd.read_csv(masked, sep="\t", index_col=0))
                    predictors = {"package": package_predictor(embedder),
                                  "raw": fit_head(embedder, "raw"),
                                  "invariant": fit_head(embedder, "invariant")}
                    res = score(dataset, gt_path, predictors)
                    row = pd.DataFrame([{"dataset": dataset, "dim": dim, "na_ratio": ratio,
                                         "run": r, **{f"nmae_{k}": v for k, v in res.items()}}])
                    row.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
                    print("      " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)


# ----------------------------------------------------------------------------- summary/plot
def summarize_and_plot():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import wilcoxon

    df = pd.read_csv(RESULTS)
    df["diff"] = df["nmae_invariant"] - df["nmae_raw"]
    df["rel_diff_pct"] = 100 * df["diff"] / df["nmae_raw"]
    rows = []
    for (dataset, dim), g in df.groupby(["dataset", "dim"]):
        p = wilcoxon(g["nmae_invariant"], g["nmae_raw"]).pvalue if len(g) >= 6 else np.nan
        rows.append(dict(dataset=dataset, dim=dim, n=len(g),
                         nmae_package=g["nmae_package"].mean(), nmae_raw=g["nmae_raw"].mean(),
                         nmae_invariant=g["nmae_invariant"].mean(),
                         median_rel_diff_pct=g["rel_diff_pct"].median(),
                         frac_invariant_better=(g["diff"] < 0).mean(), wilcoxon_p=p))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    sns.boxplot(data=df, x="dim", y="rel_diff_pct", hue="dataset", gap=0.2, fliersize=3, ax=ax,
                palette=sns.color_palette("tab10", df["dataset"].nunique()))
    ax.axhline(0, color="0.5", lw=1, ls="--", zorder=0)
    ax.set_xlabel("Encoding size", fontsize=13)
    ax.set_ylabel("Relative change of nMAE (%)\ninvariant vs. current head input", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title=None, frameon=False, fontsize=11)
    fig.savefig(OUT_DIR / "nmae_difference.pdf")
    fig.savefig(OUT_DIR / "nmae_difference.png", dpi=150)
    print(f"Wrote {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(NA_ENCODING), choices=list(NA_ENCODING))
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--na-ratios", nargs="+", default=["0.1", "0.25"])
    parser.add_argument("--runs", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--discretization", default="z", choices=["z", "nonlinear"])
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        run(args)
    summarize_and_plot()


if __name__ == "__main__":
    main()
