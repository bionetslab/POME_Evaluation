"""Ablation of early-stopping criteria for POME's inductive mode, scored by linear-probe regret.

Question (reviewer): does a label-free stopping criterion select a training length with good
*downstream* generalization, and how should its tolerance / patience be chosen? This script
compares the most promising label-free criteria -- including simply training to the full budget
of 2000 epochs -- by their regret in linear probing, aggregated over all individual binary
targets (HANCOCK: 3, TCGA-LUAD: 3, MIMIC-IV: 2).

Data: ``generate_inductive_stopping_ablation.py`` (per dataset x dim in {16, 32, 64} x the 10 fixed
train/test splits of the linear-probing experiment):
  cv_trace.csv     label-free signals of POME's inductive sample-holdout CV (3 folds inside the
                   training split, checked every 10 epochs) over 2000 epochs, without stopping.
  grid_probes.csv  linear-probe AP per target of full fits trained for E epochs (E on a grid of
                   25..2000; cosine lr over E), probe fit on the training split, scored on the
                   inductively embedded test split.

Criteria (each applied online to the fold-mean CV signal with ``pome.utils.StoppingRule``, i.e.
the code ``Embedder.fit(inductive=True)`` runs; smoothing is a causal exponential moving average,
so every rule only looks at past checks and selects exactly the epoch it would select during
training):
  budget           train the full 2000 epochs (= POME 1.4: its RankMe rule never fires)
  fixed_epochs     a shorter fixed training length (hyperparameter: E)
  rankme_gap       sample-matched train-vs-held-out RankMe gap, "onset" rule (POME 1.4's criterion)
  rankme_val       RankMe (effective rank) of the held-out embeddings, maximal ("best" rule; the
                   original use of RankMe for model selection)
  val_loss         held-out link-prediction BCE, minimal ("best" rule = classical early stopping)
  val_loss_ema     val_loss after exponential smoothing
  val_auc          held-out link-prediction ROC-AUC, maximal ("best" rule)
  val_auc_ema      val_auc after exponential smoothing
Hyperparameters: tolerance (absolute, or relative for val_loss / rankme_val) x patience (checks of
10 epochs) [x smoothing factor].

Regret: per dataset x dim x split x target, regret = max_E AP(E) - AP(E_sel), the AP lost vs. the
best grid epoch of that split and target (an oracle that uses the test labels; AP(E_sel) is
interpolated linearly in log E between grid fits). Aggregate = mean over the 8 targets (each
target weighted equally; per target the mean over dims and splits).

Hyperparameter choice: ONE setting per criterion, chosen to minimize the target-mean regret over
all three datasets (the rule a user would get as POME's default; the reported regret is
in-sample). Uncertainty: 95 % bootstrap CI over splits
(resampled within dataset). Test vs. budget: paired Wilcoxon signed-rank over the 30
dataset x split units (regret difference averaged over targets and dims).

Run from the project root (CPU; needs POME >= the inductive-revision branch):
    POME_REPO=$HOME/Projects/POME-inductive-revision \\
        container/run.sh scripts/inductive_stopping_criteria_ablation.py

Inputs:  output/inductive_stopping_ablation/{dataset}/dim_{D}/split_{NN}/{cv_trace,grid_probes}.csv
Outputs: output/inductive_stopping_ablation/criteria_ablation/
  selections.csv        selected epoch per criterion x hyperparameters x dataset x dim x split
                        (all dims with a CV trace)
  regret_per_target.csv regret per criterion x hyperparameters x dataset x dim x split x target
                        (box plot in panel a of analyze_inductive_epoch_probes.py)
  hyperparameters.csv   chosen hyperparameters per criterion
  summary.csv           per criterion: regret, CI, per dataset, p vs budget
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "output" / "inductive_stopping_ablation"
DATASETS = {"hancock": "HANCOCK", "luad_noleak": "TCGA-LUAD", "mimic": "MIMIC-IV"}
BUDGET = 2000
EVAL_EVERY = 10
KEY = ["dataset", "dim", "split"]
# embedding sizes the criteria are compared and tuned on (all sizes with a CV trace)
TUNING_DIMS = (16, 32, 64)
PATIENCES = [1, 2, 3, 5, 10, 20]
EMA = [0.1, 0.3]  # weight of the newest check

# name: label, signal column, orientation (+1: lower is better), rule mode, relative tol,
# tolerance grid, smoothing grid (None = raw signal)
CRITERIA = {
    "rankme_gap": dict(label="RankMe gap (onset)", signal="overfit_index", sign=1, mode="onset",
                       relative=False, tols=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.05], ema=[None]),
    "rankme_val": dict(label="held-out RankMe (max.)", signal="rankme_val", sign=-1, mode="best",
                       relative=True, tols=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02], ema=[None]),
    "val_loss": dict(label="held-out link loss (min.)", signal="val_bce", sign=1, mode="best",
                     relative=True, tols=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02], ema=[None]),
    "val_loss_ema": dict(label="held-out link loss, smoothed", signal="val_bce", sign=1,
                         mode="best", relative=True, tols=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02],
                         ema=EMA),
    "val_auc": dict(label="held-out link AUC (max.)", signal="val_auc", sign=-1, mode="best",
                    relative=False, tols=[0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01], ema=[None]),
    "val_auc_ema": dict(label="held-out link AUC, smoothed", signal="val_auc", sign=-1,
                        mode="best", relative=False, tols=[0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01],
                        ema=EMA),
}
LABELS = {"budget": f"full budget ({BUDGET} epochs)", "fixed_epochs": "shorter fixed budget",
          **{k: c["label"] for k, c in CRITERIA.items()}}


def ema(x, alpha):
    out, s = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        s = alpha * v + (1 - alpha) * s
        out[i] = s
    return out


def load(root):
    files = sorted(root.glob("*/dim_*/split_*/cv_trace.csv"))
    traces = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    signals = sorted({c["signal"] for c in CRITERIA.values()})
    trace = traces.groupby(KEY + ["epoch"])[signals].mean().reset_index()  # fold mean
    probes = pd.concat([pd.read_csv(f) for f in sorted(root.glob("*/dim_*/split_*/grid_probes.csv"))],
                       ignore_index=True)
    # only units with a CV trace (grid fits without one, e.g. d16, feed analyze_inductive_epoch_probes)
    units = traces[KEY].drop_duplicates()
    probes = probes.merge(units, on=KEY)
    # AP curve per unit x target: {key: (targets, grid epochs, AP matrix targets x grid)}
    curves = {}
    for k, g in probes.groupby(KEY):
        a = g.pivot_table(index="target", columns="epochs", values="average_precision")
        curves[k] = (a.index.to_list(), a.columns.to_numpy(float), a.to_numpy())
    return trace, curves


def select_all(trace, grid_epochs):
    """Selected epoch for every criterion x hyperparameter setting x unit."""
    from pome.utils import select_stopping_epoch
    rows = []
    for k, tr in trace.groupby(KEY):
        tr = tr.sort_values("epoch")
        epochs = tr["epoch"].to_numpy()
        base = dict(zip(KEY, k))
        rows.append(dict(base, criterion="budget", param="-", epochs=BUDGET))
        for e in grid_epochs:
            if e != BUDGET:
                rows.append(dict(base, criterion="fixed_epochs", param=f"E={e}", epochs=e))
        for name, c in CRITERIA.items():
            raw = c["sign"] * tr[c["signal"]].to_numpy(float)
            for alpha in c["ema"]:
                vals = raw if alpha is None else ema(raw, alpha)
                for tol, pat in itertools.product(c["tols"], PATIENCES):
                    e = select_stopping_epoch(epochs, vals, BUDGET, mode=c["mode"], tol=tol,
                                              patience=pat, relative=c["relative"])
                    param = f"tol={tol:g},pat={pat}" + ("" if alpha is None else f",ema={alpha:g}")
                    rows.append(dict(base, criterion=name, param=param,
                                     epochs=max(int(e), EVAL_EVERY)))
    return pd.DataFrame(rows)


def regrets(sel, curves):
    """Per-target regret of every selection (AP interpolated in log epochs)."""
    rows = []
    for k, g in sel.groupby(KEY):
        if k not in curves:
            continue
        targets, grid, ap = curves[k]
        lg = np.log(grid)
        oracle = ap.max(axis=1)
        for r in g.itertuples():
            ap_sel = np.array([np.interp(np.log(r.epochs), lg, row) for row in ap])
            for t, o, a in zip(targets, oracle, ap_sel):
                rows.append((r.criterion, r.param, *k, r.epochs, t, a, o - a))
    return pd.DataFrame(rows, columns=["criterion", "param", *KEY, "epochs", "target", "ap",
                                       "regret"])


def target_mean(df, value="regret"):
    """Mean over the targets (each target: mean over dims and splits)."""
    return df.groupby("target")[value].mean().mean()


def choose(reg, datasets):
    """Hyperparameters per criterion minimizing the target-mean regret on `datasets`."""
    sub = reg[reg.dataset.isin(datasets)]
    per = sub.groupby(["criterion", "param", "target"])["regret"].mean() \
        .groupby(["criterion", "param"]).mean()
    return per.groupby("criterion").idxmin().map(lambda t: t[1])


def tuned_regret(reg):
    """Rows of `reg` with each criterion's hyperparameters chosen on all datasets."""
    best = choose(reg, list(DATASETS))
    tuned = pd.concat([reg[(reg.criterion == c) & (reg.param == p)] for c, p in best.items()],
                      ignore_index=True)
    hp = pd.DataFrame([dict(criterion=c, param=p) for c, p in best.items()])
    return tuned, hp


def bootstrap_ci(df, n=2000, seed=0):
    """95 % CI of the target-mean regret, resampling splits within each dataset."""
    rng = np.random.default_rng(seed)
    unit = df.groupby(["dataset", "split", "target"])["regret"].mean().unstack("target")
    splits = {d: unit.loc[d] for d in unit.index.get_level_values(0).unique()}
    stats = []
    for _ in range(n):
        cols = []
        for d, u in splits.items():
            idx = rng.integers(0, len(u), len(u))
            cols.append(u.iloc[idx].mean(axis=0).dropna())
        stats.append(pd.concat(cols).mean())
    return np.percentile(stats, [2.5, 97.5])


def summarize(tuned):
    """Per criterion: regret (95 % CI, p vs budget, per dataset)."""
    budget = tuned[tuned.criterion == "budget"].groupby(["dataset", "split"])["regret"].mean()
    rows = []
    for crit in LABELS:
        d = tuned[tuned.criterion == crit]
        if d.empty:
            continue
        lo, hi = bootstrap_ci(d)
        diff = d.groupby(["dataset", "split"])["regret"].mean() - budget
        p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else np.nan
        row = dict(criterion=crit, label=LABELS[crit], regret=target_mean(d), ci_low=lo,
                   ci_high=hi,
                   ap=target_mean(d, "ap"), delta_vs_budget=diff.mean(), p_vs_budget=p,
                   better_splits=int((diff < 0).sum()), worse_splits=int((diff > 0).sum()))
        for ds in DATASETS:
            dd = d[d.dataset == ds]
            row[f"regret_{ds}"] = target_mean(dd)
            row[f"epochs_median_{ds}"] = dd.groupby(KEY)["epochs"].first().median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("regret").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    out = args.root / "criteria_ablation"
    out.mkdir(parents=True, exist_ok=True)

    trace, curves = load(args.root)
    grid_epochs = sorted({int(e) for _, g, _ in curves.values() for e in g})
    sel = select_all(trace, grid_epochs)
    sel.to_csv(out / "selections.csv", index=False)
    reg = regrets(sel[sel.dim.isin(TUNING_DIMS)], curves)
    reg.to_csv(out / "regret_per_target.csv", index=False)
    tuned, hp = tuned_regret(reg)
    hp.to_csv(out / "hyperparameters.csv", index=False)
    summary = summarize(tuned)
    summary.to_csv(out / "summary.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("== Stopping criteria: linear-probe regret, mean over 8 targets ==")
    print(summary.drop(columns="label").round(4).to_string(index=False))
    print("\n== Hyperparameters ==")
    print(hp.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
