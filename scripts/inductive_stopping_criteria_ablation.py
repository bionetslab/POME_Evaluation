"""Ablation of early-stopping criteria for POME's inductive mode, scored by linear-probe regret.

Question (reviewer): does a label-free stopping criterion select a training length with good
*downstream* generalization, and how should its tolerance / patience be chosen? This script
compares the most promising label-free criteria -- including simply training to the full budget
of 2000 epochs -- by their regret in linear probing, aggregated over all individual binary
targets (HANCOCK: 3, TCGA-LUAD: 3, MIMIC-IV: 2).

Data: ``generate_inductive_stopping_ablation.py`` (per dataset x dim in {32, 64} x the 10 fixed
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

Hyperparameter choice (--tuning): by default ONE setting per criterion, chosen to minimize the
target-mean regret over all three datasets (the rule a user would get as POME's default; the
reported regret is in-sample). With --tuning lodo, leave-one-dataset-out instead (each dataset gets
the setting tuned on the two others); summary.csv also lists the other scheme (regret_other).
Uncertainty: 95 % bootstrap CI over splits
(resampled within dataset). Test vs. budget: paired Wilcoxon signed-rank over the 30
dataset x split units (regret difference averaged over targets and dims).

Run from the project root (CPU; needs POME >= the inductive-revision branch):
    POME_REPO=$HOME/Projects/POME-inductive-revision \\
        container/run.sh scripts/inductive_stopping_criteria_ablation.py

Inputs:  output/inductive_stopping_ablation/{dataset}/dim_{D}/split_{NN}/{cv_trace,grid_probes}.csv
Outputs: output/inductive_stopping_ablation/criteria_ablation/
  selections.csv        selected epoch per criterion x hyperparameters x dataset x dim x split
  regret_per_target.csv regret per criterion x hyperparameters x dataset x dim x split x target
  hyperparameters.csv   LODO / in-sample hyperparameters per criterion
  summary.csv           per criterion: regret (primary scheme; regret_other: other scheme), CI, per dataset, p vs budget
  figure_stopping_criteria.pdf/.png  manuscript figure
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "output" / "inductive_stopping_ablation"
DATASETS = {"hancock": "HANCOCK", "luad": "TCGA-LUAD", "mimic": "MIMIC-IV"}
DATASET_COLORS = {"hancock": "#2a78d6", "luad": "#eb6834", "mimic": "#1baf7a"}
BUDGET = 2000
EVAL_EVERY = 10
KEY = ["dataset", "dim", "split"]
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


SCHEME_LABELS = {"insample": "one setting tuned on all three datasets",
                 "lodo": "tuned leave-one-dataset-out"}


def lodo_regret(reg):
    """Rows of `reg` with each dataset's hyperparameters chosen on the other datasets."""
    parts, hp = [], []
    for held in DATASETS:
        best = choose(reg, [d for d in DATASETS if d != held])
        for crit, param in best.items():
            parts.append(reg[(reg.dataset == held) & (reg.criterion == crit) & (reg.param == param)])
            hp.append(dict(criterion=crit, scheme="lodo", held_out=held, param=param))
    best = choose(reg, list(DATASETS))
    insample = pd.concat([reg[(reg.criterion == c) & (reg.param == p)] for c, p in best.items()])
    hp += [dict(criterion=c, scheme="in-sample", held_out="-", param=p) for c, p in best.items()]
    return pd.concat(parts, ignore_index=True), insample, pd.DataFrame(hp)


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


def summarize(primary, other):
    """Per criterion: regret under the primary tuning scheme (CI, p vs budget, per dataset) and
    the target-mean regret under the other scheme (`regret_other`)."""
    lodo, insample = primary, other
    budget = lodo[lodo.criterion == "budget"].groupby(["dataset", "split"])["regret"].mean()
    rows = []
    for crit in LABELS:
        d = lodo[lodo.criterion == crit]
        if d.empty:
            continue
        lo, hi = bootstrap_ci(d)
        diff = d.groupby(["dataset", "split"])["regret"].mean() - budget
        p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else np.nan
        row = dict(criterion=crit, label=LABELS[crit], regret=target_mean(d), ci_low=lo,
                   ci_high=hi, regret_other=target_mean(insample[insample.criterion == crit]),
                   ap=target_mean(d, "ap"), delta_vs_budget=diff.mean(), p_vs_budget=p,
                   better_splits=int((diff < 0).sum()), worse_splits=int((diff > 0).sum()))
        for ds in DATASETS:
            dd = d[d.dataset == ds]
            row[f"regret_{ds}"] = target_mean(dd)
            row[f"epochs_median_{ds}"] = dd.groupby(KEY)["epochs"].first().median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("regret").reset_index(drop=True)


def plot(summary, lodo, out, tuning):
    s = summary.iloc[::-1].reset_index(drop=True)  # best on top
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.32 * len(s) + 1.3), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1.1], wspace=0.12))
    xmax = 1.1 * max(s.ci_high.max(), *(s[f"regret_{d}"].max() for d in DATASETS))
    xticks = np.arange(0, xmax - 0.01, 0.02)  # no label at the shared panel edge
    ink, muted, grid = "#1f1f1e", "#8a8986", "#e6e5e0"
    for ax in axes:
        ax.grid(axis="x", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7)
    hl = s.criterion == "budget"

    # (a) overall regret under the chosen tuning scheme (95% CI)
    ax = axes[0]
    for yi, r in zip(y, s.itertuples()):
        col = ink if r.criterion == "budget" else muted
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.regret, yi, "o", color=col, ms=6, mec="white", mew=1, zorder=3)
        if np.isfinite(r.p_vs_budget):
            note = f"p={r.p_vs_budget:.2g}" if r.p_vs_budget >= 1e-3 else "p<0.001"
        elif r.criterion != "budget":
            note = "never fires: = full budget" if r.criterion == "rankme_gap" else "= full budget"
        else:
            note = None
        if note:
            ax.text(0.98, yi - 0.42, note, transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=5.5, color=muted)
    ax.set_yticks(y, s.label, fontsize=7)
    for lab, h in zip(ax.get_yticklabels(), hl):
        lab.set_fontweight("bold" if h else "normal")
    ax.set_xlabel("linear-probe regret (AP)\nmean over 8 targets", fontsize=7)
    ax.set_xlim(0, None)
    ax.set_title("a  regret vs. best epoch", fontsize=8, loc="left", fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)

    # (b) per-dataset regret
    ax = axes[1]
    off = {"hancock": -0.22, "luad": 0.0, "mimic": 0.22}
    for ds, dsl in DATASETS.items():
        ax.plot(s[f"regret_{ds}"], y + off[ds], "o", color=DATASET_COLORS[ds], ms=4.5,
                mec="white", mew=0.6, label=dsl)
    ax.set_xlabel("linear-probe regret (AP)\nmean over the dataset's targets", fontsize=7)
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.set_title("b  per dataset", fontsize=8, loc="left", fontweight="bold")

    # (c) selected training length
    ax = axes[2]
    for ds, dsl in DATASETS.items():
        for yi, crit in zip(y, s.criterion):
            e = lodo[(lodo.criterion == crit) & (lodo.dataset == ds)].groupby(KEY)["epochs"].first()
            jitter = np.random.default_rng(0).uniform(-0.07, 0.07, len(e))
            ax.plot(e.to_numpy() * np.exp(jitter), np.full(len(e), yi + off[ds]), "|",
                    color=DATASET_COLORS[ds], ms=4, alpha=0.6, mew=0.8)
            ax.plot(e.median(), yi + off[ds], "o", color=DATASET_COLORS[ds], ms=4,
                    mec="white", mew=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(8, 2600)
    ax.set_xlabel("selected training epochs\n(ticks: splits x dims; dot: median)", fontsize=7)
    ax.set_title("c  selected epochs", fontsize=8, loc="left", fontweight="bold")
    for ax in axes:
        for yi, h in zip(y, hl):
            if h:
                ax.axhspan(yi - 0.45, yi + 0.45, color="#f2f1ec", zorder=0, lw=0)
    handles = [
        plt.Line2D([], [], marker="o", color=muted, lw=2, ms=5, mec="white",
                   label=SCHEME_LABELS[tuning] + " (95% CI)"),
        *[plt.Line2D([], [], marker="o", ls="", color=DATASET_COLORS[d], ms=4.5, mec="white",
                     label=dl) for d, dl in DATASETS.items()]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5,
               fontsize=6, frameon=False, handletextpad=0.3, columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_stopping_criteria.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--tuning", choices=["insample", "lodo"], default="insample",
                   help="primary hyperparameter choice: one setting per criterion tuned on all "
                        "datasets (default), or leave-one-dataset-out (written to .../lodo/)")
    args = p.parse_args()
    out = args.root / "criteria_ablation"
    if args.tuning == "lodo":
        out = out / "lodo"
    out.mkdir(parents=True, exist_ok=True)

    trace, curves = load(args.root)
    grid_epochs = sorted({int(e) for _, g, _ in curves.values() for e in g})
    sel = select_all(trace, grid_epochs)
    sel.to_csv(out / "selections.csv", index=False)
    reg = regrets(sel, curves)
    reg.to_csv(out / "regret_per_target.csv", index=False)
    lodo, insample, hp = lodo_regret(reg)
    primary, other = (insample, lodo) if args.tuning == "insample" else (lodo, insample)
    hp.to_csv(out / "hyperparameters.csv", index=False)
    summary = summarize(primary, other)
    summary.to_csv(out / "summary.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"== Stopping criteria: linear-probe regret, mean over 8 targets ({args.tuning}) ==")
    print(summary.drop(columns="label").round(4).to_string(index=False))
    print("\n== Hyperparameters ==")
    print(hp.pivot_table(index="criterion", columns=["scheme", "held_out"], values="param",
                         aggfunc="first").to_string())
    plot(summary, primary, out, args.tuning)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
