"""Linear-probe AP of POME's inductive embeddings vs. training length, per target.

Does downstream generalization to unseen samples degrade (overfit) when POME's inductive mode
trains longer, and is there a reproducible AP peak before the default budget of 2000 epochs that
an early-stopping rule could exploit?

Consumes the grid fits of ``generate_inductive_stopping_ablation.py``: per dataset x dim x split
(the 10 fixed train/test splits of data/train_test_splits; one training seed each) POME is fit for
E epochs, E on a grid of 25..2000. Downstream quality is the linear-probe average precision (AP):
an L2 logistic regression per binary clinical target, fit on the transductive embeddings of the
training split and scored on the inductive (transform()) embeddings of the held-out test split.
Targets (``linear_probe_inductive_embeddings.DATASET_TARGETS``): HANCOCK recurrence, survival
status, RFS event; TCGA-LUAD disease-free, disease-specific survival and progression-free status;
MIMIC-IV aplasia, neutropenic fever.

``epoch_peaks.csv`` compares, per dataset x dim (per target and their mean "(mean)"), the best
fixed epoch count E* (chosen on the mean over splits, i.e. with the test labels -- optimistic)
with the budget E=2000 (paired Wilcoxon over the 10 splits), lists the per-split argmax, and gives
a leave-one-split-out estimate (E* chosen on 9 splits, AP on the 10th) of what a fixed,
dataset-specific epoch count would gain over the budget.

``ap_per_target`` also marks, per target and embedding size, the training length that the best
label-free early-stopping criterion of ``inductive_stopping_criteria_ablation.py`` selects (lowest
regret among the CV-based criteria, i.e. excluding the full and the shorter fixed budget; its
hyperparameters tuned on d32/64 of all three datasets, whether or not it beats the full budget
significantly): the median selected epoch over the 10 splits, drawn as a star on the mean curve.
The rule is label-free, so per dataset, dim and split it selects the same epoch for every target.

Run from the project root (CPU; after inductive_stopping_criteria_ablation.py):
    container/run.sh scripts/analyze_inductive_epoch_probes.py

Inputs:  output/inductive_stopping_ablation/{dataset}/dim_{D}/split_{NN}/grid_probes.csv,
         criteria_ablation/{summary,hyperparameters,selections}.csv
Outputs (output/inductive_stopping_ablation/):
  epoch_peaks.csv           AP maxima over the epoch grid vs. the 2000-epoch budget
  ap_per_target.pdf/.png    AP vs. epochs per target: mean +- SEM over splits, per dim, with the
                            epoch selected by the best early-stopping criterion (POME's default
                            since 1.6); last panel: mean regret (95 % CI) of that rule vs. the
                            smoothed held-out link AUC, the RankMe gap of POME 1.4 (tuned) and
                            the full 2000-epoch budget (criteria_ablation/summary.csv)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "output" / "inductive_stopping_ablation"
DATASETS = {"hancock": "HANCOCK", "luad_noleak": "TCGA-LUAD", "mimic": "MIMIC-IV"}
BUDGET = 2000  # POME's default inductive training budget
DIM_COLORS = {16: "#2ca02c", 32: "#1f77b4", 64: "#d62728"}
# regret panel: POME's default early stopping vs. the comparison rules (summary.csv criteria)
REGRET_BARS = {"val_loss_ema": "POME\n(link loss)",
               "val_auc_ema": "Link AUC",
               "rankme_gap": "RankMe gap",
               "budget": "Max. budget\n(2000 epochs)"}


def load(root):
    files = sorted(root.glob("*/dim_*/split_*/grid_probes.csv"))
    probes = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return probes[probes["epochs"] <= BUDGET]


def epoch_peaks(probes):
    """Is the AP maximum over the epoch grid reproducible across splits? (see module docstring)"""
    def wil(x):
        x = np.asarray(x, float)
        return float(wilcoxon(x).pvalue) if np.abs(x).sum() > 0 else 1.0

    probes = pd.concat([probes.assign(target="(mean)"), probes])  # mean over targets + each
    rows = []
    for (ds, dim, target), g in probes.groupby(["dataset", "dim", "target"]):
        a = g.groupby(["split", "epochs"])["average_precision"].mean().unstack()  # split x E
        m = a.mean()
        best = int(m.idxmax())
        loso = np.array([a.loc[s, a.drop(s).mean().idxmax()] for s in a.index])
        rows.append(dict(dataset=ds, dim=dim, target=target, best_epochs=best,
                         ap_best=m[best], ap_budget=m[BUDGET], delta_best=m[best] - m[BUDGET],
                         p_best_vs_budget=wil(a[best] - a[BUDGET]),
                         ap_loso=loso.mean(), delta_loso=loso.mean() - m[BUDGET],
                         p_loso_vs_budget=wil(loso - a[BUDGET].to_numpy()),
                         split_argmax=" ".join(str(int(e)) for e in a.idxmax(axis=1))))
    return pd.DataFrame(rows)


def best_criterion_epochs(root):
    """(label, {(dataset, dim): median selected epoch}) of the best early-stopping criterion."""
    crit_dir = root / "criteria_ablation"
    summary = pd.read_csv(crit_dir / "summary.csv")
    stop = summary[~summary.criterion.isin(["budget", "fixed_epochs"])]
    best = stop.sort_values("regret").iloc[0]
    hp = pd.read_csv(crit_dir / "hyperparameters.csv")
    param = hp.loc[hp.criterion == best.criterion, "param"].iloc[0]
    sel = pd.read_csv(crit_dir / "selections.csv")
    sel = sel[(sel.criterion == best.criterion) & (sel.param == param)]
    g = sel.groupby(["dataset", "dim"])["epochs"]
    # only (dataset, dim) units with a selection for every split (CV traces may still be running)
    med = g.median()[g.size() == sel.split.nunique()].to_dict()
    return f"{best.label} ({param})", med


def plot_regret(ax, summary):
    """Mean regret (target mean, 95 % bootstrap CI over splits) of the REGRET_BARS criteria."""
    s = summary.set_index("criterion").loc[list(REGRET_BARS)]
    x = np.arange(len(s))
    colors = ["#444444"] + ["#bbbbbb"] * (len(s) - 1)
    ax.bar(x, s.regret, color=colors, width=0.65,
           yerr=[s.regret - s.ci_low, s.ci_high - s.regret], capsize=3, ecolor="black",
           error_kw=dict(lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(REGRET_BARS.values(), fontsize=8)
    ax.set_ylabel("Mean regret (AP)")
    ax.set_title("Early-stopping regret (mean, 95 % CI)", fontsize=10)


def plot_targets(probes, out, stop_epochs, summary):
    """AP vs epochs per target: mean +- SEM over splits, one line per embedding size; the free
    panel shows the regret of the early-stopping rules."""
    targets = {ds: sorted(probes[probes.dataset == ds].target.unique()) for ds in DATASETS}
    ncol = max(len(t) for t in targets.values())
    fig, axes = plt.subplots(len(DATASETS), ncol, figsize=(4 * ncol, 2.9 * len(DATASETS)),
                             squeeze=False)
    for i, (ds, dsl) in enumerate(DATASETS.items()):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(targets[ds]):
                ax.axis("off")
                continue
            t = targets[ds][j]
            g = probes[(probes.dataset == ds) & (probes.target == t)]
            for dim, col in DIM_COLORS.items():
                a = g[g.dim == dim].pivot_table(index="epochs", columns="split",
                                                values="average_precision")
                if a.empty:
                    continue
                m, se = a.mean(axis=1), a.sem(axis=1)
                ax.plot(m.index, m, "o-", color=col, ms=3, lw=1.6, label=f"d={dim}")
                ax.fill_between(m.index, m - se, m + se, color=col, alpha=.2, lw=0)
                e = stop_epochs.get((ds, dim))
                if e is not None:
                    y = np.interp(e, m.index.to_numpy(float), m.to_numpy())
                    # before the first grid fit (25 epochs) the AP is not measured: the star
                    # sits at the AP of the first fit, joined to the curve by a dashed line
                    first = m.index.min()
                    if e < first:
                        ax.plot([e, first], [y, y], "--", color=col, lw=1.2, zorder=4)
                    ax.plot(e, y, "*", color=col, ms=13, mec="black", mew=0.7, zorder=5)
            ax.set_xlim(-60, BUDGET * 1.03)
            ax.set_title(f"{dsl}: {t}", fontsize=10)
            if j == 0:
                ax.set_ylabel("Probe AP (test split)")
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Training epochs")
    handles = axes[0, 0].get_legend_handles_labels()[0] + [
        plt.Line2D([], [], ls="", marker="*", ms=12, color="grey", mec="black", mew=0.7,
                   label="POME early stopping (median selected epoch over splits)")]
    free = [ax for ax in axes.ravel() if not ax.axison]
    if free:
        free[0].axis("on")
        plot_regret(free[0], summary)
    for k, ax in enumerate(ax for ax in axes.ravel() if ax.axison):
        ax.text(-0.13, 1.13, "abcdefghijklmnop"[k], transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(handles), fontsize=9, frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ap_per_target.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()

    probes = load(args.root)
    peaks = epoch_peaks(probes)
    peaks.to_csv(args.root / "epoch_peaks.csv", index=False)
    stop_label, stop_epochs = best_criterion_epochs(args.root)
    summary = pd.read_csv(args.root / "criteria_ablation" / "summary.csv")
    plot_targets(probes, args.root, stop_epochs, summary)
    print("best early-stopping criterion:", stop_label)
    print({k: v for k, v in sorted(stop_epochs.items())})
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(f"== AP maxima over the epoch grid vs {BUDGET} epochs (Wilcoxon over 10 splits) ==")
        print(peaks.round(4).to_string(index=False))
    print(f"\nwrote outputs to {args.root}")


if __name__ == "__main__":
    main()
