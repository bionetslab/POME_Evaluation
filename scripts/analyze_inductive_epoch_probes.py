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

Run from the project root (CPU):
    container/run.sh scripts/analyze_inductive_epoch_probes.py

Inputs:  output/inductive_stopping_ablation/{dataset}/dim_{D}/split_{NN}/grid_probes.csv
Outputs (output/inductive_stopping_ablation/):
  epoch_peaks.csv           AP maxima over the epoch grid vs. the 2000-epoch budget
  ap_per_target.pdf/.png    AP vs. epochs per target: every split (thin), mean +- SEM (thick)
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
DIM_COLORS = {32: "#1f77b4", 64: "#d62728"}


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


def plot_targets(probes, out):
    """AP vs epochs per target: every split (thin) and mean +- SEM over splits (thick)."""
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
                ax.plot(a.index, a.to_numpy(), color=col, lw=.4, alpha=.25)
                m, se = a.mean(axis=1), a.sem(axis=1)
                ax.plot(m.index, m, "o-", color=col, ms=3, lw=1.6, label=f"d={dim}")
                ax.fill_between(m.index, m - se, m + se, color=col, alpha=.2, lw=0)
            ax.axvline(BUDGET, color="grey", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_title(f"{dsl}: {t}", fontsize=10)
            if j == 0:
                ax.set_ylabel("probe AP (test split)")
            if i == len(DATASETS) - 1:
                ax.set_xlabel("training epochs")
    handles = axes[0, 0].get_legend_handles_labels()[0] + [plt.Line2D(
        [], [], ls=":", color="grey", label=f"default budget ({BUDGET} epochs)")]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle("Linear-probe AP per target (thin: individual splits; thick: mean ± SEM "
                 "over 10 splits)", fontsize=11)
    fig.tight_layout()
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
    plot_targets(probes, args.root)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(f"== AP maxima over the epoch grid vs {BUDGET} epochs (Wilcoxon over 10 splits) ==")
        print(peaks.round(4).to_string(index=False))
    print(f"\nwrote outputs to {args.root}")


if __name__ == "__main__":
    main()
