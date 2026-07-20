"""Overlay POME probe quality and the (label-free) RankMe metric against training
epochs, to visually judge whether RankMe tracks downstream linear-probing quality
and could serve as an unsupervised early-stopping criterion.

Consumes ``output/linear_probing/inductive_epoch_snapshots_results.csv`` (written
by ``scripts/linear_probe_inductive_epoch_snapshots.py``, which now also records
``rankme_train`` / ``rankme_test``). Layout is a dataset x embedding-size grid;
each panel has a dual y-axis:

    left  (solid)  : mean probe average precision over all targets (+/- SEM band)
    right (dashed) : RankMe of the inductive TEST embeddings
    right (dotted) : RankMe of the TRAIN embeddings

Aggregation is over targets x splits x runs for AP, and over splits x runs for
RankMe (which is target-independent). Each panel is annotated with the Spearman
correlation between the per-epoch mean AP and mean test RankMe -- a compact,
quantitative read on how well RankMe tracks probe quality.

POME only (UMAP has no training-epoch axis).

Run from the project root:

    conda run -n torch python scripts/plot_inductive_epoch_ap_vs_rankme.py

Output: output/linear_probing/inductive_epoch_ap_vs_rankme.pdf
"""

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(
    SCRIPT_DIR, "..", "output", "linear_probing",
    "inductive_epoch_snapshots_results.csv")
OUT_PATH = os.path.join(
    SCRIPT_DIR, "..", "output", "linear_probing",
    "inductive_epoch_ap_vs_rankme.pdf")

DATASET_LABELS = {"hancock": "HANCOCK", "luad": "TCGA-LUAD", "mimic": "MIMIC-IV"}

# Colors: AP vs. the two RankMe curves.
AP_COLOR = "#1b9e77"      # green
RK_TEST_COLOR = "#d95f02"  # orange
RK_TRAIN_COLOR = "#7570b3"  # purple

labelfontsize = 15
titlefontsize = 15
ticklabelsize = 12

df = pd.read_csv(RESULTS_PATH)
df = df[df["method"] == "POME"].copy()

datasets = [d for d in ["hancock", "luad", "mimic"] if d in set(df["dataset"])]
dims = sorted(df["dim"].unique())
epochs = sorted(df["epoch"].unique())


def epoch_mean_sem(frame, col):
    """Per-epoch mean and SEM of `col`, reindexed onto the global epoch axis."""
    g = frame.groupby("epoch")[col]
    return g.mean().reindex(epochs), g.sem().reindex(epochs)


# ---------------------------------------------------------------------------
# Figure: rows = datasets, cols = embedding sizes.
# ---------------------------------------------------------------------------
nrows, ncols = len(datasets), len(dims)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows),
                         squeeze=False, layout="constrained")

for r, dataset in enumerate(datasets):
    for c, dim in enumerate(dims):
        ax = axes[r][c]
        sub = df[(df["dataset"] == dataset) & (df["dim"] == dim)]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Left axis: probe average precision (aggregated over targets/splits/runs).
        ap_mean, ap_sem = epoch_mean_sem(sub, "average_precision")
        ax.plot(epochs, ap_mean, color=AP_COLOR, marker="o", linewidth=2,
                zorder=3)
        ax.fill_between(epochs, ap_mean - ap_sem, ap_mean + ap_sem,
                        color=AP_COLOR, alpha=0.2, zorder=1)
        ax.tick_params(axis="y", labelcolor=AP_COLOR, labelsize=ticklabelsize)
        ax.tick_params(axis="x", labelsize=ticklabelsize)
        ax.set_xticks(epochs)

        # Right axis: RankMe (target-independent -> one row per split/run/epoch).
        ax2 = ax.twinx()
        rk = sub.drop_duplicates(["split", "run", "epoch"])
        rk_test_mean, _ = None, None
        for col, style, color in [("rankme_test", "--", RK_TEST_COLOR),
                                  ("rankme_train", ":", RK_TRAIN_COLOR)]:
            m, s = epoch_mean_sem(rk, col)
            if m.notna().any():
                ax2.plot(epochs, m, style, color=color, marker="s",
                         linewidth=2, zorder=2)
                ax2.fill_between(epochs, m - s, m + s, color=color, alpha=0.12)
            if col == "rankme_test":
                rk_test_mean = m
        ax2.tick_params(axis="y", labelcolor=RK_TEST_COLOR,
                        labelsize=ticklabelsize)

        # Spearman correlation between per-epoch mean AP and mean test RankMe.
        rho_txt = ""
        if rk_test_mean is not None and rk_test_mean.notna().sum() >= 2:
            rho = ap_mean.corr(rk_test_mean, method="spearman")
            rho_txt = f"\n$\\rho_{{AP,RankMe}}$ = {rho:.2f}"
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}: dim {dim}{rho_txt}",
                     fontsize=titlefontsize)

        if c == 0:
            ax.set_ylabel("Average precision", color=AP_COLOR,
                          fontsize=labelfontsize)
        if c == ncols - 1:
            ax2.set_ylabel("RankMe (effective rank)", color=RK_TEST_COLOR,
                           fontsize=labelfontsize)
        if r == nrows - 1:
            ax.set_xlabel("Training epochs", fontsize=labelfontsize)

# Shared legend.
handles = [
    mlines.Line2D([], [], color=AP_COLOR, marker="o", linewidth=2,
                  label="Probe average precision"),
    mlines.Line2D([], [], color=RK_TEST_COLOR, marker="s", linestyle="--",
                  linewidth=2, label="RankMe (test / inductive)"),
    mlines.Line2D([], [], color=RK_TRAIN_COLOR, marker="s", linestyle=":",
                  linewidth=2, label="RankMe (train)"),
]
fig.legend(handles=handles, loc="outside upper center", ncol=3,
           fontsize=labelfontsize, frameon=True)

fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved figure to {OUT_PATH}")

# Compact summary: Spearman rho(AP, RankMe_test) per dataset x dim.
print("\n=== Spearman corr between mean probe AP and mean test RankMe "
      "(across epochs) ===")
for dataset in datasets:
    for dim in dims:
        sub = df[(df["dataset"] == dataset) & (df["dim"] == dim)]
        if sub.empty:
            continue
        ap_mean, _ = epoch_mean_sem(sub, "average_precision")
        rk_mean, _ = epoch_mean_sem(sub.drop_duplicates(["split", "run", "epoch"]),
                                    "rankme_test")
        if rk_mean.notna().sum() >= 2:
            rho = ap_mean.corr(rk_mean, method="spearman")
            print(f"  {DATASET_LABELS.get(dataset, dataset):10s} dim {dim:3d}: "
                  f"rho = {rho:+.3f}")
