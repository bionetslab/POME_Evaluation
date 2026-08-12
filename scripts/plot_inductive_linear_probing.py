"""Plot inductive linear probing results.

Reproduces the layout of scripts/plot_fig3_linear_probing.ipynb (3x3 mosaic:
one aggregated bar panel + eight per-target boxplot panels), using the
inductive linear probing results (embeddings fit on the train split and
transformed onto the held-out test split) instead of that notebook's
transductive k-fold scores.

Only the inductive scores are shown: each embedding size draws two boxes,
POME and UMAP.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(
    SCRIPT_DIR, "..", "output", "linear_probing", "inductive_linear_probing_results.csv"
)
OUT_PATH = os.path.join(
    SCRIPT_DIR, "..", "output", "linear_probing", "inductive_linear_probing_results.pdf"
)

df = pd.read_csv(RESULTS_PATH)

# The results file may also carry the variable-type-restricted runs (mode
# numeric_only / cat_only) that feed plot_inductive_type_combination.py. This
# figure is the all-variables comparison only.
if "mode" in df.columns:
    df = df[df["mode"] == "combined"].copy()

# Use average precision as the score, mirroring the "Average precision" y-axis
# of the original figure.
df = df.rename(columns={"average_precision": "Score", "method": "Method", "target": "Target"})

# "Setting" = method, used as the boxplot hue in the per-target panels
# (POME = Set2 mint-green, UMAP = Set2 orange, matching the paper).
df["Setting"] = df["Method"]
panel_df = df
setting_order = ["POME", "UMAP"]
setting_palette = dict(zip(setting_order, sns.color_palette("Set2")))

# ---------------------------------------------------------------------------
# Panel a: aggregate over targets and embedding sizes. For each (Method, Target,
# dim) take the median score, then the best median per (Method, Target), and
# count how often each method wins.
# ---------------------------------------------------------------------------
def method_win_counts(frame):
    """(#targets POME wins, #targets UMAP wins) by best-across-dim median score."""
    median_scores = frame.groupby(["Method", "Target", "dim"])["Score"].median().reset_index()
    max_scores = median_scores.groupby(["Method", "Target"])["Score"].max().reset_index()
    pivot = max_scores.pivot(index="Target", columns="Method", values="Score")
    return int((pivot["POME"] > pivot["UMAP"]).sum()), int((pivot["UMAP"] > pivot["POME"]).sum())


pome_ind, umap_ind = method_win_counts(df)
agg_df = pd.DataFrame([
    {"Setting": "POME", "count": pome_ind},
    {"Setting": "UMAP", "count": umap_ind},
])

# ---------------------------------------------------------------------------
# Per-target panels. (dataset, target, panel key, title)
# ---------------------------------------------------------------------------
panels = [
    ("mimic", "Aplasia", "b", "MIMIC-IV: Aplasia"),
    ("mimic", "Neutropenic Fever", "c", "MIMIC-IV: Neutropenic fever"),
    ("hancock", "Recurrence", "d", "HANCOCK: Recurrence"),
    ("hancock", "Survival", "e", "HANCOCK: Survival"),
    ("hancock", "RFS Event", "f", "HANCOCK: Recurrence-free \nsurvival event"),
    ("luad", "Disease Free Status", "g", "TCGA-LUAD: Disease-free status"),
    ("luad", "DSS Status", "h", "TCGA-LUAD: Disease-specific survival"),
    ("luad", "Progression Free Status", "i", "TCGA-LUAD: Progression-free status"),
]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
mosaic = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
fig, axes = plt.subplot_mosaic(mosaic, figsize=(16, 12), layout="constrained")

offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)
labelfontsize = 20
titlefontsize = 20
ticklabelsize = 16
legendfontsize = 16

palette = sns.color_palette("Set2")

# Panel a: aggregated bar plot, one bar per method.
counts = dict(zip(agg_df["Setting"], agg_df["count"]))
panel_a_order = ["POME", "UMAP"]
bar_positions = [0.0, 0.95]
bar_width = 0.85

axes["a"].bar(
    bar_positions,
    [counts[s] for s in panel_a_order],
    width=bar_width,
    color=[setting_palette[s] for s in panel_a_order],
)
axes["a"].set_xlabel("")
axes["a"].set_xticks(bar_positions)
axes["a"].set_xticklabels(panel_a_order, fontsize=ticklabelsize)
axes["a"].set_ylabel("Number of targets with \nbest median score", fontsize=labelfontsize)
axes["a"].tick_params(axis="y", labelsize=ticklabelsize)
axes["a"].set_title("Results aggregated over \ntargets and embedding sizes", fontsize=titlefontsize)
axes["a"].text(
    0.0, 1.0, "a", transform=axes["a"].transAxes + offset, fontsize=labelfontsize, fontweight="bold"
)

# Per-target boxplot panels.
legend_handles, legend_labels = None, None
for dataset, target, key, title in panels:
    sub = panel_df[(panel_df["dataset"] == dataset) & (panel_df["Target"] == target)]
    # Positive class ratio (dashed line) = mean test positive ratio for target.
    naive = sub["test_pos_ratio"].mean()

    show_legend = key == "b"
    sns.boxplot(
        data=sub,
        x="dim",
        y="Score",
        hue="Setting",
        hue_order=setting_order,
        ax=axes[key],
        dodge=True,
        gap=0.2,
        legend=show_legend,
        palette=setting_palette,
    )

    axes[key].axhline(
        naive, color="black", linestyle="--", linewidth=2, label="Positive class ratio"
    )

    axes[key].set_xlabel("Embedding size", fontsize=labelfontsize)
    axes[key].set_ylabel("Average precision", fontsize=labelfontsize)
    axes[key].tick_params(axis="x", labelsize=ticklabelsize)
    axes[key].tick_params(axis="y", labelsize=ticklabelsize)
    axes[key].set_title(title, fontsize=titlefontsize)
    axes[key].text(
        0.0,
        1.0,
        key,
        transform=axes[key].transAxes + offset,
        fontsize=labelfontsize,
        fontweight="bold",
    )

    if show_legend:
        legend_handles, legend_labels = axes[key].get_legend_handles_labels()
        axes[key].legend_.remove()

# Shared legend at the top.
legend1 = fig.legend(
    legend_handles,
    legend_labels,
    loc="center",
    bbox_to_anchor=(0.49, 1.02),
    ncol=len(legend_labels),
    fontsize=legendfontsize,
    frameon=True,
    fancybox=False,
)

fig.savefig(OUT_PATH, bbox_inches="tight", bbox_extra_artists=[legend1])
print(f"Saved figure to {OUT_PATH}")
print("Aggregated counts:")
print(agg_df.to_string(index=False))
