"""Plot inductive linear probing results.

Reproduces the layout of scripts/plot_fig3_linear_probing.ipynb (3x3 mosaic:
one aggregated bar panel + eight per-target boxplot panels), using the
inductive linear probing results.

In addition to the inductive scores (embeddings fit on the train split and
transformed onto the test split), each per-target panel overlays the
*full-dataset* scores produced by
``scripts/linear_probe_full_dataset_embeddings.py`` -- embeddings trained on the
whole dataset, probed on the *same* train/test splits. Each embedding size
therefore shows up to four boxes: POME / UMAP x inductive / full. Only the
embedding sizes present in *both* analyses are drawn so the comparison is
matched; if the full-dataset results file is missing, the plot falls back to the
inductive-only layout.
"""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import seaborn as sns


def lighten(color, amount):
    """Blend ``color`` toward white by ``amount`` in [0, 1] (0 = unchanged)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(
    SCRIPT_DIR, "..", "output", "linear_probing", "inductive_linear_probing_results.csv"
)
FULL_RESULTS_PATH = os.path.join(
    SCRIPT_DIR, "..", "output", "linear_probing", "full_dataset_linear_probing_results.csv"
)
OUT_PATH = os.path.join(
    SCRIPT_DIR, "..", "output", "linear_probing", "inductive_linear_probing_results.pdf"
)

df = pd.read_csv(RESULTS_PATH)

# Use average precision as the score, mirroring the "Average precision" y-axis
# of the original figure.
df = df.rename(columns={"average_precision": "Score", "method": "Method", "target": "Target"})

# Full-dataset probing results (optional). Same schema as the inductive results;
# tag each source so inductive and full boxes can share a single hue dimension.
full_df = None
if os.path.exists(FULL_RESULTS_PATH):
    full_df = pd.read_csv(FULL_RESULTS_PATH).rename(
        columns={"average_precision": "Score", "method": "Method", "target": "Target"}
    )

# "Setting" = method x training regime, used as the boxplot hue in the per-target
# panels. Colours pair by method (POME vs UMAP), light = full, dark = inductive.
df["Setting"] = df["Method"] + " (inductive)"
if full_df is not None:
    full_df["Setting"] = full_df["Method"] + " (full)"
    # Restrict to embedding sizes present in both analyses for a matched comparison.
    common_dims = sorted(set(df["dim"]) & set(full_df["dim"]))
    panel_df = pd.concat(
        [df[df["dim"].isin(common_dims)], full_df[full_df["dim"].isin(common_dims)]],
        ignore_index=True,
    )
    setting_order = ["POME (inductive)", "POME (full)", "UMAP (inductive)", "UMAP (full)"]
    # Keep POME/UMAP identity by hue family (POME = Set2 mint-green, UMAP = Set2
    # orange, matching the paper); inductive uses the canonical Set2 colour and
    # full a lighter tint of the same hue.
    set2 = sns.color_palette("Set2")
    pome_color, umap_color = set2[0], set2[1]
    setting_palette = {
        "POME (inductive)": pome_color,
        "POME (full)": lighten(pome_color, 0.55),
        "UMAP (inductive)": umap_color,
        "UMAP (full)": lighten(umap_color, 0.55),
    }
else:
    panel_df = df
    setting_order = ["POME (inductive)", "UMAP (inductive)"]
    setting_palette = dict(zip(setting_order, sns.color_palette("Set2")))

# ---------------------------------------------------------------------------
# Panel a: aggregate over targets and embedding sizes, separately per regime.
# For each (Method, Target, dim) take the median score, then the best median
# per (Method, Target), and count how often each method wins -- computed
# independently for the inductive and full-dataset results.
# ---------------------------------------------------------------------------
def method_win_counts(frame):
    """(#targets POME wins, #targets UMAP wins) by best-across-dim median score."""
    median_scores = frame.groupby(["Method", "Target", "dim"])["Score"].median().reset_index()
    max_scores = median_scores.groupby(["Method", "Target"])["Score"].max().reset_index()
    pivot = max_scores.pivot(index="Target", columns="Method", values="Score")
    return int((pivot["POME"] > pivot["UMAP"]).sum()), int((pivot["UMAP"] > pivot["POME"]).sum())


pome_ind, umap_ind = method_win_counts(df)
agg_rows = [
    {"Setting": "POME (inductive)", "count": pome_ind},
    {"Setting": "UMAP (inductive)", "count": umap_ind},
]
if full_df is not None:
    pome_full, umap_full = method_win_counts(full_df)
    agg_rows += [
        {"Setting": "POME (full)", "count": pome_full},
        {"Setting": "UMAP (full)", "count": umap_full},
    ]
agg_df = pd.DataFrame(agg_rows)

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

# Panel a: aggregated bar plot, one bar per method x regime. Bars are grouped by
# regime (both inductive bars, then both full bars), with tighter spacing within
# a regime pair than between the pairs.
counts = dict(zip(agg_df["Setting"], agg_df["count"]))
if full_df is not None:
    panel_a_order = ["POME (inductive)", "UMAP (inductive)", "POME (full)", "UMAP (full)"]
    bar_positions = [0.0, 0.95, 2.4, 3.35]  # within-pair gap << between-pair gap
else:
    panel_a_order = ["POME (inductive)", "UMAP (inductive)"]
    bar_positions = [0.0, 0.95]
bar_width = 0.85

axes["a"].bar(
    bar_positions,
    [counts[s] for s in panel_a_order],
    width=bar_width,
    color=[setting_palette[s] for s in panel_a_order],
)
axes["a"].set_xlabel("")
# Per-bar tick labels show the method; the regime is annotated once per pair.
axes["a"].set_xticks(bar_positions)
axes["a"].set_xticklabels(
    [s.split(" (")[0] for s in panel_a_order], fontsize=ticklabelsize
)
_blend = transforms.blended_transform_factory(axes["a"].transData, axes["a"].transAxes)
for regime, pair in (("inductive", bar_positions[:2]), ("full", bar_positions[2:])):
    if pair:
        axes["a"].text(
            sum(pair) / len(pair), -0.16, regime, transform=_blend,
            ha="center", va="top", fontsize=ticklabelsize,
        )
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
