"""Plot inductive linear probing results.

Reproduces the layout of scripts/plot_fig3_linear_probing.ipynb (3x3 mosaic:
one aggregated bar panel + eight per-target boxplot panels), using the
inductive linear probing results (embeddings fit on the train split and
transformed onto the held-out test split) instead of that notebook's
transductive k-fold scores.

Only the inductive scores are shown: each embedding size draws one box per
method (POME and UMAP, plus the tabular competitors HI-VAE, SAINT and
missForest+AE when the results file has them; same names and colours as the
clustering figure). Label-free TabICL v1.1 embeddings (method TABICL_LF) are drawn as "TabICL"
and compete in panel a. Target-conditioned TabICL embeddings (method TABICL) are drawn as
"TabICL (supervised)" and are not counted in panel a (wins among the label-free methods).

Usage:
    container/run.sh scripts/plot_inductive_linear_probing.py \
        [--results CSV] [--output PDF] [--dims 16 32 64] [--methods POME UMAP]
Defaults reproduce the original figure (all dims and methods in the results file).

The same script draws the non-linear (XGBoost) probing figure from the results of
``linear_probe_inductive_embeddings.py --probe xgboost``; the layout is unchanged. A results file
that mixes probes (``probe`` column; absent = logistic regression) is rejected.
"""

import argparse
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
parser = argparse.ArgumentParser(description="Plot inductive linear probing results.")
parser.add_argument("--results", default=RESULTS_PATH, help="probing results CSV")
parser.add_argument("--output", default=OUT_PATH, help="output PDF")
parser.add_argument("--dims", nargs="+", type=int, default=None,
                    help="embedding sizes to show (default: all in the results file)")
parser.add_argument("--luad-dataset", default="luad",
                    help="results dataset drawn in the TCGA-LUAD panels: luad (default) or "
                         "luad_noleak (without the outcome/follow-up variables)")
parser.add_argument("--methods", nargs="+", default=None,
                    help="methods to show, by figure name (default: all in the results file)")
args = parser.parse_args()
RESULTS_PATH, OUT_PATH = args.results, args.output

df = pd.read_csv(RESULTS_PATH)
probes = set(df["probe"].fillna("logreg")) if "probe" in df else {"logreg"}
if len(probes) != 1:
    raise SystemExit(f"{RESULTS_PATH} mixes probes {sorted(probes)}")
if args.dims is not None:
    df = df[df["dim"].isin(args.dims)].copy()

# The results file may also carry the variable-type-restricted runs (mode
# numeric_only / cat_only) that feed plot_inductive_type_combination.py. This
# figure is the all-variables comparison only.
if "mode" in df.columns:
    df = df[df["mode"] == "combined"].copy()
# One LUAD variant only: luad and luad_noleak share target names, so mixing them would also
# merge their scores in the panel-a win counts.
df = df[df["dataset"].isin(["mimic", "hancock", args.luad_dataset])].copy()
df.loc[df["dataset"] == args.luad_dataset, "dataset"] = "luad"
if df[df["dataset"] == "luad"].empty:
    raise SystemExit(f"no {args.luad_dataset} rows in {RESULTS_PATH}")

# Use average precision as the score, mirroring the "Average precision" y-axis
# of the original figure.
df = df.rename(columns={"average_precision": "Score", "method": "Method", "target": "Target"})
# results-file method label -> figure name, in the figure order of the clustering analysis
# TABICL_LF: label-free TabICL v1.1 row embeddings (embed_tabicl_labelfree_inductive.py).
# TABICL: TabICLv2 embeddings conditioned on the probed target (embed_tabicl_inductive.py), i.e. a
# supervised reference, which the figure name says.
METHOD_NAMES = {"POME": "POME", "UMAP": "UMAP", "HIVAE": "HI-VAE", "SAINT": "SAINT",
                "MFAE": "missForest+AE", "TABICL_LF": "TabICL",
                "TABICL": "TabICL (supervised)"}
# Supervised references: drawn in the per-target panels, but not counted in panel a, which
# compares the label-free representations.
REFERENCE_METHODS = {"TabICL (supervised)"}
df["Method"] = df["Method"].map(METHOD_NAMES).fillna(df["Method"])
present = set(df["Method"])
methods = args.methods or [m for m in METHOD_NAMES.values() if m in present]
missing = [m for m in methods if m not in present]
if missing:
    raise SystemExit(f"methods not in {RESULTS_PATH}: {missing}")
df = df[df["Method"].isin(methods)].copy()

# "Setting" = method, used as the boxplot hue in the per-target panels
# (POME = Set2 mint-green, UMAP = Set2 orange, matching the paper).
df["Setting"] = df["Method"]
panel_df = df
setting_order = list(methods)
# fixed colour per method (Set2 in METHOD_NAMES order), so a subset keeps its colours
setting_palette = dict(zip(METHOD_NAMES.values(), sns.color_palette("Set2")))

# ---------------------------------------------------------------------------
# Panel a: aggregate over targets and embedding sizes. For each (Method, Target,
# dim) take the median score, then the best median per (Method, Target), and
# count how often each method has the strictly highest one (ties count for nobody).
# Supervised references (REFERENCE_METHODS) take no part: no bar and not compared.
# ---------------------------------------------------------------------------
def method_win_counts(frame, order):
    """{method: #targets it wins} by best-across-dim median score."""
    median_scores = frame.groupby(["Method", "Target", "dim"])["Score"].median().reset_index()
    max_scores = median_scores.groupby(["Method", "Target"])["Score"].max().reset_index()
    pivot = max_scores.pivot(index="Target", columns="Method", values="Score")[order]
    best = pivot.max(axis=1)
    winners = pivot.eq(best, axis=0)
    unique = winners.sum(axis=1) == 1
    return {m: int((winners[m] & unique).sum()) for m in order}


panel_a_order = [m for m in setting_order if m not in REFERENCE_METHODS]
wins = method_win_counts(df, panel_a_order)
agg_df = pd.DataFrame({"Setting": panel_a_order, "count": [wins[m] for m in panel_a_order]})

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
bar_positions = [0.95 * i for i in range(len(panel_a_order))]
bar_width = 0.85

axes["a"].bar(
    bar_positions,
    [counts[s] for s in panel_a_order],
    width=bar_width,
    color=[setting_palette[s] for s in panel_a_order],
)
axes["a"].set_xlabel("")
axes["a"].set_xticks(bar_positions)
if len(panel_a_order) > 2:
    axes["a"].set_xticklabels(panel_a_order, fontsize=ticklabelsize - 2, rotation=30, ha="right",
                              rotation_mode="anchor")
else:
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
