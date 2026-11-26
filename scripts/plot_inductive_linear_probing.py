"""Plot the inductive probing results of both probes (logistic regression and MLP) in one figure.

3x3 mosaic: one aggregated bar panel (a) + eight per-target panels (b-i), from the results of
``scripts/linear_probe_inductive_embeddings.py --probe {logreg,mlp}`` (embeddings fit on the
train split and transformed onto the held-out test split). Methods: POME, UMAP and the tabular
competitors HI-VAE, SAINT and missForest+AE, with the names and colours of the clustering
figure. The TCGA-LUAD panels show ``luad_noleak`` (LUAD without its outcome/follow-up variables).

* Panels b-i: for every embedding size and method, the median test AP (over splits x runs) of
  the logistic regression (filled circle) and of the MLP (empty circle), joined by a line, all
  in the colour of the method. Dashed lines: positive class ratio (black) and the supervised
  TabICL reference (red; median test AP over splits of TabICL fit/predicted directly on the raw
  split tables with the target labels, ``scripts/tabicl_inductive_reference.py``).
* Panel a: per method and probe (filled bar = logistic regression, empty bar = MLP), the number
  of targets on which the method has the strictly best median AP (best over embedding sizes)
  among the five methods. TabICL takes no part in the count; the wins that also beat the TabICL
  reference are printed.

    container/run.sh scripts/plot_inductive_linear_probing.py [--output PDF] [--tabicl CSV] \
        [--dims 16 32 64] [--methods POME UMAP]

Inputs: ``output/linear_probing/inductive_{logreg,mlp}_probing_results.csv`` and
``output/linear_probing/inductive_tabicl_reference.csv``.
Output: ``output/linear_probing/inductive_probing_results.pdf``.
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "output", "linear_probing")
LUAD_DATASET = "luad_noleak"
PROBES = {"logreg": "Logistic regression", "mlp": "MLP"}   # filled / empty markers and bars

parser = argparse.ArgumentParser(description="Plot the inductive probing results.")
for probe in PROBES:
    parser.add_argument(f"--results-{probe}",
                        default=os.path.join(OUT_DIR, f"inductive_{probe}_probing_results.csv"),
                        help=f"{probe} probing results CSV")
parser.add_argument("--tabicl", default=os.path.join(OUT_DIR, "inductive_tabicl_reference.csv"),
                    help="supervised TabICL reference CSV")
parser.add_argument("--output", default=os.path.join(OUT_DIR, "inductive_probing_results.pdf"))
parser.add_argument("--dims", nargs="+", type=int, default=None,
                    help="embedding sizes to show (default: all in the results files)")
parser.add_argument("--methods", nargs="+", default=None,
                    help="methods to show, by figure name (default: all in the results files)")
args = parser.parse_args()

# results-file method label -> figure name, in the figure order of the clustering analysis
METHOD_NAMES = {"POME": "POME", "UMAP": "UMAP", "HIVAE": "HI-VAE", "SAINT": "SAINT",
                "MFAE": "missForest+AE"}
# fixed colour per method (Set2 in METHOD_NAMES order), so a subset keeps its colours
palette = dict(zip(METHOD_NAMES.values(), sns.color_palette("Set2")))


def load(path, probe):
    df = pd.read_csv(path)
    if set(df["probe"]) != {probe}:
        raise SystemExit(f"{path}: expected probe {probe}, found {sorted(set(df['probe']))}")
    if "mode" in df.columns:          # numeric_only / cat_only feed the type-combination figure
        df = df[df["mode"] == "combined"]
    df = df[df["dataset"].isin(["mimic", "hancock", LUAD_DATASET])].copy()
    df.loc[df["dataset"] == LUAD_DATASET, "dataset"] = "luad"
    df["method"] = df["method"].map(METHOD_NAMES).fillna(df["method"])
    return df


df = pd.concat([load(getattr(args, f"results_{p}"), p) for p in PROBES], ignore_index=True)
if args.dims is not None:
    df = df[df["dim"].isin(args.dims)]
present = set(df["method"])
methods = args.methods or [m for m in METHOD_NAMES.values() if m in present]
missing = [m for m in methods if m not in present]
if missing:
    raise SystemExit(f"methods not in the results files: {missing}")
df = df[df["method"].isin(methods)]
dims = sorted(df["dim"].unique())

# median AP per (probe, dataset, target, method, dim) over splits x runs
med = (df.groupby(["probe", "dataset", "target", "method", "dim"])["average_precision"]
       .median())
pos_ratio = df.groupby(["dataset", "target"])["test_pos_ratio"].mean()

# Supervised TabICL reference: median test AP over splits per (dataset, target).
tabicl = pd.read_csv(args.tabicl)
tabicl.loc[tabicl["dataset"] == LUAD_DATASET, "dataset"] = "luad"
tabicl_ref = tabicl.groupby(["dataset", "target"])["average_precision"].median()
missing_ref = set(pos_ratio.index) - set(tabicl_ref.index)
if missing_ref:
    raise SystemExit(f"no TabICL reference in {args.tabicl} for {sorted(missing_ref)}")


# ---------------------------------------------------------------------------
# Panel a: per probe, the targets on which each method has the strictly best
# best-across-dim median AP (ties count for nobody).
# ---------------------------------------------------------------------------
def win_counts(probe):
    best = med[probe].groupby(["dataset", "target", "method"]).max().unstack("method")[methods]
    top = best.max(axis=1)
    winners = best.eq(top, axis=0)
    unique = winners.sum(axis=1) == 1
    above = top > tabicl_ref.reindex(best.index)
    wins = {m: int((winners[m] & unique).sum()) for m in methods}
    wins_above = {m: int((winners[m] & unique & above).sum()) for m in methods}
    return wins, wins_above


wins = {p: win_counts(p) for p in PROBES}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
panels = [  # (dataset, target, panel key, title)
    ("mimic", "Aplasia", "b", "MIMIC-IV: Aplasia"),
    ("mimic", "Neutropenic Fever", "c", "MIMIC-IV: Neutropenic fever"),
    ("hancock", "Recurrence", "d", "HANCOCK: Recurrence"),
    ("hancock", "Survival", "e", "HANCOCK: Survival"),
    ("hancock", "RFS Event", "f", "HANCOCK: Recurrence-free \nsurvival event"),
    ("luad", "Disease Free Status", "g", "TCGA-LUAD: Disease-free status"),
    ("luad", "DSS Status", "h", "TCGA-LUAD: Disease-specific survival"),
    ("luad", "Progression Free Status", "i", "TCGA-LUAD: Progression-free status"),
]
mosaic = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
fig, axes = plt.subplot_mosaic(mosaic, figsize=(16, 12), layout="constrained")

offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)
labelfontsize = 20
titlefontsize = 20
ticklabelsize = 16
legendfontsize = 16
markersize = 9


def panel_label(ax, key):
    ax.text(0.0, 1.0, key, transform=ax.transAxes + offset, fontsize=labelfontsize,
            fontweight="bold")


# Panel a: two bars per method (filled = logistic regression, empty = MLP).
ax = axes["a"]
x = np.arange(len(methods))
bar_width = 0.4
for i, probe in enumerate(PROBES):
    counts = [wins[probe][0][m] for m in methods]
    colors = [palette[m] for m in methods]
    ax.bar(x + (i - 0.5) * bar_width, counts, width=bar_width * 0.9,
           color=colors if probe == "logreg" else "white", edgecolor=colors, linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=ticklabelsize - 2, rotation=30, ha="right",
                   rotation_mode="anchor")
ax.set_ylabel("Number of targets with \nbest median score", fontsize=labelfontsize)
ax.tick_params(axis="y", labelsize=ticklabelsize)
ax.yaxis.get_major_locator().set_params(integer=True)
ax.set_title("Results aggregated over \ntargets and embedding sizes", fontsize=titlefontsize)
panel_label(ax, "a")

# Per-target panels: dumbbell per (dim, method), logreg filled -> MLP empty.
YLABEL_PANELS = {"b", "d", "g"}   # first target panel of each row
group_width = 0.8
step = group_width / len(methods)
for dataset, target, key, title in panels:
    ax = axes[key]
    for j, m in enumerate(methods):
        xs = np.arange(len(dims)) - group_width / 2 + (j + 0.5) * step
        y_lr = [med.get(("logreg", dataset, target, m, d), np.nan) for d in dims]
        y_mlp = [med.get(("mlp", dataset, target, m, d), np.nan) for d in dims]
        for xi, a, b in zip(xs, y_lr, y_mlp):
            ax.plot([xi, xi], [a, b], color=palette[m], linewidth=2, zorder=2)
        ax.plot(xs, y_lr, "o", color=palette[m], markersize=markersize, zorder=3)
        ax.plot(xs, y_mlp, "o", markerfacecolor="white", markeredgecolor=palette[m],
                markeredgewidth=2, markersize=markersize, zorder=3)
    ax.axhline(pos_ratio[(dataset, target)], color="black", linestyle="--", linewidth=2)
    ax.axhline(tabicl_ref[(dataset, target)], color="firebrick", linestyle="--", linewidth=2)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_xlim(-0.5, len(dims) - 0.5)
    ax.set_xlabel("Embedding size", fontsize=labelfontsize)
    if key in YLABEL_PANELS:
        ax.set_ylabel("Median average precision", fontsize=labelfontsize - 2)
    ax.tick_params(axis="both", labelsize=ticklabelsize)
    ax.set_title(title, fontsize=titlefontsize)
    panel_label(ax, key)

# Shared legend at the top: methods (colour), probes (fill), reference lines.
handles = [Patch(facecolor=palette[m], edgecolor=palette[m], label=m) for m in methods]
handles += [
    Line2D([], [], marker="o", linestyle="", color="dimgray", markersize=markersize,
           label=PROBES["logreg"]),
    Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
           markeredgecolor="dimgray", markeredgewidth=2, markersize=markersize,
           label=PROBES["mlp"]),
    Line2D([], [], color="black", linestyle="--", linewidth=2, label="Positive class ratio"),
    Line2D([], [], color="firebrick", linestyle="--", linewidth=2, label="TabICL (supervised)"),
]
# legends fill column-wise: interleave so row 1 = methods, row 2 = probes + reference lines
first, second = handles[:len(methods)], handles[len(methods):]
second += [Patch(visible=False, label="")] * (len(first) - len(second))   # pad row 2
handles = [h for pair in zip(first, second) for h in pair]
legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 1.04),
                    ncol=len(first), fontsize=legendfontsize, frameon=True,
                    fancybox=False)

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
fig.savefig(args.output, bbox_inches="tight", bbox_extra_artists=[legend])
print(f"Saved figure to {args.output}")
summary = pd.DataFrame({f"{p}_wins": wins[p][0] for p in PROBES}
                       | {f"{p}_wins_above_tabicl": wins[p][1] for p in PROBES})
print(summary.to_string())
