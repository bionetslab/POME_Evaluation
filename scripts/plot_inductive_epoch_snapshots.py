"""Plot POME inductive linear-probing quality as a function of training epochs.

Consumes the tidy results of
``scripts/linear_probe_inductive_epoch_snapshots.py``
(``output/linear_probing/inductive_epoch_snapshots_results.csv``) and renders the
same 3x3 mosaic layout as ``scripts/plot_inductive_linear_probing.py`` -- one
aggregate panel (a) plus eight per-target panels (b-i) -- but with training epoch
on the x-axis and average precision on the y-axis. Each line is one embedding
size; the band is the 95% CI across the 10 splits x 10 runs.

POME only (UMAP has no training-epoch axis).

Run from the project root:

    conda run -n torch python scripts/plot_inductive_epoch_snapshots.py

Output: output/linear_probing/inductive_epoch_snapshots.pdf
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
    SCRIPT_DIR, "..", "output", "linear_probing",
    "inductive_epoch_snapshots_results.csv")
OUT_PATH = os.path.join(
    SCRIPT_DIR, "..", "output", "linear_probing",
    "inductive_epoch_snapshots.pdf")

df = pd.read_csv(RESULTS_PATH)
df = df[df["method"] == "POME"].copy()  # POME only; UMAP has no epoch axis
df = df.rename(columns={"average_precision": "Score", "target": "Target"})

# Embedding sizes present, ordinal -> sequential palette (small=light, large=dark).
dims = sorted(df["dim"].unique())
dim_palette = dict(zip(dims, sns.color_palette("crest", n_colors=len(dims))))
epochs = sorted(df["epoch"].unique())

# ---------------------------------------------------------------------------
# Per-target panels: (dataset, target, panel key, title) -- same order/titles as
# scripts/plot_inductive_linear_probing.py.
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


def draw_lines(ax, data, legend=False):
    """Average-precision vs epoch, one line (mean + 95% CI band) per dim."""
    sns.lineplot(
        data=data, x="epoch", y="Score", hue="dim", hue_order=dims,
        palette=dim_palette, marker="o", ax=ax, legend=legend, errorbar=("ci", 95),
    )
    ax.set_xticks(epochs)


# Panel a: aggregate across all targets (mean AP per dim/epoch over every
# target x split x run), giving the overall training-epoch trend. The shared
# legend is captured here (always populated) rather than from a target panel,
# which may be empty when only a subset of datasets was probed.
draw_lines(axes["a"], df, legend=True)
legend_handles, legend_labels = axes["a"].get_legend_handles_labels()
legend_labels = [f"Embedding size {lbl}" if lbl.isdigit() else lbl
                 for lbl in legend_labels]
axes["a"].legend_.remove()
axes["a"].set_xlabel("Training epochs", fontsize=labelfontsize)
axes["a"].set_ylabel("Average precision", fontsize=labelfontsize)
axes["a"].tick_params(axis="x", labelsize=ticklabelsize)
axes["a"].tick_params(axis="y", labelsize=ticklabelsize)
axes["a"].set_title("Mean over all targets", fontsize=titlefontsize)
axes["a"].text(0.0, 1.0, "a", transform=axes["a"].transAxes + offset,
               fontsize=labelfontsize, fontweight="bold")

# Per-target panels.
naive_handle = None
for dataset, target, key, title in panels:
    sub = df[(df["dataset"] == dataset) & (df["Target"] == target)]
    if sub.empty:
        axes[key].set_visible(False)
        continue

    draw_lines(axes[key], sub, legend=False)

    # Positive class ratio (dashed) = mean test positive ratio for the target.
    naive = sub["test_pos_ratio"].mean()
    naive_handle = axes[key].axhline(
        naive, color="black", linestyle="--", linewidth=2,
        label="Positive class ratio")

    axes[key].set_xlabel("Training epochs", fontsize=labelfontsize)
    axes[key].set_ylabel("Average precision", fontsize=labelfontsize)
    axes[key].tick_params(axis="x", labelsize=ticklabelsize)
    axes[key].tick_params(axis="y", labelsize=ticklabelsize)
    axes[key].set_title(title, fontsize=titlefontsize)
    axes[key].text(0.0, 1.0, key, transform=axes[key].transAxes + offset,
                   fontsize=labelfontsize, fontweight="bold")

# Append the positive-class-ratio entry to the shared legend.
if naive_handle is not None:
    legend_handles = legend_handles + [naive_handle]
    legend_labels = legend_labels + ["Positive class ratio"]

# Shared legend at the top.
legend1 = fig.legend(
    legend_handles, legend_labels, loc="center", bbox_to_anchor=(0.49, 1.03),
    ncol=len(legend_labels), fontsize=legendfontsize, frameon=True, fancybox=False,
)

fig.savefig(OUT_PATH, bbox_inches="tight", bbox_extra_artists=[legend1])
print(f"Saved figure to {OUT_PATH}")

# Compact summary: mean AP by dim x epoch, aggregated over targets/splits/runs.
print("\n=== Mean average precision by embedding size x epoch "
      "(over all targets/splits/runs) ===")
summary = (df.groupby(["dim", "epoch"])["Score"].mean().round(4)
           .unstack("epoch"))
with pd.option_context("display.max_rows", None, "display.width", 160):
    print(summary)
