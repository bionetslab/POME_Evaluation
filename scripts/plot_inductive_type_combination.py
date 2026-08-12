"""Plot the effect of data-type integration under *inductive* linear probing.

Inductive counterpart to ``scripts/plot_type_combination_effect.ipynb``, which builds
the same figure from the transductive k-fold scores in
``output/supervised/*_regression_results_*.csv`` (produced by the hand-toggled
``src/pome_evaluation/analyze_*_embedding_separability.ipynb`` notebooks). That
legacy path is kept in the repo for comparison; new work uses this script.

The question is unchanged: for each embedding method, on how many targets does
embedding *all* variables beat embedding only the continuous or only the
categorical ones? What changes is the evaluation protocol -- scores now come
from ``output/linear_probing/inductive_linear_probing_results.csv``, i.e. a
logistic regression fit on the train-split embedding and evaluated on the
held-out test-split embedding, over the same ``data/train_test_splits/``
partitions used by ``scripts/plot_inductive_linear_probing.py``. The three
modes are embedded from variable-type subsets of those very same splits, so
Combined / Numeric only / Categorical only differ *only* in which variables the
encoder saw.

Requires the restricted modes to have been generated and probed first:

    python scripts/generate_inductive_embeddings.py --modes numeric_only cat_only ...
    python scripts/linear_probe_inductive_embeddings.py

Layout (2x4 mosaic), mirroring the notebook figure:

    a  HANCOCK   : #targets where Combined wins, per method
    b,c,d        : HANCOCK per-target boxplots (Recurrence, Survival, RFS event)
    e  TCGA-LUAD : #targets where Combined wins, per method
    f,g,h        : TCGA-LUAD per-target boxplots (DFS, DSS, PFS)

MIMIC-IV is absent by construction: its UMAP matrix has no categorical block,
so the numeric-only and combined arms would be identical.

All three UMAP arms are scored on the full test split thanks to the NA-aware
umap fork's pairwise-removal distances (see the docstring of
``scripts/generate_inductive_embeddings.py``). A non-zero ``n_test_nonfinite``
in the results CSV means stock umap-learn was picked up instead, which would
make the single-modality arms complete-case-only and not comparable.

Run from the project root:

    python scripts/plot_inductive_type_combination.py
    python scripts/plot_inductive_type_combination.py --dim 32

Output: output/linear_probing/inductive_type_combination_effect.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import seaborn as sns

# --- Configuration -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "output" / "linear_probing" / \
    "inductive_linear_probing_results.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "linear_probing" / \
    "inductive_type_combination_effect.pdf"
DEFAULT_DIM = 64

# Raw mode key -> display label / plotting order (matches the notebook's hue).
MODE_LABELS = {
    "combined": "Combined",
    "numeric_only": "Numeric only",
    "cat_only": "Categorical only",
}
MODE_ORDER = ["Combined", "Numeric only", "Categorical only"]
METHOD_ORDER = ["POME", "UMAP"]

# (dataset, target label in the inductive results, panel key, panel title).
# The bar panel of each dataset aggregates exactly over its three targets.
PANELS = [
    ("hancock", "Recurrence", "b", "HANCOCK: Recurrence"),
    ("hancock", "Survival", "c", "HANCOCK: Survival"),
    ("hancock", "RFS Event", "d", "HANCOCK: Recurrence-free \nsurvival event"),
    ("luad", "Disease Free Status", "f", "TCGA-LUAD: \nDisease-free status"),
    ("luad", "DSS Status", "g", "TCGA-LUAD: \nDisease-specific survival"),
    ("luad", "Progression Free Status", "h",
     "TCGA-LUAD: \nProgression-free status"),
]
BAR_PANELS = [("hancock", "a", "HANCOCK"), ("luad", "e", "TCGA-LUAD")]


# --- Analysis ----------------------------------------------------------------
def combined_win_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """#targets per method whose Combined median beats both single-type medians.

    Medians are taken over splits x runs within the selected embedding size,
    reproducing the notebook's per-(Method, Target, Mode) median comparison.
    """
    medians = (frame.groupby(["Method", "Target", "Mode"])["Score"]
               .median().unstack("Mode"))
    missing = [m for m in MODE_ORDER if m not in medians.columns]
    if missing:
        raise SystemExit(
            f"Missing modes {missing} in the results -- generate and probe them "
            f"first (see this script's docstring).")
    medians["Combined_best"] = (
        (medians["Combined"] > medians["Categorical only"])
        & (medians["Combined"] > medians["Numeric only"]))
    counts = (medians.groupby(level="Method")["Combined_best"]
              .agg(count_higher="sum", total_targets="count").reset_index())
    return counts[counts["Method"].isin(METHOD_ORDER)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"embedding size to plot (default: {DEFAULT_DIM})")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f"Results file not found: {args.results}")
    df = pd.read_csv(args.results)
    if "mode" not in df.columns:
        raise SystemExit(
            f"{args.results} has no `mode` column -- re-run "
            f"scripts/linear_probe_inductive_embeddings.py after generating the "
            f"numeric_only / cat_only embeddings.")

    df = df.rename(columns={"average_precision": "Score", "method": "Method",
                            "target": "Target"})
    df["Mode"] = df["mode"].map(MODE_LABELS)
    df = df[df["Mode"].notna() & df["Method"].isin(METHOD_ORDER)]

    dim_df = df[df["dim"] == args.dim]
    if dim_df.empty:
        raise SystemExit(f"No results for dim {args.dim} in {args.results}; "
                         f"available: {sorted(df['dim'].unique())}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    mosaic = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(16, 9.5),
                                   layout="constrained")

    offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)
    labelfontsize = 20
    titlefontsize = 18
    ticklabelsize = 16
    legendfontsize = 16

    # Boxplot hue = mode (default palette, as in the notebook); the aggregate
    # bars are per method (Set2, matching plot_inductive_linear_probing.py).
    mode_palette = dict(zip(MODE_ORDER, sns.color_palette()))
    method_palette = dict(zip(METHOD_ORDER, sns.color_palette("Set2")))

    summaries = {}
    for dataset, key, title in BAR_PANELS:
        targets = [t for ds, t, *_ in PANELS if ds == dataset]
        summary = combined_win_counts(
            dim_df[(dim_df["dataset"] == dataset)
                   & (dim_df["Target"].isin(targets))])
        summaries[dataset] = summary

        counts = dict(zip(summary["Method"], summary["count_higher"]))
        axes[key].bar([0.0, 0.95], [counts.get(m, 0) for m in METHOD_ORDER],
                      width=0.85,
                      color=[method_palette[m] for m in METHOD_ORDER])
        axes[key].set_xticks([0.0, 0.95])
        axes[key].set_xticklabels(METHOD_ORDER, fontsize=ticklabelsize)
        axes[key].set_xlabel("Embedding method", fontsize=labelfontsize)
        axes[key].set_ylabel("Number of targets benefiting \nfrom data "
                             "integration", fontsize=labelfontsize)
        axes[key].tick_params(axis="y", labelsize=ticklabelsize)
        axes[key].set_title(title, fontsize=titlefontsize)
        axes[key].set_ylim(0, len(targets) + 0.1)
        axes[key].set_yticks(range(len(targets) + 1))
        axes[key].text(0.0, 1.0, key, transform=axes[key].transAxes + offset,
                       fontsize=labelfontsize, fontweight="bold")

    legend_handles, legend_labels = None, None
    for dataset, target, key, title in PANELS:
        sub = dim_df[(dim_df["dataset"] == dataset)
                     & (dim_df["Target"] == target)]
        if sub.empty:
            print(f"  [warn] no rows for {dataset}/{target} at dim {args.dim}")
            continue
        # Baseline: the naive average precision of a constant predictor is the
        # test-set positive rate. The transductive notebook read it from a
        # `Method == "Naive"` row; inductively it varies per split, so the
        # median over splits x runs is drawn.
        naive = sub["test_pos_ratio"].median()

        show_legend = key == "b"
        sns.boxplot(data=sub, x="Method", y="Score", hue="Mode",
                    order=METHOD_ORDER, hue_order=MODE_ORDER, ax=axes[key],
                    dodge=True, gap=0.2, legend=show_legend,
                    palette=mode_palette)
        axes[key].axhline(naive, color="black", linestyle="--", linewidth=2,
                          label="Positive class ratio")

        axes[key].set_xlabel("Embedding method", fontsize=labelfontsize)
        axes[key].set_ylabel("Average precision", fontsize=labelfontsize)
        axes[key].tick_params(axis="x", labelsize=ticklabelsize)
        axes[key].tick_params(axis="y", labelsize=ticklabelsize)
        axes[key].set_title(title, fontsize=titlefontsize)
        axes[key].text(0.0, 1.0, key, transform=axes[key].transAxes + offset,
                       fontsize=labelfontsize, fontweight="bold")

        if show_legend:
            legend_handles, legend_labels = \
                axes[key].get_legend_handles_labels()
            axes[key].legend_.remove()

    extra_artists = []
    if legend_handles:
        legend1 = fig.legend(legend_handles, legend_labels, loc="center",
                             bbox_to_anchor=(0.49, 1.03),
                             ncol=len(legend_labels), fontsize=legendfontsize,
                             frameon=True, fancybox=False)
        extra_artists.append(legend1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", bbox_extra_artists=extra_artists)
    print(f"Saved figure to {args.output}  (dim {args.dim})")
    for dataset, summary in summaries.items():
        print(f"\n[{dataset}] targets benefiting from data integration:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
