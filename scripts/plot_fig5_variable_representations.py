"""Revised variable-semantics figure (manuscript Sec. 2.6), replacing the best-run panels.

Uses POME's learned variable encodings e_j (``raw``; identifiable in POME >= 1.4, where the sample
and value encodings are fixed) on the 10 seeds x 3 encoding sizes produced by
generate_variable_representations.py and analyzed by analyze_variable_representation_stability.py.

    a  Spearman correlation between CatBoost feature importance and target-to-lab cosine
       similarity across the 10 seeds.
    b  Cross-seed reproducibility: Spearman correlation of the target-to-lab cosine profiles
       between all 45 pairs of seeds.
    c  Cluster reproducibility: fraction of seeds in which a lab variable falls into the Leiden
       cluster of Aplasia (top labs, encoding size 32).

CPU only:  container/run.sh scripts/plot_fig5_variable_representations.py [--tag TAG] [--option OPT]
Output:    output/variable_embeddings/variable_results_revised{TAG}[_{OPT}].pdf (+ .png)
           (default tag ``_pome14``; ``_{OPT}`` only for a non-default representation)
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAB_DIR = PROJECT_ROOT / "output" / "variable_embeddings" / "representation_stability"
OUT = PROJECT_ROOT / "output" / "variable_embeddings" / "variable_results_revised"
LABITEMS = PROJECT_ROOT / "data" / "input_datasets" / "d_labitems.csv.gz"
OPTION = "raw"
COCLUSTER_DIM = 32
TOP_LABS = 12
TARGET_NAMES = {"Aplasia": "Aplasia", "NF": "Neutropenic Fever"}
HUE_ORDER = list(TARGET_NAMES.values())
PALETTE = dict(zip(HUE_ORDER, sns.color_palette("tab10", 2)))

labelfontsize, titlefontsize, ticklabelsize = 20, 19, 16
xylabelsize, legendfontsize = 18, 15


def load(option=OPTION):
    imp = pd.read_csv(STAB_DIR / "importance_correlation.csv")
    imp = imp[imp["option"] == option].replace({"target": TARGET_NAMES})

    prof = pd.read_csv(STAB_DIR / "target_profiles.csv", dtype={"lab": str})
    prof = prof[prof["option"] == option]
    rows = []
    for (dim, target), grp in prof.groupby(["dim", "target"]):
        p = grp.pivot(index="lab", columns="run", values="cosine")
        for a, b in itertools.combinations(p.columns, 2):
            rows.append(dict(dim=dim, target=TARGET_NAMES[target], rho=spearmanr(p[a], p[b])[0]))
    repro = pd.DataFrame(rows)

    co = pd.read_csv(STAB_DIR / "aplasia_coclustering.csv", dtype={"lab": str})
    co = co[(co["option"] == option) & (co["dim"] == COCLUSTER_DIM)]
    labels = pd.read_csv(LABITEMS)
    label_of = dict(zip(labels["itemid"].astype(str), labels["label"] + " (" + labels["fluid"] + ")"))
    co["name"] = co["lab"].map(label_of).str.replace(" (Blood)", "", regex=False)
    co = co.sort_values("frac_runs_with_aplasia", ascending=False).head(TOP_LABS)
    return imp, repro, co


def panel_label(ax, letter, offset):
    ax.text(0.0, 1.0, letter, transform=ax.transAxes + offset, fontsize=labelfontsize,
            fontweight="bold")


def boxes(ax, data, y, show_points):
    sns.boxplot(data=data, x="dim", y=y, hue="target", hue_order=HUE_ORDER, palette=PALETTE,
                dodge=True, gap=0.2, ax=ax, showfliers=not show_points, legend=True)
    if show_points:
        sns.stripplot(data=data, x="dim", y=y, hue="target", hue_order=HUE_ORDER,
                      palette=PALETTE, dodge=True, ax=ax, size=5, color="black", alpha=0.6,
                      edgecolor="white", linewidth=0.5, legend=False)
    ax.axhline(0, color="0.5", lw=1, ls="--", zorder=0)
    ax.set_xlabel("Encoding size", fontsize=xylabelsize)
    ax.tick_params(labelsize=ticklabelsize)
    ax.get_legend().remove()


def main():
    global STAB_DIR, OUT
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", default="_pome14",
                        help="suffix of the analyzed stability dir and of the output "
                             "(default: _pome14)")
    parser.add_argument("--option", default=OPTION,
                        help="variable representation (default: raw = learned e_j); a "
                             "non-default one adds _{option} to the output name")
    args = parser.parse_args()
    STAB_DIR = STAB_DIR.with_name(f"{STAB_DIR.name}{args.tag}")
    OUT = OUT.with_name(f"{OUT.name}{args.tag}")
    if args.option != OPTION:
        OUT = OUT.with_name(f"{OUT.name}_{args.option}")
    imp, repro, co = load(args.option)
    fig, axes = plt.subplot_mosaic([["a", "b", "c"]], figsize=(18, 5.5), layout="constrained",
                                   width_ratios=[1, 1, 1.15])
    offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)

    ax = axes["a"]
    boxes(ax, imp, "spearman", show_points=False)
    ax.set_ylabel("Spearman correlation", fontsize=xylabelsize)
    ax.set_title("Correlation of encoding similarities\nwith feature importances",
                 fontsize=titlefontsize)
    panel_label(ax, "a", offset)

    ax = axes["b"]
    boxes(ax, repro, "rho", show_points=False)
    ax.set_ylabel("Spearman correlation", fontsize=xylabelsize)
    ax.set_title("Reproducibility of target\nsimilarities across seeds",
                 fontsize=titlefontsize)
    panel_label(ax, "b", offset)

    ax = axes["c"]
    ax.barh(co["name"], co["frac_runs_with_aplasia"], color=PALETTE["Aplasia"], height=0.7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of seeds in Aplasia cluster", fontsize=xylabelsize)
    ax.set_title(f"Co-clustering with Aplasia\nacross seeds (encoding size {COCLUSTER_DIM})",
                 fontsize=titlefontsize)
    ax.tick_params(axis="x", labelsize=ticklabelsize)
    ax.tick_params(axis="y", labelsize=ticklabelsize * 0.75)
    panel_label(ax, "c", offset)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[t]) for t in HUE_ORDER]
    fig.legend(handles, HUE_ORDER, loc="outside lower center", ncol=2, fontsize=legendfontsize,
               frameon=False)
    for ax in axes.values():
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT.parent / (OUT.name + ".pdf"))
    fig.savefig(OUT.parent / (OUT.name + ".png"), dpi=150)
    print(f"Wrote {OUT}.pdf")


if __name__ == "__main__":
    main()
