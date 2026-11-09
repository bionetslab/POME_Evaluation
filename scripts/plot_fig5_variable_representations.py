"""Revised variable-semantics figure (manuscript Sec. 2.6), replacing the best-run panels.

Uses the identifiable variable representation returned by POME >= 1.2 (``centroid_input``: mean of
each variable's normalized value-node inputs) on the 10 seeds x 3 encoding sizes produced by
generate_variable_representations.py and analyzed by analyze_variable_representation_stability.py.

    a  Spearman correlation between CatBoost feature importance and target-to-lab cosine
       similarity, one point per seed; k/10 = seeds with one-sided permutation p < 0.05.
    b  Cross-seed reproducibility: Spearman correlation of the target-to-lab cosine profiles
       between all 45 pairs of seeds.
    c  Cluster reproducibility: fraction of seeds in which a lab variable falls into the Leiden
       cluster of Aplasia (top labs, largest encoding size).

CPU only:  container/run.sh scripts/plot_fig5_variable_representations.py
Output:    output/variable_embeddings/variable_results_revised.pdf (+ .png)
"""

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
OPTION = "centroid_input"
COCLUSTER_DIM = 64
TOP_LABS = 12
TARGET_NAMES = {"Aplasia": "Aplasia", "NF": "Neutropenic Fever"}
HUE_ORDER = list(TARGET_NAMES.values())
PALETTE = dict(zip(HUE_ORDER, sns.color_palette("tab10", 2)))

labelfontsize, titlefontsize, ticklabelsize = 20, 19, 16
xylabelsize, legendfontsize = 18, 15


def load():
    imp = pd.read_csv(STAB_DIR / "importance_correlation.csv")
    imp = imp[imp["option"] == OPTION].replace({"target": TARGET_NAMES})

    prof = pd.read_csv(STAB_DIR / "target_profiles.csv", dtype={"lab": str})
    prof = prof[prof["option"] == OPTION]
    rows = []
    for (dim, target), grp in prof.groupby(["dim", "target"]):
        p = grp.pivot(index="lab", columns="run", values="cosine")
        for a, b in itertools.combinations(p.columns, 2):
            rows.append(dict(dim=dim, target=TARGET_NAMES[target], rho=spearmanr(p[a], p[b])[0]))
    repro = pd.DataFrame(rows)

    co = pd.read_csv(STAB_DIR / "aplasia_coclustering.csv", dtype={"lab": str})
    co = co[(co["option"] == OPTION) & (co["dim"] == COCLUSTER_DIM)]
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
    imp, repro, co = load()
    fig, axes = plt.subplot_mosaic([["a", "b", "c"]], figsize=(18, 5.5), layout="constrained",
                                   width_ratios=[1, 1, 1.15])
    offset = transforms.ScaledTranslation(-36 / 72, 27 / 72, fig.dpi_scale_trans)

    ax = axes["a"]
    boxes(ax, imp, "spearman", show_points=True)
    ax.set_ylabel("Spearman correlation", fontsize=xylabelsize)
    ax.set_title("Correlation of encoding similarities\nwith feature importances",
                 fontsize=titlefontsize)
    dims = sorted(imp["dim"].unique())
    top = imp["spearman"].max()
    for i, dim in enumerate(dims):
        for j, target in enumerate(HUE_ORDER):
            sub = imp[(imp["dim"] == dim) & (imp["target"] == target)]
            k = int((sub["perm_p"] < 0.05).sum())
            ax.text(i + (j - 0.5) * 0.4, top + 0.06, f"{k}/{len(sub)}", ha="center",
                    va="bottom", fontsize=ticklabelsize * 0.8, color="0.25")
    ax.set_ylim(top=top + 0.2)
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
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    print(f"Wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
