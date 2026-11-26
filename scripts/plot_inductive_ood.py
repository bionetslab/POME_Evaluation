"""Manuscript figure: are POME's inductive embeddings of unseen samples in-distribution?

One panel per dataset. For every train/test split, the Rosenbaum cross-match z-score (scXmatch)
of the held-out test samples against the training samples, in four representations:

  Raw values        binary sample x value indicators (model-free; shift already in the data)
  Spectral          spectral coordinates, test samples by Nystrom extension (model-free)
  POME (published)  transform() embeddings, published configuration
  POME (stable)     transform() embeddings, adopted stable_k8 configuration

z ~ 0: test and training samples are indistinguishable; z < 0: shifted. Points = splits,
bar = mean over splits; d = 32 and d = 64 side by side (raw values have no dimension).
The Nystrom-initialized variant of stable_k8 is not shown (identical within noise).

Inputs:  output/inductive_ood/ood_crossmatch_results.csv (evaluate_inductive_ood.py)
Outputs: output/inductive_ood/inductive_ood_crossmatch.{pdf,png}

    container/run.sh scripts/plot_inductive_ood.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "output" / "inductive_ood"

GROUPS = [  # (config, label, color)
    ("raw_onehot", "Raw\nvalues", "#8c8c8c"),
    ("spectral", "Spectral\ncoords.", "#8c8c8c"),
    ("default", "POME\npublished", "#eb6834"),
    ("stable_k8", "POME\nstable", "#2a78d6"),
]
DATASETS = [("hancock", "HANCOCK"), ("luad_noleak", "TCGA-LUAD"), ("mimic", "MIMIC")]
DIMS = {32: ("o", -0.17), 64: ("s", 0.17)}
INK, MUTED = "#222222", "#6b6b6b"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=OUT_ROOT / "ood_crossmatch_results.csv")
    p.add_argument("--output", type=Path, default=OUT_ROOT / "inductive_ood_crossmatch")
    args = p.parse_args()

    res = pd.read_csv(args.results)
    res = res[res.comparison == "main"]

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (ds, title) in zip(axes, DATASETS):
        ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
        for gi, (cfg, _, color) in enumerate(GROUPS):
            sub = res[(res.dataset == ds) & (res.config == cfg)]
            for dim, g in sub.groupby("dim"):
                marker, off = DIMS.get(int(dim), ("D", 0.0))
                x = gi + off
                z = g.z_score.to_numpy()
                ax.bar(x, z.mean(), width=0.3, color=color, alpha=0.25, zorder=1)
                ax.hlines(z.mean(), x - 0.15, x + 0.15, color=color, lw=2, zorder=2)
                sig = g.p_value.to_numpy() < 0.05
                jit = rng.uniform(-0.07, 0.07, len(z))
                ax.scatter(x + jit[sig], z[sig], s=12, marker=marker, color=color,
                           edgecolor="white", linewidth=0.4, zorder=3)
                ax.scatter(x + jit[~sig], z[~sig], s=12, marker=marker, facecolor="white",
                           edgecolor=color, linewidth=0.8, zorder=3)
        ax.set_title(title, fontsize=9, color=INK, pad=12)
        ax.set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS], fontsize=7)
        ax.axvline(1.5, color=MUTED, lw=0.5, ls=":", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -3), max(hi, 2))
    axes[0].set_ylabel("Cross-match z, test vs. train\n(0 = in-distribution, < 0 = shifted)")
    for ax in axes:
        ax.text(0.5, 1.0, "model-free", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6.5, color=MUTED)
        ax.text(2.5, 1.0, "embeddings", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6.5, color=MUTED)
    handles = [Line2D([], [], ls="", marker="D", color=MUTED, label="raw (no d)"),
               Line2D([], [], ls="", marker="o", color=MUTED, label="d = 32"),
               Line2D([], [], ls="", marker="s", color=MUTED, label="d = 64"),
               Line2D([], [], ls="", marker="o", color=MUTED, label="p < 0.05"),
               Line2D([], [], ls="", marker="o", markerfacecolor="white",
                      markeredgecolor=MUTED, label="p ≥ 0.05")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.output}.{ext}", dpi=300)
    print(f"Saved figure to {args.output}.pdf / .png")


if __name__ == "__main__":
    main()
