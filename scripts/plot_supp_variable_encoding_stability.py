"""Supplementary figure: between-run stability of MIMIC variable encodings, original vs. revised.

Compares the raw learned variable encodings e_j (POME <= 1.1, used in the original Sec. 2.6
analysis) with the identifiable value-node centroids returned by POME >= 1.2 (``centroid_input``),
on the 10 seeds x 3 encoding sizes written by generate_variable_representations.py.

Stability of one pair of runs = Spearman correlation between their variable-variable cosine
similarity matrices (upper triangle, all 102 variables incl. the two targets). Cosines are
rotation invariant, so runs are compared without alignment. One box = all 45 pairs of seeds.

CPU only:  container/run.sh scripts/plot_supp_variable_encoding_stability.py
Output:    output/variable_embeddings/supp_variable_encoding_stability.{pdf,png,csv}
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_ROOT = PROJECT_ROOT / "data" / "embeddings" / "MIMIC" / "variable_representations"
OUT = PROJECT_ROOT / "output" / "variable_embeddings" / "supp_variable_encoding_stability"
DIMS = (16, 32, 64)
OPTIONS = {"raw": "Original (raw $e_j$)", "centroid_input": "Revised (value-node centroids)"}
PALETTE = dict(zip(OPTIONS.values(), ["#9e9e9e", sns.color_palette("tab10")[0]]))


def similarity_vectors(dim, option):
    """Upper-triangle cosine similarities between all variables, per run."""
    sims, variables = {}, None
    for run_dir in sorted((IN_ROOT / f"dim_{dim}").glob("run_*")):
        df = pd.read_csv(run_dir / f"{option}.tsv", sep="\t", index_col=0)
        df.index = df.index.astype(str)
        variables = variables if variables is not None else list(df.index)
        X = df.loc[variables].to_numpy()
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        sims[int(run_dir.name[4:])] = (X @ X.T)[np.triu_indices(len(variables), k=1)]
    return sims


def main():
    rows = []
    for dim, option in itertools.product(DIMS, OPTIONS):
        sims = similarity_vectors(dim, option)
        rows += [dict(dim=dim, encoding=OPTIONS[option], seed_a=a, seed_b=b,
                      spearman=spearmanr(sims[a], sims[b])[0])
                 for a, b in itertools.combinations(sorted(sims), 2)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    print(df.groupby(["encoding", "dim"])["spearman"].median().unstack().round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    sns.boxplot(data=df, x="dim", y="spearman", hue="encoding", hue_order=list(OPTIONS.values()),
                palette=PALETTE, gap=0.2, fliersize=3, ax=ax)
    ax.axhline(0, color="0.5", lw=1, ls="--", zorder=0)
    ax.set_xlabel("Encoding size", fontsize=14)
    ax.set_ylabel("Between-run correlation of\nvariable encoding similarities", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title=None, fontsize=11, frameon=False, loc="upper left")
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    print(f"Wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
