"""Combine the linear-probing scores of a re-trained POME variant with the published UMAP rows.

The UMAP embeddings do not change when POME is re-trained (e.g. ``--pome-config stable``), and
the MIMIC UMAP embeddings are not stored in this repo, so the UMAP comparison rows are taken
from the published results file instead of being re-probed. The output has the same schema as
``inductive_linear_probing_results.csv`` and can be plotted with
``plot_inductive_linear_probing.py --results <output>``.

    container/run.sh scripts/linear_probe_inductive_embeddings.py --methods pome \\
        --modes combined --dims 16 32 64 --runs 0 1 2 3 4 \\
        --emb-root output/inductive_stable \\
        --output output/linear_probing/inductive_linear_probing_results_stable_pome.csv
    container/run.sh scripts/merge_probing_variant.py \\
        --pome output/linear_probing/inductive_linear_probing_results_stable_pome.csv \\
        --output output/linear_probing/inductive_linear_probing_results_stable.csv
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = PROJECT_ROOT / "output" / "linear_probing" / "inductive_linear_probing_results.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pome", type=Path, required=True, help="probing results of the POME variant")
    p.add_argument("--reference", type=Path, default=PUBLISHED,
                   help="published results providing the UMAP rows")
    p.add_argument("--mode", default="combined", help="variable-type mode to keep")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    pome = pd.read_csv(args.pome)
    pome = pome[(pome["method"] == "POME") & (pome["mode"] == args.mode)]
    ref = pd.read_csv(args.reference)
    umap = ref[(ref["method"] == "UMAP") & (ref["mode"] == args.mode)]
    keys = ["dataset", "dim"]
    umap = umap.merge(pome[keys].drop_duplicates(), on=keys)  # same datasets x dims only
    out = pd.concat([pome, umap], ignore_index=True)
    out.to_csv(args.output, index=False)
    print(out.groupby(["dataset", "method"]).agg(
        dims=("dim", lambda x: sorted(x.unique())), splits=("split", "nunique"),
        runs=("run", "nunique"), rows=("dim", "size")).to_string())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
