"""Combine the linear-probing scores of re-trained POME embeddings with the published UMAP rows.

The UMAP embeddings do not change when POME is re-trained (e.g. with POME 1.4), and
the MIMIC UMAP embeddings are not stored in this repo, so the UMAP comparison rows are taken
from the published results file instead of being re-probed. The output has the same schema as
``inductive_linear_probing_results.csv`` and can be plotted with
``plot_inductive_linear_probing.py --results <output>``.

    container/run.sh scripts/linear_probe_inductive_embeddings.py --methods pome \\
        --modes combined --dims 16 32 64 --runs 0 1 2 3 4 \\
        --emb-root output/inductive_pome14 \\
        --output output/linear_probing/inductive_linear_probing_results_pome14_pome.csv
    container/run.sh scripts/merge_probing_variant.py \\
        --pome output/linear_probing/inductive_linear_probing_results_pome14_pome.csv \\
        --output output/linear_probing/inductive_linear_probing_results_pome14.csv

``--competitors`` appends the probing rows of further (dataset, method) pairs that are not yet
present, restricted to the same datasets x dims as the POME rows. These are the tabular
competitors HIVAE / SAINT / MFAE, probed from ``output/inductive_competitors``, and, for datasets
that the published file lacks (``luad_noleak``), freshly probed UMAP rows:

    container/run.sh scripts/merge_probing_variant.py \\
        --pome output/linear_probing/inductive_linear_probing_results_pome14_pome.csv \\
        --competitors output/linear_probing/inductive_linear_probing_results_competitors.csv \\
        --output output/linear_probing/inductive_linear_probing_results_pome14_competitors.csv

The published file holds logistic-regression scores only. For the non-linear (``--probe
xgboost``) results, pass ``--reference none`` and give freshly probed UMAP rows via
``--competitors``. All inputs must come from the same probe (``probe`` column; files without it
are logistic regression); mixing probes is an error.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = PROJECT_ROOT / "output" / "linear_probing" / "inductive_linear_probing_results.csv"


def probe_of(frame, name):
    """The single probe a results table was made with (no ``probe`` column = logreg)."""
    probes = set(frame["probe"].fillna("logreg")) if "probe" in frame else {"logreg"}
    if len(probes) != 1:
        raise SystemExit(f"{name} mixes probes: {sorted(probes)}")
    return probes.pop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pome", type=Path, nargs="+", required=True,
                   help="probing results of the POME variant (several files are concatenated)")
    p.add_argument("--reference", default=str(PUBLISHED),
                   help="published results providing the UMAP rows "
                        "(logistic regression; 'none' to skip)")
    p.add_argument("--mode", default="combined", help="variable-type mode to keep")
    p.add_argument("--competitors", type=Path, nargs="+", default=[],
                   help="probing results whose (dataset, method) pairs not yet present "
                        "are appended")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    pome = pd.concat([pd.read_csv(f) for f in args.pome], ignore_index=True)
    pome = pome[(pome["method"] == "POME") & (pome["mode"] == args.mode)]
    probe = probe_of(pome, "--pome")
    keys = ["dataset", "dim"]
    parts = [pome]
    if args.reference != "none":
        ref = pd.read_csv(args.reference)
        if probe_of(ref, args.reference) != probe:
            raise SystemExit(f"--reference is {probe_of(ref, args.reference)}-probed, POME is "
                             f"{probe}-probed; use --reference none")
        umap = ref[(ref["method"] == "UMAP") & (ref["mode"] == args.mode)]
        parts.append(umap.merge(pome[keys].drop_duplicates(), on=keys))  # same datasets x dims
    for path in args.competitors:
        comp = pd.read_csv(path)
        if probe_of(comp, path) != probe:
            raise SystemExit(f"{path} is {probe_of(comp, path)}-probed, POME is {probe}-probed")
        comp = comp[comp["mode"] == args.mode].merge(pome[keys].drop_duplicates(), on=keys)
        present = pd.concat(parts)[["dataset", "method"]].drop_duplicates()
        new = comp.merge(present, on=["dataset", "method"], how="left", indicator=True)
        parts.append(new[new["_merge"] == "left_only"].drop(columns="_merge"))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(args.output, index=False)
    print(out.groupby(["dataset", "method"]).agg(
        dims=("dim", lambda x: sorted(x.unique())), splits=("split", "nunique"),
        runs=("run", "nunique"), rows=("dim", "size")).to_string())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
