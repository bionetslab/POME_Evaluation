"""Rosenbaum cross-match (scXmatch) test of POME's inductive embeddings against the transductive ones.

Reads the embeddings written by ``generate_inductive_ood_embeddings.py`` and, per POME config x
dataset x split x dim, runs two scXmatch tests (squared Euclidean distances):

  main      train_transductive (reference) vs test_inductive   -- are unseen samples embedded
            into the distribution of the training samples?
  control   train_transductive (reference) vs train_inductive  -- same samples, only the embedding
            route differs: isolates the method effect of transform() from genuine train/test shift

A high p-value / z >= 0 means the two groups are indistinguishable (in-distribution); a low
p-value with negative z means they are separable (fewer cross-group matches than expected).
For spectral configs with Nystrom-initialized new samples (``*_nystrom.csv``) the same two
tests are run as ``main_nystrom`` / ``control_nystrom``.
For the control tests it also reports ``cos_same``, the mean cosine similarity between a training
sample's transductive and inductive embedding (1 = transform() reproduces the training embedding).

Exact matching on the full distance graph (``k = None``) is used up to ``--max-full`` samples
per test (default 10000, covers MIMIC: ~20 s at 7000 samples), otherwise scXmatch's k-NN graph
with ``--k`` neighbours; ``k`` is recorded per test.

Runs in a separate scXmatch conda environment (graph-tool >= 2.92 is conda-only and not in the
POME container). One-time setup on the cluster front end (conda's CPU detection fails there, so
the microarchitecture is overridden; package cache outside the inode-limited workspace):
    export PATH=/apps/python/3.12-conda/bin:$PATH CONDA_OVERRIDE_ARCHSPEC=x86_64_v3
    CONDA_PKGS_DIRS=<local tmp dir> conda create -p $WORK/envs/scxmatch \
        -c conda-forge -c bioconda scxmatch "graph-tool>=2.92"
Run:
    $WORK/envs/scxmatch/bin/python scripts/evaluate_inductive_ood.py

Outputs: output/inductive_ood/ood_crossmatch_results.csv (per test) and a printed summary.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "output" / "inductive_ood"


def crossmatch(scxmatch, ad, ref, test, k):
    names = [f"ref::{s}" for s in ref.index] + [f"test::{s}" for s in test.index]
    obs = pd.DataFrame({"group": ["ref"] * len(ref) + ["test"] * len(test)}, index=names)
    adata = ad.AnnData(X=np.vstack([ref.to_numpy(), test.to_numpy()]).astype(np.float64), obs=obs)
    return dict(scxmatch.test(adata, group_by="group", test_group="test", reference="ref",
                              metric="sqeuclidean", k=k))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=OUT_ROOT)
    p.add_argument("--scxmatch-src", type=Path, default=None,
                   help="source checkout of scxmatch to put on sys.path (if not installed)")
    p.add_argument("--max-full", type=int, default=10000,
                   help="largest test size (samples) using the exact full distance graph")
    p.add_argument("--k", type=int, default=100, help="k-NN graph size above --max-full")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.scxmatch_src:
        sys.path.insert(0, str(args.scxmatch_src))
    import anndata as ad
    import scxmatch

    out_path = args.output or args.root / "ood_crossmatch_results.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    key = ["config", "dataset", "split", "dim", "comparison"]
    done_keys = set(map(tuple, done[key].astype(str).to_numpy())) if len(done) else set()

    rows = []
    for d in sorted(args.root.glob("*/*/split_*/dim_*")):
        files = {k: d / f"{k}.csv" for k in ("train_transductive", "test_inductive",
                                              "train_inductive")}
        if not all(f.exists() for f in files.values()):
            continue
        config, dataset = d.parts[-4], d.parts[-3]
        split, dim = int(d.parts[-2][6:]), int(d.parts[-1][4:])
        emb = {k: pd.read_csv(f, index_col=0) for k, f in files.items()}
        tests = [("main", emb["test_inductive"]), ("control", emb["train_inductive"])]
        for kind, comparison in (("test_inductive_nystrom", "main_nystrom"),
                                 ("train_inductive_nystrom", "control_nystrom")):
            if (d / f"{kind}.csv").exists():
                tests.append((comparison, pd.read_csv(d / f"{kind}.csv", index_col=0)))
        for comparison, test in tests:
            meta = dict(config=config, dataset=dataset, split=split, dim=dim,
                        comparison=comparison)
            if tuple(str(meta[c]) for c in key) in done_keys:
                continue
            ref = emb["train_transductive"]
            n = len(ref) + len(test)
            k = None if n <= args.max_full else args.k
            res = crossmatch(scxmatch, ad, ref, test, k)
            row = dict(meta, n_reference=len(ref), n_test=len(test), k=k if k else "full",
                       p_value=res["p_value"], z_score=res["z_score"],
                       coverage=res["coverage"],
                       effect_strength_ratio=res["effect_strength_ratio"])
            if comparison.startswith("control"):
                a = ref.loc[test.index].to_numpy()
                b = test.to_numpy()
                cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
                row["cos_same"] = float(np.mean(cos))
            rows.append(row)
            print(f"{config:10s} {dataset:8s} split {split:02d} dim {dim:3d} {comparison:15s} "
                  f"p={row['p_value']:.3g} z={row['z_score']:+.2f} k={row['k']}"
                  + (f" cos_same={row['cos_same']:.3f}" if "cos_same" in row else ""),
                  flush=True)
            # Persist incrementally (resumable).
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(out_path, index=False)

    res = pd.read_csv(out_path)
    summary = res.groupby(["dataset", "dim", "comparison", "config"]).agg(
        splits=("split", "nunique"), z_mean=("z_score", "mean"), z_sd=("z_score", "std"),
        p_median=("p_value", "median"), sig=("p_value", lambda p: int((p < 0.05).sum())),
        cos_same=("cos_same", "mean") if "cos_same" in res else ("z_score", "size"))
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
