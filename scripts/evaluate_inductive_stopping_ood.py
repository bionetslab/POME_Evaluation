"""Cross-match in-distribution test of the inductive-mode ablation embeddings vs training epochs.

Reads the train (transductive) / test (inductive) embedding pairs that
``generate_inductive_stopping_ablation.py`` saved for every grid fit and runs the Rosenbaum
cross-match test (scXmatch, squared Euclidean distances, exact matching up to ``--max-full``
samples) of train vs. test, as the ``main`` comparison of ``evaluate_inductive_ood.py``:
z ~ 0 / high p = unseen samples are embedded into the distribution of the training samples;
negative z = they are separable.

The current inductive mode (its RankMe criterion never fires with POME 1.4) trains the full 2000
epochs.

Runs in the scXmatch environment (see ``evaluate_inductive_ood.py`` for its setup):
    $WORK/envs/scxmatch/bin/python scripts/evaluate_inductive_stopping_ood.py
    $WORK/envs/scxmatch/bin/python scripts/evaluate_inductive_stopping_ood.py --dims 32 64 \\
        --epochs 100 500 1000 2000

Inputs:  output/inductive_stopping_ablation/{dataset}/dim_{D}/split_{NN}/emb_e{E}.npz
Output:  output/inductive_stopping_ablation/ood_crossmatch.csv (resumable, one row per test)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_inductive_ood import crossmatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "output" / "inductive_stopping_ablation"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--dims", type=int, nargs="+", default=[32])
    p.add_argument("--epochs", type=int, nargs="+", default=[100, 300, 500, 1000, 2000])
    p.add_argument("--max-full", type=int, default=10000)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    import anndata as ad
    import scxmatch

    out_path = args.output or args.root / "ood_crossmatch.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done_keys = set()
    if len(done):
        done_keys = {(r.dataset, int(r.dim), int(r.split), int(r.epochs))
                     for r in done.itertuples()}

    rows = []
    for f in sorted(args.root.glob("*/dim_*/split_*/emb_e*.npz")):
        dataset, dim = f.parts[-4], int(f.parts[-3][4:])
        split, epochs = int(f.parts[-2][6:]), int(f.stem[5:])
        if dim not in args.dims or epochs not in args.epochs:
            continue
        meta = dict(dataset=dataset, dim=dim, split=split, epochs=epochs)
        if (dataset, dim, split, epochs) in done_keys:
            continue
        z = np.load(f)
        ref = pd.DataFrame(z["train"].astype(np.float64), index=z["train_index"])
        test = pd.DataFrame(z["test"].astype(np.float64), index=z["test_index"])
        n = len(ref) + len(test)
        k = None if n <= args.max_full else args.k
        res = crossmatch(scxmatch, ad, ref, test, k)
        rows.append(dict(meta, n_reference=len(ref), n_test=len(test), k=k if k else "full",
                         p_value=res["p_value"], z_score=res["z_score"]))
        print(f"{dataset:8s} dim {dim:3d} split {split:02d} E {epochs:5d} "
              f"p={res['p_value']:.3g} z={res['z_score']:+.2f}", flush=True)
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(out_path, index=False)

    res = pd.read_csv(out_path)
    summary = res.groupby(["dataset", "dim", "epochs"]).agg(
        splits=("split", "nunique"), z_mean=("z_score", "mean"), z_sd=("z_score", "std"),
        sig=("p_value", lambda q: int((q < 0.05).sum())))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summary.round(2).to_string())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
