"""Embeddings for the in-distribution check of POME's inductive transform() (GPU part).

Question: do POME's inductive embeddings of unseen samples (transform()) land in the same
distribution as the transductive embeddings of the training samples?

  pome14_spectral   POME defaults (spectral input encoding; a new sample node gets a zero input
                    row, so its embedding comes from the value nodes it connects to)

Per dataset x split x dim, one fit on the training split (POME defaults, fixed ``--epochs``, no
CV-tuned stopping, seed ``--seed``) and three embeddings:

  train_transductive   get_embeddings() of the training samples (what the model was trained on)
  test_inductive       transform() of the held-out test samples
  train_inductive      transform() of the *training* samples (method-effect control: same samples,
                       only the embedding route differs)

The cross-match tests are run by ``evaluate_inductive_ood.py --root output/inductive_ood_encodings``
(scXmatch environment).

GPU (H200), one job per dataset:
    sbatch container/submit.sh scripts/generate_inductive_ood_embeddings.py --datasets hancock
CPU smoke test:
    container/run.sh scripts/generate_inductive_ood_embeddings.py --datasets luad --splits 0 \\
        --dims 16 --epochs 30 --device cpu --out-root <scratch>

Inputs:  data/train_test_splits/{dataset}/split_{NN}_{train,test}_pome.tsv
Outputs: output/inductive_ood_encodings/pome14_spectral/{dataset}/split_{NN}/dim_{D}/
         {train_transductive,test_inductive,train_inductive}.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
OUT_ROOT = PROJECT_ROOT / "output" / "inductive_ood_encodings"
CONFIG = "pome14_spectral"
NA_ENCODING = -99.0
KINDS = ("train_transductive", "test_inductive", "train_inductive")


def load_graph(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def standardize(emb: pd.DataFrame) -> pd.DataFrame:
    emb = emb.copy()
    emb.index = emb.index.astype(str)
    emb.columns = [f"dim_{i}" for i in range(emb.shape[1])]
    return emb


def run_one(dataset, split, dim, args, device):
    from pome.gnn_embedding import Embedder, make_deterministic

    out_dir = args.out_root / CONFIG / dataset / f"split_{split:02d}" / f"dim_{dim}"
    paths = {k: out_dir / f"{k}.csv" for k in KINDS}
    if all(p.exists() for p in paths.values()) and not args.overwrite:
        print(f"[skip] {CONFIG} {dataset} split {split:02d} dim {dim}", flush=True)
        return
    split_dir = SPLITS_ROOT / dataset
    train_df = load_graph(split_dir / f"split_{split:02d}_train_pome.tsv")
    test_df = load_graph(split_dir / f"split_{split:02d}_test_pome.tsv")

    t0 = time.perf_counter()
    make_deterministic(args.seed)
    emb = Embedder(embedding_dimension=dim, epochs=args.epochs, bins_per_continuous=15,
                   discretization_type="z", na_encoding=NA_ENCODING, device=device,
                   inductive=False)
    emb.fit(train_df)
    out = {"train_transductive": emb.get_embeddings()[0],
           "test_inductive": emb.transform(test_df),
           "train_inductive": emb.transform(train_df)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        standardize(df).to_csv(paths[k])
    print(f"[ok]   {CONFIG} {dataset} split {split:02d} dim {dim}  "
          f"train {out['train_transductive'].shape} test {out['test_inductive'].shape}  "
          f"{time.perf_counter() - t0:.1f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=["hancock", "luad", "mimic"])
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in args.datasets:
        for split in args.splits:
            for dim in args.dims:
                run_one(dataset, split, dim, args, device)


if __name__ == "__main__":
    main()
