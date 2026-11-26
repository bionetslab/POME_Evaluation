"""Downstream quality of POME's inductive mode vs. training length (GPU part).

How does the downstream generalization of POME's inductive embeddings depend on the number of
training epochs, up to the default inductive budget of 2000 epochs? This answers the reviewer
request to validate the (former) early-stopping criterion against downstream performance: POME
>= 1.5 trains to the fixed budget without early stopping.

Per dataset x split x dim (one seed) and every training length E in ``--epoch-grid``, POME is fit
on all training samples (``Embedder(inductive=True, epochs=E)``, cosine lr over E) and writes

  grid_probes.csv  linear-probe average precision per binary target (logreg fit on the
                   transductive training embeddings, scored on the inductive test embeddings;
                   identical to ``linear_probe_inductive_embeddings.py``)
  emb_e{E}.npz     the train (transductive) / test (inductive) embeddings of each fit, for the
                   cross-match in-distribution test (``evaluate_inductive_stopping_ood.py``)

With ``--cv-trace`` it additionally records cv_trace.csv, the label-free signals of the former
sample-holdout CV (RankMe gap, held-out link loss / AUC; 3 folds, constant lr, every 10 epochs)
over 2000 epochs, for the stopping-criterion analyses. That needs the POME branch
``inductive-revision`` (``Embedder.cv_trace``), whose fixed-epoch fits are bit-identical to POME
main >= 1.5 (checked on LUAD split 0, d32).

Run with PYTHONHASHSEED=0 (POME fits differ across processes otherwise):

    sbatch --export=POME_REPO=$HOME/Projects/POME.git,PYTHONHASHSEED=0 \\
        container/submit.sh scripts/generate_inductive_stopping_ablation.py --datasets hancock
CPU smoke test:
    APPTAINERENV_PYTHONHASHSEED=0 container/run.sh scripts/generate_inductive_stopping_ablation.py \\
        --datasets luad_noleak --splits 0 --dims 16 --epoch-grid 20 40 --device cpu --out-root <scratch>

Figures / statistics: ``analyze_inductive_epoch_probes.py``.

Inputs:  data/train_test_splits/{dataset}/split_{NN}_{train,test}_pome.tsv,
         data/input_datasets (probe targets)
Outputs: output/inductive_stopping_ablation/{dataset}/dim_{D}/split_{NN}/
         {grid_probes.csv, emb_e{EEEE}.npz[, cv_trace.csv]}
"""
import argparse
import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe_inductive_embeddings import load_binary_targets, probe_one  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
OUT_ROOT = PROJECT_ROOT / "output" / "inductive_stopping_ablation"
NA_ENCODING = -99.0
BUDGET = 2000
EPOCH_GRID = [25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000]


def load_graph(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def embedder(dim, epochs, device, **kwargs):
    """POME's inductive mode with E training epochs (POME defaults otherwise)."""
    from pome.gnn_embedding import Embedder
    return Embedder(embedding_dimension=dim, epochs=epochs, bins_per_continuous=15,
                    discretization_type="z", na_encoding=NA_ENCODING, device=device,
                    inductive=True, **kwargs)


def run_one(dataset, split, dim, targets, args, device):
    """One fit per grid epoch count (+ optional CV trace).

    Incremental: an epoch count whose probes are already in grid_probes.csv (and whose embeddings
    exist) is not refit, and probes are written after every fit, so interrupted jobs resume.
    """
    from pome.gnn_embedding import make_deterministic

    out_dir = args.out_root / dataset / f"dim_{dim}" / f"split_{split:02d}"
    trace_path, probe_path = out_dir / "cv_trace.csv", out_dir / "grid_probes.csv"
    old = pd.read_csv(probe_path) if probe_path.exists() and not args.overwrite else None
    done = set() if old is None else {
        int(e) for e in old["epochs"].unique() if (out_dir / f"emb_e{int(e):04d}.npz").exists()}
    todo = [e for e in args.epoch_grid if e not in done]
    need_trace = args.cv_trace and (args.overwrite or not trace_path.exists())
    if not todo and not need_trace:
        print(f"[skip] {dataset} split {split:02d} dim {dim}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = SPLITS_ROOT / dataset
    train_df = load_graph(split_dir / f"split_{split:02d}_train_pome.tsv")
    test_df = load_graph(split_dir / f"split_{split:02d}_test_pome.tsv")
    meta = dict(dataset=dataset, split=split, dim=dim, seed=args.seed)

    t0 = time.perf_counter()
    if need_trace:
        make_deterministic(args.seed)
        with contextlib.redirect_stdout(io.StringIO()):  # POME prints the loss every 10 epochs
            trace = embedder(dim, BUDGET, device, stopping_criterion="rankme_gap").cv_trace(train_df)
        trace.assign(**meta).to_csv(trace_path, index=False)

    probes = old[old["epochs"].isin(done)] if old is not None else pd.DataFrame()
    for epochs in todo:
        make_deterministic(args.seed)
        emb = embedder(dim, epochs, device, **args.fit_kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            emb.fit(train_df.copy())
        train_emb = emb.get_embeddings()[0]
        test_emb = emb.transform(test_df.copy())
        for z in (train_emb, test_emb):
            z.index = z.index.astype(str)
        np.savez_compressed(out_dir / f"emb_e{epochs:04d}.npz",
                            train=train_emb.to_numpy(np.float32),
                            test=test_emb.to_numpy(np.float32),
                            train_index=train_emb.index.to_numpy(str),
                            test_index=test_emb.index.to_numpy(str))
        rows = []
        for target, y in targets.items():
            res = probe_one(train_emb, test_emb, y)   # logreg_std
            if res is not None:
                rows.append(dict(meta, epochs=epochs, target=target, train_link_auc=emb._auc,
                                 **res))
        probes = pd.concat([probes, pd.DataFrame(rows)], ignore_index=True)
        probes.sort_values(["epochs", "target"]).to_csv(probe_path, index=False)
    ap = probes.groupby("epochs")["average_precision"].mean()
    print(f"[ok]   {dataset} split {split:02d} dim {dim}  new fits {todo}  "
          f"total {time.perf_counter() - t0:.0f}s  "
          f"mean AP by epochs: " + " ".join(f"{e}:{v:.3f}" for e, v in ap.items()), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=["hancock", "luad_noleak", "mimic"])
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--epoch-grid", type=int, nargs="+", default=EPOCH_GRID)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--cv-trace", action="store_true",
                   help="also record the former CV stopping signals (POME branch inductive-revision)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    import torch
    from pome.gnn_embedding import Embedder
    has_criterion = "stopping_criterion" in Embedder.__init__.__code__.co_varnames
    if args.cv_trace and not has_criterion:
        sys.exit("--cv-trace needs Embedder.cv_trace: set POME_REPO to the inductive-revision branch")
    # On the inductive-revision branch, disable its stopping rule (= POME main >= 1.5).
    args.fit_kwargs = {"stopping_criterion": "none"} if has_criterion else {}
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in args.datasets:
        targets = load_binary_targets(dataset)
        for dim in args.dims:
            for split in args.splits:
                run_one(dataset, split, dim, targets, args, device)


if __name__ == "__main__":
    main()
