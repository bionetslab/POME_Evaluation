"""Repeated-seed POME fits for the training-noise ablation (LR decay, negatives, init, anchors).

Question: how much of POME's run-to-run variation in downstream results comes from the
optimizer never settling (constant Adam lr) and from the stochastic link-prediction loss (one
random negative per positive pair per epoch) -- and how much from the random initialization
of the encoding table? POME >= 1.3 exposes all three as options:

    lr_schedule = "constant" | "cosine"     cosine decays lr -> 0 over the trained epochs
    negatives   = "sampled"  | "all"        "all" scores every alternative value of each pair
                                            (exact expectation of the sampled loss)
    init        = "random"   | "spectral"   spectral: encoding table initialized from the SVD of
                                            the degree-normalized sample x value biadjacency
    anchor_weight = w >= 0                  penalty w * mean_rows ||W - W_init||^2 keeping the
                                            table near its spectral init (inf = frozen table);
                                            ``--anchor-weights`` adds configs ``anchor{w}_*``
    gram_anchor_weight = mu >= 0            penalty mu * mean_ij (z_i.z_j - x_i.x_j)^2 pulling the
                                            sample-sample cosine geometry of the GAT output to the
                                            spectral sample embedding; ``--gram-weights`` adds
                                            configs ``gram{mu}_*`` (free table) or, combined with
                                            ``--anchor-weights``, ``anchor{w}_gram{mu}_*``
    gram_reference_components = k         number of leading spectral components in the Gram
                                            reference (``--gram-ref-components``, infix ``k{k}``)

This script fits the 2 x 2 grid of the first two options for each init (configs ``baseline``,
``lr_decay``, ``all_negatives``, ``both`` and their ``spectral_*`` counterparts) for ``--seeds`` seeds on one train/test split
per dataset (same split for all runs, so only training randomness varies) and stores, per run,

  * the transductive train-sample embeddings and the inductive test-sample embeddings of a fit
    on the full training split  (used for embedding agreement, clustering and linear probing);
  * the imputations of a second fit on the training split with a fixed 10 % of its observed
    cells masked (the same mask for every run; used for imputation variance).

Epochs are fixed (``inductive=False``) so that the CV-tuned stopping epoch -- itself a variance
source -- does not mix into the comparison; ``--epochs`` lists the budgets to test.

GPU (H200) job:
    sbatch container/submit.sh --stage-out output/training_noise \\
        scripts/generate_training_noise_runs.py
CPU smoke test:
    container/run.sh scripts/generate_training_noise_runs.py --datasets luad --seeds 2 --epochs 100

Inputs:  data/train_test_splits/{dataset}/split_{NN}_{train,test}_pome.tsv
Outputs: output/training_noise/{dataset}/imputation_mask.csv           (masked cells + truth)
         output/training_noise/{dataset}/epochs_{E}/{config}/seed_{SS}_{train,test}.csv
         output/training_noise/{dataset}/epochs_{E}/{config}/seed_{SS}_imputed.csv
         output/training_noise/runs.csv                                 (run log, wall clock)
Analysis: scripts/analyze_training_noise_variance.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_ROOT = PROJECT_ROOT / "data" / "train_test_splits"
OUT_ROOT = PROJECT_ROOT / "output" / "training_noise"

NA_ENCODING = -99.0
TYPE_COL = "type"
LEGACY_TRAINING = dict(init="random", anchor_weight=0.0, gram_anchor_weight=0.0,
                       gram_reference_components=None, lr_schedule="constant",
                       negatives="sampled")
CONFIGS = {
    "baseline": dict(lr_schedule="constant", negatives="sampled"),
    "lr_decay": dict(lr_schedule="cosine", negatives="sampled"),
    "all_negatives": dict(lr_schedule="constant", negatives="all"),
    "both": dict(lr_schedule="cosine", negatives="all"),
}
# Same grid with the spectral initialization of the encoding table (POME init="spectral").
BASE_CONFIGS = dict(CONFIGS)
CONFIGS.update({f"spectral_{k}": dict(v, init="spectral") for k, v in BASE_CONFIGS.items()})


def anchor_configs(weights):
    """Spectral-init configs with the table anchored to its init (POME anchor_weight)."""
    return {f"anchor{w:g}_{k}": dict(v, init="spectral", anchor_weight=w)
            for w in weights for k, v in BASE_CONFIGS.items()}


def gram_configs(gram_weights, anchor_weights, ref_components=(), bases=None):
    """Spectral-init configs with the output Gram anchor (POME gram_anchor_weight), optionally
    combined with table anchoring and a top-k spectral reference (gram_reference_components;
    config infix ``k{k}``, no infix = all components)."""
    out = {}
    bases = {b: BASE_CONFIGS[b] for b in (bases or BASE_CONFIGS)}
    for mu in gram_weights:
        for w in anchor_weights or [None]:
            for k in ref_components or [None]:
                prefix = f"gram{mu:g}" if w is None else f"anchor{w:g}_gram{mu:g}"
                extra = {} if w is None else dict(anchor_weight=w)
                if k is not None:
                    prefix += f"_k{k}"
                    extra["gram_reference_components"] = k
                out.update({f"{prefix}_{b}": dict(v, init="spectral", gram_anchor_weight=mu,
                                                  **extra) for b, v in bases.items()})
    return out
MASK_FRACTION = 0.10
MASK_SEED = 0


def load_graph(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def make_mask(train_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Fixed random 10 % of the observed (non-NA) training cells, with their true values.

    Variables keep at least two observed cells so that every masked variable remains in the
    graph. Stored once per dataset and reused by every run.
    """
    path = out_dir / "imputation_mask.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"variable": str, "sample": str})
    values = train_df.drop(columns=[TYPE_COL])
    rng = np.random.default_rng(MASK_SEED)
    rows = []
    for var in values.index:
        observed = values.columns[values.loc[var] != NA_ENCODING]
        k = min(int(round(MASK_FRACTION * len(observed))), max(len(observed) - 2, 0))
        for s in rng.choice(observed, size=k, replace=False):
            rows.append({"variable": var, "sample": str(s), "type": train_df.loc[var, TYPE_COL],
                         "true": values.loc[var, s]})
    mask = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask.to_csv(path, index=False)
    return mask


def apply_mask(train_df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    masked = train_df.copy()
    masked.columns = masked.columns.astype(str)
    for var, s in zip(mask["variable"], mask["sample"]):
        masked.loc[var, s] = NA_ENCODING
    return masked


def standardize_columns(emb: pd.DataFrame) -> pd.DataFrame:
    emb = emb.copy()
    emb.columns = [f"dim_{i}" for i in range(emb.shape[1])]
    return emb


def run_one(dataset, split, dim, epochs, config, seed, train_df, test_df, mask, device, out_dir):
    from pome.gnn_embedding import Embedder, make_deterministic

    paths = {k: out_dir / f"seed_{seed:02d}_{k}.csv" for k in ("train", "test", "imputed")}
    if all(p.exists() for p in paths.values()):
        print(f"[skip] {dataset} epochs {epochs} {config} seed {seed}", flush=True)
        return None
    # Configs name only the options they change; everything else is the POME <= 1.2 behaviour
    # (POME >= 1.3 defaults to the adopted stable_k8 settings, so the legacy values are explicit).
    params = dict(embedding_dimension=dim, epochs=epochs, na_encoding=NA_ENCODING,
                  device=device, inductive=False,
                  **{**LEGACY_TRAINING, **CONFIGS[config]})
    t0 = time.perf_counter()

    # Fit 1: full training split -> train (transductive) and test (inductive) embeddings.
    make_deterministic(seed)
    emb = Embedder(**params)
    emb.fit(train_df)
    train_emb, *_ = emb.get_embeddings()
    test_emb = emb.transform(test_df)

    # Fit 2: masked training split -> imputations of the masked cells.
    make_deterministic(seed)
    imp = Embedder(enable_imputation=True, **params)
    imp.fit(apply_mask(train_df, mask))
    imputed = imp.impute_all(na_value=NA_ENCODING)
    imputed.columns = imputed.columns.astype(str)
    imputed_cells = mask.assign(imputed=[imputed.loc[v, s] for v, s in
                                         zip(mask["variable"], mask["sample"])])

    out_dir.mkdir(parents=True, exist_ok=True)
    standardize_columns(train_emb).to_csv(paths["train"])
    standardize_columns(test_emb).to_csv(paths["test"])
    imputed_cells.to_csv(paths["imputed"], index=False)
    seconds = time.perf_counter() - t0
    print(f"[ok]   {dataset} epochs {epochs} {config} seed {seed}  {seconds:.1f}s", flush=True)
    return {"dataset": dataset, "split": split, "dim": dim, "epochs": epochs, "config": config,
            "seed": seed, "seconds": round(seconds, 1), "device": device}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=["hancock", "luad"])
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--epochs", type=int, nargs="+", default=[500, 2000])
    p.add_argument("--configs", nargs="+", default=None,
                   help=f"subset of {list(CONFIGS)} (default: all of them, unless "
                        "--anchor-weights is given)")
    p.add_argument("--anchor-weights", type=float, nargs="+", default=[],
                   help="also run the 4 base configs with spectral init + anchoring at these "
                        "weights (config anchor{w}_{base}; 'inf' freezes the table)")
    p.add_argument("--gram-weights", type=float, nargs="+", default=[],
                   help="run the 4 base configs with spectral init + output Gram anchor at these "
                        "weights (config gram{mu}_{base}, or anchor{w}_gram{mu}_{base} together "
                        "with --anchor-weights)")
    p.add_argument("--gram-ref-components", type=int, nargs="+", default=[],
                   help="with --gram-weights: top-k spectral Gram references to test (config "
                        "infix k{k}; POME gram_reference_components); default all components")
    p.add_argument("--bases", nargs="+", default=None, choices=list(BASE_CONFIGS),
                   help="restrict the anchor/gram configs to these base configs (default: all 4)")
    p.add_argument("--extra-configs", nargs="+", default=[],
                   help="further named configs to run alongside the generated ones, e.g. "
                        "baseline anchorinf_gram10_lr_decay as references")
    p.add_argument("--log-name", default="runs.csv",
                   help="run-log file under --out-root; give concurrent jobs distinct names")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    out_root = args.out_root
    CONFIGS.update(anchor_configs(args.anchor_weights))
    CONFIGS.update(gram_configs(args.gram_weights, args.anchor_weights))
    generated = gram_configs(args.gram_weights, args.anchor_weights, args.gram_ref_components,
                             args.bases)
    CONFIGS.update(generated)
    if args.configs is not None:
        configs = args.configs
    elif args.gram_weights:
        configs = args.extra_configs + list(generated)
    elif args.anchor_weights:
        configs = list(anchor_configs(args.anchor_weights))
    else:
        configs = list(CONFIGS)
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        p.error(f"unknown configs: {sorted(unknown)}")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    for dataset in args.datasets:
        split_dir = SPLITS_ROOT / dataset
        train_df = load_graph(split_dir / f"split_{args.split:02d}_train_pome.tsv")
        test_df = load_graph(split_dir / f"split_{args.split:02d}_test_pome.tsv")
        mask = make_mask(train_df, out_root / dataset)
        print(f"{dataset}: {train_df.shape[1] - 1} train / {test_df.shape[1] - 1} test samples, "
              f"{len(mask)} masked cells", flush=True)
        for epochs in args.epochs:
            for config in configs:
                out_dir = out_root / dataset / f"epochs_{epochs}" / config
                for seed in range(args.seeds):
                    row = run_one(dataset, args.split, args.dim, epochs, config, seed,
                                  train_df, test_df, mask, device, out_dir)
                    if row:
                        log.append(row)
    if log:
        path = out_root / args.log_name
        new = pd.DataFrame(log)
        if path.exists():
            old = pd.read_csv(path)
            key = ["dataset", "epochs", "config", "seed"]
            old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values(["dataset", "epochs", "config", "seed"]).to_csv(path, index=False)


if __name__ == "__main__":
    main()
