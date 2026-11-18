"""Repeated-seed POME fits with individual vs. one joint (shared) sample encoding.

Question: does POME need one learnable input encoding per sample, or is the GAT aggregation
over a sample's value nodes enough? With a shared encoding every sample node gets the same
input feature, so all sample-specific information in the final embedding comes from the GAT.
POME branch ``shared-sample-encoding`` (based on v1.2, commit 6b375ba) adds

    sample_encoding  = "individual"        one learnable table row per sample (POME <= 1.2)
                     | "shared_learnable"  one joint learnable row for all samples
                     | "shared_fixed"      one joint row, frozen at its random initialization
    regressor_inputs = "invariant"         numeric-imputation head on the normalized
                                           pre-encoder sample feature (POME 1.2 default)
                     | "latent"            ... on the GAT output sample embedding instead

With a shared encoding the pre-encoder sample features coincide for all samples, so the
default ("invariant") regression head degenerates to one constant per variable; shared
configs therefore use ``regressor_inputs="latent"``. ``individual_latent`` separates the
effect of the head's input from that of the shared encoding. New samples in ``transform()``
get the shared row as their input feature (individual: zeros, as in POME 1.2).

Configs:  individual          (= POME 1.2 defaults)
          individual_latent   individual encodings, latent regression head
          shared_learnable    joint learnable encoding, latent regression head
          shared_fixed        joint frozen encoding, latent regression head

Follow-up (``--configs`` / ``--set followup``): ways to give unseen samples a well-defined
input feature without losing downstream quality. The branch adds

    mask_rate      = r   per epoch, hide a fraction r of the sample->value edges from message
                         passing and score only those (masked link prediction)
    sample_dropout = p   individual encodings; per epoch, zero the input of a fraction p of the
                         sample nodes (= the input transform() gives unseen samples)
    gat_heads      = h   multi-head GAT attention

          individual_sdrop50       individual, sample_dropout 0.5
          individual_mask30        individual, mask_rate 0.3
          shared_mask30            shared_learnable, mask_rate 0.3, latent head
          shared_heads4            shared_learnable, 4 GAT heads, latent head
          shared_mask30_heads4     shared_learnable, mask_rate 0.3, 4 GAT heads, latent head

Per dataset x epochs x config x seed, on one train/test split (same split for all runs), the
script stores, like generate_training_noise_runs.py (whose mask/IO helpers it reuses),

  * the transductive train-sample and inductive test-sample embeddings of a fit on the full
    training split (clustering, linear probing);
  * the imputations of a second fit on the training split with the fixed 10 % cell mask
    (same mask file construction as the training-noise ablation).

Epochs are fixed (``inductive=False``) so CV epoch tuning does not mix into the comparison.

POME must be the ``shared-sample-encoding`` branch; point the wrappers at its worktree
(submit.sh sets --export=NONE, so POME_REPO must be passed via sbatch --export):
GPU (H200) jobs (first experiment / follow-up):
    sbatch --export=POME_REPO=$HOME/Projects/POME-shared-sample-encoding container/submit.sh \\
        --stage-out output/shared_sample_encoding scripts/generate_shared_sample_encoding_runs.py
    sbatch --export=POME_REPO=$HOME/Projects/POME-shared-sample-encoding,PYTHONHASHSEED=0 \\
        container/submit.sh --stage-out output/shared_sample_encoding_followup \\
        scripts/generate_shared_sample_encoding_runs.py --set followup \\
        --out-root output/shared_sample_encoding_followup
CPU smoke test:
    POME_REPO=~/Projects/POME-shared-sample-encoding container/run.sh \\
        scripts/generate_shared_sample_encoding_runs.py --datasets luad --seeds 2 --epochs 20 \\
        --out-root output/shared_sample_encoding_smoke

Inputs:  data/train_test_splits/{dataset}/split_{NN}_{train,test}_pome.tsv
Outputs: output/shared_sample_encoding/{dataset}/imputation_mask.csv
         output/shared_sample_encoding/{dataset}/epochs_{E}/{config}/seed_{SS}_{train,test,imputed}.csv
         output/shared_sample_encoding/runs.csv          (run log, wall clock)
Analysis: scripts/analyze_shared_sample_encoding.py
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_training_noise_runs as tn  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "output" / "shared_sample_encoding"

CONFIGS = {
    "individual": dict(sample_encoding="individual", regressor_inputs="invariant"),
    "individual_latent": dict(sample_encoding="individual", regressor_inputs="latent"),
    "shared_learnable": dict(sample_encoding="shared_learnable", regressor_inputs="latent"),
    "shared_fixed": dict(sample_encoding="shared_fixed", regressor_inputs="latent"),
    "individual_sdrop50": dict(sample_encoding="individual", sample_dropout=0.5),
    "individual_mask30": dict(sample_encoding="individual", mask_rate=0.3),
    "shared_mask30": dict(sample_encoding="shared_learnable", mask_rate=0.3,
                          regressor_inputs="latent"),
    "shared_heads4": dict(sample_encoding="shared_learnable", gat_heads=4,
                          regressor_inputs="latent"),
    "shared_mask30_heads4": dict(sample_encoding="shared_learnable", mask_rate=0.3, gat_heads=4,
                                 regressor_inputs="latent"),
}
# Config sets: the first experiment, and the follow-up with its two reference configs re-run in
# the same job (same PYTHONHASHSEED, see below).
SETS = {
    "first": ["individual", "individual_latent", "shared_learnable", "shared_fixed"],
    "followup": ["individual", "shared_learnable", "individual_sdrop50", "individual_mask30",
                 "shared_mask30", "shared_heads4", "shared_mask30_heads4"],
}


def check_pome_branch():
    """Abort unless the imported pome supports sample_encoding (shared-sample-encoding branch)."""
    import inspect
    from pome.gnn_embedding import Embedder
    if "sample_encoding" not in inspect.signature(Embedder.__init__).parameters:
        import pome
        sys.exit(f"pome at {Path(pome.__file__).parent} has no sample_encoding option -- set "
                 "POME_REPO to the shared-sample-encoding worktree.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=["hancock", "luad"])
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--epochs", type=int, nargs="+", default=[500, 2000])
    p.add_argument("--set", choices=list(SETS), default="first",
                   help="config set to run (ignored when --configs is given)")
    p.add_argument("--configs", nargs="+", default=None, choices=list(CONFIGS))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--log-name", default="runs.csv")
    args = p.parse_args()
    configs = args.configs or SETS[args.set]
    check_pome_branch()
    if os.environ.get("PYTHONHASHSEED") is None:
        # POME's graph construction iterates over sets of strings, so the value-node / negative
        # ordering -- and hence the fit -- depends on Python's per-process hash seed. Runs that
        # are compared should share it (set PYTHONHASHSEED=0 for the job).
        print("warning: PYTHONHASHSEED is not set; results differ between processes.", flush=True)

    # run_one() of the training-noise script looks configs up in its module-level CONFIGS.
    tn.CONFIGS.clear()
    tn.CONFIGS.update(CONFIGS)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    for dataset in args.datasets:
        split_dir = tn.SPLITS_ROOT / dataset
        train_df = tn.load_graph(split_dir / f"split_{args.split:02d}_train_pome.tsv")
        test_df = tn.load_graph(split_dir / f"split_{args.split:02d}_test_pome.tsv")
        mask = tn.make_mask(train_df, args.out_root / dataset)
        print(f"{dataset}: {train_df.shape[1] - 1} train / {test_df.shape[1] - 1} test samples, "
              f"{len(mask)} masked cells", flush=True)
        for epochs in args.epochs:
            for config in configs:
                out_dir = args.out_root / dataset / f"epochs_{epochs}" / config
                for seed in range(args.seeds):
                    row = tn.run_one(dataset, args.split, args.dim, epochs, config, seed,
                                     train_df, test_df, mask, device, out_dir)
                    if row:
                        log.append(row)
    if log:
        path = args.out_root / args.log_name
        new = pd.DataFrame(log)
        if path.exists():
            old = pd.read_csv(path)
            key = ["dataset", "epochs", "config", "seed"]
            old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values(["dataset", "epochs", "config", "seed"]).to_csv(path, index=False)


if __name__ == "__main__":
    main()
