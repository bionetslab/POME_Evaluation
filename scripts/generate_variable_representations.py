"""Fit POME on MIMIC-IV *including* the two targets and save candidate variable representations.

Re-runs the setup of the variable-semantics analysis (manuscript Sec. 2.6: all variables incl.
label_aplasia / label_nf, d in {16, 32, 64}, 10 seeds, 2000 epochs) with POME >= 1.2 and, for
every run, stores all candidate per-variable representations defined in
``pome_evaluation.variable_representations`` (raw e_j, value-node centroids at input and latent
level, latent contrast vectors, numeric concepts) plus their norms.

GPU job (MIMIC inputs are gitignored and must be present under data/input_datasets):

    sbatch container/submit.sh scripts/generate_variable_representations.py
    sbatch container/submit.sh scripts/generate_variable_representations.py --dims 32 --runs 2
    sbatch container/submit.sh scripts/generate_variable_representations.py --pome-config stable_k8

Output: data/embeddings/MIMIC/variable_representations[_{config}]/dim_{d}/run_{r}/{option}.tsv
        (no suffix for the published ``default`` config)
"""

import argparse
from pathlib import Path

import pandas as pd

from pome_evaluation.variable_representations import compute_options

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pome_evaluation.pome_configs import POME_CONFIGS, pome_config  # noqa: E402
INPUT_DATASETS = PROJECT_ROOT / "data" / "input_datasets"
OUT_ROOT = PROJECT_ROOT / "data" / "embeddings" / "MIMIC" / "variable_representations"
NA_ENCODING = -99.0
TARGETS = ("label_nf", "label_aplasia")


def representations_root(config: str) -> Path:
    """Output root per POME config; the published ``default`` keeps the original location."""
    return OUT_ROOT if config == "default" else OUT_ROOT.with_name(f"{OUT_ROOT.name}_{config}")


def load_mimic_with_targets() -> pd.DataFrame:
    """Lab variables (graph format) plus the two binary targets as categorical rows."""
    df = pd.read_csv(INPUT_DATASETS / "mimic_aggregated_wo_targets.tsv", sep="\t", index_col=0)
    df.index = df.index.astype(str)
    targets = pd.read_csv(INPUT_DATASETS / "mimic_targets.csv", index_col=0)
    samples = [c for c in df.columns if c != "type"]
    missing = set(samples) - set(targets.index)
    if missing:
        raise ValueError(f"{len(missing)} samples have no target row, e.g. {sorted(missing)[:3]}")
    target_rows = targets.loc[samples, list(TARGETS)].T.fillna(NA_ENCODING)
    target_rows["type"] = "cat"
    return pd.concat([df, target_rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--runs", type=int, default=10, help="seeds 0..runs-1")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pome-config", default="default", choices=list(POME_CONFIGS),
                        help="named POME training configuration (src/pome_evaluation/"
                             "pome_configs.py); 'default' = published settings")
    args = parser.parse_args()

    import torch
    from pome import Embedder, make_deterministic

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    df = load_mimic_with_targets()
    n_num = int((df["type"] == "numerical").sum())
    print(f"MIMIC with targets: {df.shape[0]} variables ({n_num} numeric), "
          f"{df.shape[1] - 1} samples | device={device}", flush=True)
    out_root = representations_root(args.pome_config)
    print(f"POME config: {args.pome_config} {pome_config(args.pome_config)} -> {out_root}",
          flush=True)

    for dim in args.dims:
        for run in range(args.runs):
            out_dir = out_root / f"dim_{dim}" / f"run_{run}"
            if not args.overwrite and (out_dir / "norms.tsv").exists():
                print(f"[skip] dim {dim} run {run}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run]  dim {dim} run {run}", flush=True)
            make_deterministic(run)
            embedder = Embedder(**pome_config(args.pome_config), embedding_dimension=dim, epochs=args.epochs,
                                na_encoding=NA_ENCODING, device=device)
            embedder.fit(df.copy())
            reps, norms = compute_options(embedder)
            for name, rep in reps.items():
                rep.to_csv(out_dir / f"{name}.tsv", sep="\t")
            pd.DataFrame(norms).to_csv(out_dir / "norms.tsv", sep="\t")


if __name__ == "__main__":
    main()
