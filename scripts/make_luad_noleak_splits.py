"""Derive the leakage-free TCGA-LUAD train/test splits (dataset key ``luad_noleak``).

Four TCGA-LUAD input variables are recorded after diagnosis and describe the outcome or the
follow-up. They leak the probing targets (Disease Free / Disease-specific Survival /
Progression Free Status) into the embeddings:

    New Neoplasm Event Post Initial Therapy Indicator   new tumour after therapy, i.e. the
                                                        recurrence/progression event itself
                                                        (single-variable AUC 0.93 DFS, 0.91 PFS)
    Person Neoplasm Cancer Status                       with tumour / tumour free at last
                                                        follow-up (AUC 0.84 DSS)
    Last Communication Contact from Initial             follow-up length (AUC 0.77 DSS)
        Pathologic Diagnosis Date
    Last Alive Less Initial Pathologic Diagnosis Date   follow-up time (constant in this data,
        Calculated Day Value                            removed because it is outcome-type)

This script draws the 10 random 80/20 train/test splits of ``scripts/generate_splits.py`` (same
seed 42 as for the other datasets) from the TCGA-LUAD inputs
``data/input_datasets/TCGA_LUAD_wo_targets_graph.tsv`` (POME graph format) and
``TCGA_LUAD_wo_targets.csv`` (UMAP sample matrix), drops these variables and writes
``data/train_test_splits/luad_noleak/split_NN_{train,test}_{pome.tsv,umap.csv}``. It is the only
LUAD dataset of the inductive analyses.

    container/run.sh scripts/make_luad_noleak_splits.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_splits import make_splits  # noqa: E402

INPUTS = PROJECT_ROOT / "data" / "input_datasets"
GRAPH = INPUTS / "TCGA_LUAD_wo_targets_graph.tsv"
SAMPLES = INPUTS / "TCGA_LUAD_wo_targets.csv"
DST = PROJECT_ROOT / "data" / "train_test_splits" / "luad_noleak"
LEAKAGE_VARIABLES = [
    "New Neoplasm Event Post Initial Therapy Indicator",
    "Person Neoplasm Cancer Status",
    "Last Communication Contact from Initial Pathologic Diagnosis Date",
    "Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value",
]


def main() -> None:
    graph = pd.read_csv(GRAPH, sep="\t", index_col=0)
    samples = pd.read_csv(SAMPLES, index_col=0)
    samples.index = samples.index.astype(str)
    missing = (set(LEAKAGE_VARIABLES) - set(graph.index)) | (set(LEAKAGE_VARIABLES) - set(samples.columns))
    if missing:
        raise SystemExit(f"variables not found: {sorted(missing)}")
    DST.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, g_train, g_test, s_train, s_test in make_splits(graph, samples):
        prefix = DST / f"split_{i:02d}"
        g_train.drop(index=LEAKAGE_VARIABLES).to_csv(f"{prefix}_train_pome.tsv", sep="\t")
        g_test.drop(index=LEAKAGE_VARIABLES).to_csv(f"{prefix}_test_pome.tsv", sep="\t")
        s_train.drop(columns=LEAKAGE_VARIABLES).to_csv(f"{prefix}_train_umap.csv")
        s_test.drop(columns=LEAKAGE_VARIABLES).to_csv(f"{prefix}_test_umap.csv")
        n += 4
    print(f"wrote {n} split files without {len(LEAKAGE_VARIABLES)} leakage variables to {DST}")


if __name__ == "__main__":
    main()
