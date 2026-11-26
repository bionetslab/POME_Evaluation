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

This script copies every ``data/train_test_splits/luad/split_NN_{train,test}_{pome.tsv,umap.csv}``
to ``data/train_test_splits/luad_noleak/``. It drops these variables (rows of the POME graph
format, columns of the UMAP sample matrix) and changes nothing else: the samples, partitions and
values stay identical. All inductive scripts accept ``luad_noleak`` as a dataset. Outputs land
next to the ``luad`` ones under their own ``luad_noleak`` directory, and the original LUAD
results stay untouched.

    container/run.sh scripts/make_luad_noleak_splits.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data" / "train_test_splits" / "luad"
DST = PROJECT_ROOT / "data" / "train_test_splits" / "luad_noleak"
LEAKAGE_VARIABLES = [
    "New Neoplasm Event Post Initial Therapy Indicator",
    "Person Neoplasm Cancer Status",
    "Last Communication Contact from Initial Pathologic Diagnosis Date",
    "Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value",
]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(SRC.glob("split_*_pome.tsv")):
        df = pd.read_csv(path, sep="\t", index_col=0)
        missing = set(LEAKAGE_VARIABLES) - set(df.index)
        if missing:
            raise SystemExit(f"{path.name}: variables not found: {sorted(missing)}")
        df.drop(index=LEAKAGE_VARIABLES).to_csv(DST / path.name, sep="\t")
        n += 1
    for path in sorted(SRC.glob("split_*_umap.csv")):
        df = pd.read_csv(path, index_col=0)
        missing = set(LEAKAGE_VARIABLES) - set(df.columns)
        if missing:
            raise SystemExit(f"{path.name}: variables not found: {sorted(missing)}")
        df.drop(columns=LEAKAGE_VARIABLES).to_csv(DST / path.name)
        n += 1
    print(f"wrote {n} split files without {len(LEAKAGE_VARIABLES)} leakage variables to {DST}")


if __name__ == "__main__":
    main()
