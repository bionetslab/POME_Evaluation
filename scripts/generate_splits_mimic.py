"""
Generate 10 random 80/20 train/test splits of the MIMIC dataset at patient level.

Splits are stratified by patient (subject_id): all hospital admissions belonging
to the same patient land in the same partition, preventing patient-level leakage.

Both graph format (vars x samples, mimic_aggregated_wo_targets.tsv) and sample
format (samples x vars, mimic_aggregated_wo_targets_umap.csv) are saved for each
split.

Output layout:
  data/splits/mimic/
    split_00_train_pome.tsv   split_00_test_pome.tsv
    split_00_train_umap.csv   split_00_test_umap.csv
    split_01_...
    ...
"""

import argparse
import os

import numpy as np
import pandas as pd

TYPE_COL = "type"
N_SPLITS = 10
TRAIN_FRAC = 0.8
SEED = 42


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graph", default="data/mimic_aggregated_wo_targets.tsv")
    parser.add_argument("--umap", default="data/mimic_aggregated_wo_targets_umap.csv")
    parser.add_argument(
        "--targets", default="data/mimic_with_targets_patientIDs.csv",
        help="CSV with subject_id column; row order matches sample columns in --graph",
    )
    parser.add_argument("--outdir", default="data/splits/mimic")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    graph_df = pd.read_csv(args.graph, sep="\t", index_col=0)
    umap_df = pd.read_csv(args.umap, index_col=0)
    targets_df = pd.read_csv(args.targets)

    sample_cols = [c for c in graph_df.columns if c != TYPE_COL]

    if len(sample_cols) != len(targets_df):
        raise ValueError(
            f"Sample count mismatch: graph has {len(sample_cols)} samples, "
            f"targets has {len(targets_df)} rows."
        )

    # Map SampleN column name → subject_id using targets row order.
    sample_to_patient = dict(zip(sample_cols, targets_df["subject_id"]))
    all_samples = np.array(sample_cols)
    patient_ids = np.array([sample_to_patient[s] for s in all_samples])

    unique_patients = np.unique(patient_ids)
    n_train_patients = round(len(unique_patients) * args.train_frac)

    total_samples = len(all_samples)
    print(
        f"Dataset: {total_samples} samples, {len(graph_df.index)} variables, "
        f"{len(unique_patients)} unique patients"
    )
    print(
        f"Generating {args.n_splits} splits  "
        f"(~{n_train_patients} train patients / "
        f"~{len(unique_patients) - n_train_patients} test patients)"
    )
    print(f"Output directory: {args.outdir}\n")

    rng = np.random.default_rng(args.seed)

    for i in range(args.n_splits):
        perm = rng.permutation(len(unique_patients))
        train_patients = set(unique_patients[perm[:n_train_patients]])

        train = [s for s, p in zip(all_samples, patient_ids) if p in train_patients]
        test = [s for s, p in zip(all_samples, patient_ids) if p not in train_patients]

        prefix = os.path.join(args.outdir, f"split_{i:02d}")

        graph_df[train + [TYPE_COL]].to_csv(f"{prefix}_train_pome.tsv", sep="\t")
        graph_df[test + [TYPE_COL]].to_csv(f"{prefix}_test_pome.tsv", sep="\t")

        umap_df.loc[train].to_csv(f"{prefix}_train_umap.csv")
        umap_df.loc[test].to_csv(f"{prefix}_test_umap.csv")

        print(f"Split {i:02d}: train={len(train)} samples  test={len(test)} samples")

    print(f"\nDone. Files written to {args.outdir}/")


if __name__ == "__main__":
    main()
