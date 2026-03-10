"""Analysis helpers for imputation rank benchmarking across cohorts."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


COHORT_FILENAMES = {
    "tcga": "TCGA_LUAD_imputation_{mode}_bins_{bins}.csv",
    "hancock": "HANCOCK_imputation_{mode}_bins_{bins}.csv",
    "mimic": "MIMIC_imputation_{mode}_bins_{bins}.csv",
}

COHORT_LABELS = {
    "tcga": "TCGA-LUAD",
    "hancock": "HANCOCK",
    "mimic": "MIMIC IV",
}

POME_DATASET_ORDER = ["HANCOCK", "MIMIC", "TCGA_LUAD"]
POME_DATASET_LABELS = {
    "HANCOCK": "HANCOCK",
    "MIMIC": "MIMIC IV",
    "TCGA_LUAD": "TCGA-LUAD",
}
POME_STRATEGY_ORDER = ["z", "nonlinear"]
POME_NBINS_ORDER = [7, 11, 15]
POME_DIM_ORDER = [16, 32, 64]
POME_FILE_RE = re.compile(r"^(?P<dataset>.+?)_imputation_(?P<strategy>z|nonlinear)_bins_(?P<nbins>\d+)\.csv$")


def _resolve_input_file(filename: str, data_dir: str = "data") -> Path:
    """Resolve input CSV path from project root or data directory."""
    candidates = [Path(filename), Path(data_dir) / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in project root or {data_dir}/")


def load_imputation_benchmark_data(
    dim: int = 32,
    bins: int = 15,
    mode: str = "z",
    data_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """Load and filter imputation benchmark CSV files for all cohorts."""
    frames = {}
    for cohort, pattern in COHORT_FILENAMES.items():
        filename = pattern.format(mode=mode, bins=bins)
        file_path = _resolve_input_file(filename, data_dir=data_dir)
        frame = pd.read_csv(file_path, index_col=0)
        frame = frame[frame["dim"].isin([dim, -1])].sort_values(by="tool", ascending=False)
        frames[cohort] = frame
    return frames


def _summed_ranks_by_cohort(
    frame: pd.DataFrame,
    score_column: str,
    higher_is_better: bool,
    rank_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank each run/na_ratio group and aggregate ranks by method."""
    ranked = frame.copy()
    ranked[rank_column] = (
        ranked.groupby(["run", "na_ratio"])[score_column]
        .rank(method="dense", ascending=not higher_is_better)
    )

    summed = (
        ranked.groupby(["na_ratio", "tool"])[rank_column]
        .sum()
        .reset_index()
        .rename(columns={"tool": "method", rank_column: "summed_rank"})
    )

    run_counts = ranked.groupby("na_ratio")["run"].nunique().reset_index(name="n_runs")
    return summed, run_counts


def compute_average_ranks_across_cohorts(
    frames: dict[str, pd.DataFrame],
    score_column: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Compute merged total and average ranks for one metric across all cohorts."""
    rank_column = f"rank_{score_column}"

    merged_ranks = None
    merged_run_counts = None

    for cohort, frame in frames.items():
        cohort_summed, cohort_run_counts = _summed_ranks_by_cohort(
            frame=frame,
            score_column=score_column,
            higher_is_better=higher_is_better,
            rank_column=rank_column,
        )
        cohort_summed = cohort_summed.rename(columns={"summed_rank": f"summed_rank_{cohort}"})
        cohort_run_counts = cohort_run_counts.rename(columns={"n_runs": f"n_runs_{cohort}"})

        if merged_ranks is None:
            merged_ranks = cohort_summed
            merged_run_counts = cohort_run_counts
        else:
            merged_ranks = merged_ranks.merge(cohort_summed, on=["na_ratio", "method"], how="inner")
            merged_run_counts = merged_run_counts.merge(cohort_run_counts, on="na_ratio", how="inner")

    summed_rank_cols = [column for column in merged_ranks.columns if column.startswith("summed_rank_")]
    run_count_cols = [column for column in merged_run_counts.columns if column.startswith("n_runs_")]

    merged = merged_ranks.merge(merged_run_counts, on="na_ratio", how="inner")
    merged["summed_rank_total"] = merged[summed_rank_cols].sum(axis=1)
    merged["n_runs_total"] = merged[run_count_cols].sum(axis=1)
    merged["avg_rank"] = merged["summed_rank_total"] / merged["n_runs_total"]
    merged["rank"] = (
        merged.groupby("na_ratio")["summed_rank_total"]
        .rank(method="dense", ascending=not higher_is_better)
    )
    merged["na_ratio"] = merged["na_ratio"].astype(str)

    return merged


def _build_rank_tables_from_frames(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build numeric and categorical average-rank tables from already loaded frames."""
    cont_ranks = compute_average_ranks_across_cohorts(
        frames=frames,
        score_column="mae_cont",
        higher_is_better=False,
    )
    cat_ranks = compute_average_ranks_across_cohorts(
        frames=frames,
        score_column="acc_cat",
        higher_is_better=True,
    )
    return cont_ranks, cat_ranks


def _build_metric_distributions(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build long-form metric distributions per dataset and method."""
    distributions = []
    for cohort, frame in frames.items():
        dist_frame = frame[["tool", "acc_cat", "mae_cont", "na_ratio", "run"]].copy()
        dist_frame = dist_frame.rename(columns={"tool": "method"})
        dist_frame["dataset"] = COHORT_LABELS.get(cohort, cohort)
        distributions.append(dist_frame)
    return pd.concat(distributions, ignore_index=True)


def build_imputation_rank_tables(
    dim: int = 32,
    bins: int = 15,
    mode: str = "z",
    data_dir: str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build numeric and categorical average-rank tables used for plotting."""
    frames = load_imputation_benchmark_data(dim=dim, bins=bins, mode=mode, data_dir=data_dir)
    return _build_rank_tables_from_frames(frames)


def build_imputation_figure_data(
    dim: int = 32,
    bins: int = 15,
    mode: str = "z",
    data_dir: str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all data objects needed for the imputation figure.

    Returns:
        cont_ranks: Average-rank table for numeric variables (mae_cont)
        cat_ranks: Average-rank table for categorical variables (acc_cat)
        metric_distributions: Long-form raw metrics with columns
            [dataset, method, acc_cat, mae_cont, na_ratio, run]
            aggregated over all missingness ratios (no filtering by na_ratio).
    """
    frames = load_imputation_benchmark_data(dim=dim, bins=bins, mode=mode, data_dir=data_dir)
    cont_ranks, cat_ranks = _build_rank_tables_from_frames(frames)
    metric_distributions = _build_metric_distributions(frames)
    return cont_ranks, cat_ranks, metric_distributions


def load_pome_binning_results(data_dir: str = "data") -> pd.DataFrame:
    """Load and combine POME rows across datasets, strategies, and bin counts."""
    records: list[pd.DataFrame] = []
    for csv_path in sorted(Path(data_dir).glob("*_imputation_*_bins_*.csv")):
        match = POME_FILE_RE.match(csv_path.name)
        if not match:
            continue

        dataset_key = match.group("dataset")
        strategy = match.group("strategy")
        nbins = int(match.group("nbins"))

        if dataset_key not in POME_DATASET_ORDER:
            continue
        if strategy not in POME_STRATEGY_ORDER:
            continue
        if nbins not in POME_NBINS_ORDER:
            continue

        df = pd.read_csv(csv_path)
        pome_df = df[df["tool"] == "POME"].copy()
        if pome_df.empty:
            continue

        pome_df["dataset"] = dataset_key
        pome_df["strategy"] = strategy
        pome_df["nbins"] = nbins
        records.append(pome_df[["dataset", "strategy", "nbins", "dim", "acc_cat", "mae_cont"]])

    if not records:
        raise ValueError(f"No POME rows were loaded from {data_dir}.")

    combined = pd.concat(records, ignore_index=True)

    missing_combinations: list[str] = []
    for dataset_key in POME_DATASET_ORDER:
        for strategy in POME_STRATEGY_ORDER:
            for nbins in POME_NBINS_ORDER:
                subset = combined[
                    (combined["dataset"] == dataset_key)
                    & (combined["strategy"] == strategy)
                    & (combined["nbins"] == nbins)
                ]
                if subset.empty:
                    missing_combinations.append(f"{dataset_key} | {strategy} | bins={nbins}")

    if missing_combinations:
        missing_text = "\n".join(missing_combinations)
        raise ValueError(f"Missing required combinations:\n{missing_text}")

    return combined


def load_pome_z15_dim_results(data_dir: str = "data") -> pd.DataFrame:
    """Load POME rows for z-score binning with 15 bins and dim in [16, 32, 64]."""
    frames: list[pd.DataFrame] = []
    for dataset_key in POME_DATASET_ORDER:
        file_path = Path(data_dir) / f"{dataset_key}_imputation_z_bins_15.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing expected file: {file_path}")

        df = pd.read_csv(file_path)
        subset = df[(df["tool"] == "POME") & (df["dim"].isin(POME_DIM_ORDER))].copy()
        if subset.empty:
            raise ValueError(f"No POME rows with dim in {POME_DIM_ORDER} found in {file_path}")

        subset["dataset"] = POME_DATASET_LABELS[dataset_key]
        subset["dim"] = subset["dim"].astype(int)
        frames.append(subset[["dataset", "dim", "acc_cat", "mae_cont"]])

    combined = pd.concat(frames, ignore_index=True)

    missing = []
    for dataset_label in [POME_DATASET_LABELS[key] for key in POME_DATASET_ORDER]:
        for dim in POME_DIM_ORDER:
            has_rows = not combined[(combined["dataset"] == dataset_label) & (combined["dim"] == dim)].empty
            if not has_rows:
                missing.append(f"{dataset_label} | dim={dim}")

    if missing:
        missing_text = "\n".join(missing)
        raise ValueError(f"Missing required dataset/dim combinations:\n{missing_text}")

    return combined
