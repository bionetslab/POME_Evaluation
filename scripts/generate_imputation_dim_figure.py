#!/usr/bin/env python
"""Generate a 2x3 imputation figure for POME with varying embedding sizes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hancock_survival.imputation_analysis import load_pome_z15_dim_results
from hancock_survival.imputation_plotting import plot_imputation_dim_results

def main() -> None:
    """Load z-score/15-bin POME data and render a dim-effects figure."""
    output_path = "output/imputation_dims_pome_z15.pdf"

    print("Loading POME z-score (15 bins) imputation data...")
    metric_distributions = load_pome_z15_dim_results(data_dir="data")

    print(f"Generating figure and saving to {output_path}...")
    plot_imputation_dim_results(metric_distributions=metric_distributions, output_path=output_path)
    print("Done!")


if __name__ == "__main__":
    main()
