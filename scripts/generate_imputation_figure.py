#!/usr/bin/env python
"""Generate the imputation benchmarking figure."""
import argparse
from pathlib import Path
import sys

import pandas as pd

# Add src to path so we can import hancock_survival modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hancock_survival.imputation_analysis import build_imputation_figure_data
from hancock_survival.imputation_plotting import plot_imputation_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate imputation benchmarking figure for a selected embedding size.")
    parser.add_argument(
        "--dim",
        type=int,
        choices=[16, 32, 64],
        default=32,
        help="Embedding size to plot (default: 32).",
    )
    return parser.parse_args()


def main() -> None:
    """Build imputation figure data and render the PDF figure."""
    args = parse_args()
    dim = args.dim
    bins = 15
    mode = "z"
    data_dir = "data"

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_pdf = output_dir / f"imputation_ranks_with_competitors_{dim}.pdf"
    cont_path = output_dir / f"imputation_cont_ranks_dim_{dim}.csv"
    cat_path = output_dir / f"imputation_cat_ranks_dim_{dim}.csv"
    dist_path = output_dir / f"imputation_metric_distributions_dim_{dim}.csv"

    legacy_cont_path = output_dir / "imputation_cont_ranks.csv"
    legacy_cat_path = output_dir / "imputation_cat_ranks.csv"
    legacy_dist_path = output_dir / "imputation_metric_distributions.csv"

    if cont_path.exists() and cat_path.exists() and dist_path.exists():
        print(f"Loading existing imputation figure inputs for dim={dim} from output directory...")
        cont_ranks = pd.read_csv(cont_path)
        cat_ranks = pd.read_csv(cat_path)
        metric_distributions = pd.read_csv(dist_path)
    elif dim == 32 and legacy_cont_path.exists() and legacy_cat_path.exists() and legacy_dist_path.exists():
        print("Loading existing legacy imputation figure inputs for dim=32 from output directory...")
        cont_ranks = pd.read_csv(legacy_cont_path)
        cat_ranks = pd.read_csv(legacy_cat_path)
        metric_distributions = pd.read_csv(legacy_dist_path)

        cont_ranks.to_csv(cont_path, index=False)
        cat_ranks.to_csv(cat_path, index=False)
        metric_distributions.to_csv(dist_path, index=False)
    else:
        print(f"Computing imputation figure inputs for dim={dim}...")
        cont_ranks, cat_ranks, metric_distributions = build_imputation_figure_data(
            dim=dim,
            bins=bins,
            mode=mode,
            data_dir=data_dir,
        )
        cont_ranks.to_csv(cont_path, index=False)
        cat_ranks.to_csv(cat_path, index=False)
        metric_distributions.to_csv(dist_path, index=False)

    print(f"Generating figure and saving to {output_pdf}...")
    plot_imputation_results(
        cont_ranks=cont_ranks,
        cat_ranks=cat_ranks,
        metric_distributions=metric_distributions,
        output_path=str(output_pdf),
    )
    print("Done!")


if __name__ == "__main__":
    main()
