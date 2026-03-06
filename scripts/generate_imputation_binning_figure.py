#!/usr/bin/env python
"""Generate a 12-panel figure for imputation binning-strategy effects with POME."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hancock_survival.imputation_analysis import load_pome_binning_results
from hancock_survival.imputation_plotting import plot_binning_effects_results

def main() -> None:
    """Load data and render the 12-panel binning-effects figure."""
    output_path = "output/imputation_binning_effects_pome.pdf"

    print("Loading POME imputation benchmark data...")
    data = load_pome_binning_results(data_dir="data")

    print(f"Generating figure and saving to {output_path}...")
    plot_binning_effects_results(data=data, output_path=output_path)
    print("Done!")


if __name__ == "__main__":
    main()
