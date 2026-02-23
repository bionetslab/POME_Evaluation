#!/usr/bin/env python
"""
Generate survival analysis figure from HANCOCK dataset.

This script processes multiple HANCOCK sample files, computes k-NN based treatment
recommendations, and generates a manuscript-ready figure showing survivor fractions
and statistical test results.
"""
import sys
import os
from pathlib import Path

# Add src to path so we can import hancock_survival modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hancock_survival.analysis import process_multi_file_analysis
from hancock_survival.statistics import perform_wilcoxon_tests
from hancock_survival.plotting import create_survival_figure


def main():
    """Main pipeline."""
    # Configuration
    data_glob = 'data/HANCOCK_samples_*.tsv'
    output_pdf = 'figures/survival_analysis.pdf'
    feature_cols = [f'dim_{i}' for i in range(16)]
    n_neighbors = 10
    year_filter = 2019
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_pdf).parent
    output_dir.mkdir(exist_ok=True)
    
    print("Processing multi-file analysis...")
    results_df, proportions_df = process_multi_file_analysis(
        data_glob, 
        feature_cols,
        year_filter=year_filter,
        n_neighbors=n_neighbors
    )
    
    print(f"Results: {len(results_df)} rows")
    print(f"Proportions: {len(proportions_df)} rows")
    
    print("\nPerforming statistical tests...")
    wilcoxon_results_df = perform_wilcoxon_tests(results_df)
    
    print("\nWilcoxon test results:")
    print(wilcoxon_results_df)
    
    print(f"\nGenerating figure and saving to {output_pdf}...")
    fig = create_survival_figure(
        results_df, 
        proportions_df, 
        wilcoxon_results_df,
        output_path=output_pdf
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
