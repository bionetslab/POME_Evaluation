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

from hancock_survival.analysis import process_multi_file_analysis, compute_adjuvant_therapy_modality_fractions
from hancock_survival.statistics import perform_wilcoxon_tests


def main():
    """Main pipeline."""
    # Configuration
    data_glob = 'data/HANCOCK_samples_*.tsv'
    output_dir = 'output/'
    feature_cols = [f'dim_{i}' for i in range(16)]
    n_neighbors = 10
    year_filter = 2019
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Processing multi-file analysis...")
    results_df, proportions_df = process_multi_file_analysis(
        data_glob, 
        feature_cols,
        year_filter=year_filter,
        n_neighbors=n_neighbors
    )
    
    # Compute modality fractions for 2019
    modality_fractions_df = compute_adjuvant_therapy_modality_fractions(year_filter=year_filter)
    modality_fractions_df.to_csv(output_dir / 'modality_fractions.csv', index=False)
    
    print("\nPerforming statistical tests...")
    wilcoxon_results_df = perform_wilcoxon_tests(results_df)
    
    print(f"\nSaving results to {output_dir}...")
    results_df.to_csv(output_dir / 'survival_results.csv', index=False)
    proportions_df.to_csv(output_dir / 'proportions_results.csv', index=False)
    wilcoxon_results_df.to_csv(output_dir / 'wilcoxon_results.csv', index=False)
    
    print("Done!")


if __name__ == '__main__':
    main()
