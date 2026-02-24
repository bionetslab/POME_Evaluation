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
import pandas as pd

# Add src to path so we can import hancock_survival modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hancock_survival.analysis import compute_adjuvant_therapy_modality_fractions, process_multi_file_analysis
from hancock_survival.statistics import perform_wilcoxon_tests
from hancock_survival.plotting import create_survival_figure


def main():
    """Main pipeline."""
    # Configuration
    data_glob = 'data/HANCOCK_samples_*.tsv'
    output_pdf = 'output/survival_analysis.pdf'
    feature_cols = [f'dim_{i}' for i in range(16)]
    n_neighbors = 10
    year_filter = 2019
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_pdf).parent
    output_dir.mkdir(exist_ok=True)
    
    # Load output results if they exist to avoid re-computation
    if output_dir.joinpath('survival_results.csv').exists() and output_dir.joinpath('proportions_results.csv').exists():
        print("Loading existing results from output directory...")
        results_df = pd.read_csv(output_dir / 'survival_results.csv')
        proportions_df = pd.read_csv(output_dir / 'proportions_results.csv')
    else:
        print("Processing multi-file analysis...")
        results_df, proportions_df = process_multi_file_analysis(
            data_glob, 
            feature_cols,
            year_filter=year_filter,
            n_neighbors=n_neighbors
        )
        results_df.to_csv(output_dir / 'survival_results.csv', index=False)
        proportions_df.to_csv(output_dir / 'proportions_results.csv', index=False)
    
    # Load existing Wilcoxon results if they exist
    if output_dir.joinpath('wilcoxon_results.csv').exists():
        print("Loading existing Wilcoxon test results...")
        wilcoxon_results_df = pd.read_csv(output_dir / 'wilcoxon_results.csv')
        print("\nWilcoxon test results:")
        print(wilcoxon_results_df)
    else:
        print("\nPerforming statistical tests...")
        wilcoxon_results_df = perform_wilcoxon_tests(results_df)
        wilcoxon_results_df.to_csv(output_dir / 'wilcoxon_results.csv', index=False)

    # Load modality fractions for 2019 if they exist
    if output_dir.joinpath('modality_fractions.csv').exists():
        print("Loading existing modality fractions for 2019...")
        modality_fractions_df = pd.read_csv(output_dir / 'modality_fractions.csv')
    else:
        print("\nComputing modality fractions for 2019...")
        modality_fractions_df = compute_adjuvant_therapy_modality_fractions(year_filter=year_filter)
        modality_fractions_df.to_csv(output_dir / 'modality_fractions.csv', index=False)
    
    print(f"\nGenerating figure and saving to {output_pdf}...")
    fig = create_survival_figure(
        results_df, 
        proportions_df, 
        wilcoxon_results_df,
        modality_fractions_df,
        output_path=output_pdf
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
