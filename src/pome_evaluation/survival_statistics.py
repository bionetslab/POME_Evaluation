"""Statistical testing functions."""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import product 


def perform_wilcoxon_tests(results_df):
    """
    Perform paired Wilcoxon signed-rank tests on survival fractions.
    
    Tests whether survival fractions are higher for suggestion_followed=1 vs =0.
    Pairing is done by file.
    
    Args:
        results_df: DataFrame with columns: embedding_size, suggestion_followed, survived_frac, file
        
    Returns:
        pd.DataFrame: Test results with columns for each embedding size and overall
    """
    if results_df.empty:
        return pd.DataFrame()
    
    embedding_sizes = sorted(results_df['embedding_size'].unique())
    modalities = sorted(results_df['modality'].unique())
    test_results = []
    
    # Test separately for each embedding size
    for emb_size, mod in product(embedding_sizes, modalities):
        subset = results_df[(results_df['embedding_size'] == emb_size) & (results_df['modality'] == mod)]
        
        # Pivot to pair by file
        pivoted = subset.pivot_table(index='file', columns='suggestion_followed', values='survived_frac')
        followed_1 = pivoted[1].dropna()
        followed_0 = pivoted[0].dropna()
        
        # Keep only paired observations
        paired_mask = (~followed_1.isna()) & (~followed_0.isna())
        followed_1_paired = followed_1[paired_mask]
        followed_0_paired = followed_0[paired_mask]
        
        n_pairs = len(followed_1_paired)
        
        if n_pairs > 0:
            stat, pval = wilcoxon(followed_1_paired, followed_0_paired, alternative='greater')
            test_results.append({
                'embedding_size': emb_size,
                'modality': mod,
                'n_pairs': n_pairs,
                'mean_followed_1': followed_1_paired.mean(),
                'std_followed_1': followed_1_paired.std(),
                'mean_followed_0': followed_0_paired.mean(),
                'std_followed_0': followed_0_paired.std(),
                'test_statistic': stat,
                'p_value': pval
            })
        else:
            test_results.append({
                'embedding_size': emb_size,
                'modality': mod,
                'n_pairs': 0,
                'mean_followed_1': np.nan,
                'std_followed_1': np.nan,
                'mean_followed_0': np.nan,
                'std_followed_0': np.nan,
                'test_statistic': np.nan,
                'p_value': np.nan
            })
    
    # Test across all embedding sizes (still paired by file)
    for mod in modalities:
        subset = results_df[results_df['modality'] == mod]
        pivoted_all = subset.pivot_table(index='file', columns='suggestion_followed', values='survived_frac')
        followed_1_all = pivoted_all[1].dropna()
        followed_0_all = pivoted_all[0].dropna()
        
        paired_mask_all = (~followed_1_all.isna()) & (~followed_0_all.isna())
        followed_1_all_paired = followed_1_all[paired_mask_all]
        followed_0_all_paired = followed_0_all[paired_mask_all]
        
        n_pairs_all = len(followed_1_all_paired)
        
        if n_pairs_all > 0:
            stat, pval = wilcoxon(followed_1_all_paired, followed_0_all_paired, alternative='greater')
            test_results.append({
                'embedding_size': 'all',
                'modality': mod,
                'n_pairs': n_pairs_all,
                'mean_followed_1': followed_1_all_paired.mean(),
                'std_followed_1': followed_1_all_paired.std(),
                'mean_followed_0': followed_0_all_paired.mean(),
                'std_followed_0': followed_0_all_paired.std(),
                'test_statistic': stat,
                'p_value': pval
            })
        else:
            test_results.append({
                'embedding_size': 'all',
                'modality': mod,
                'n_pairs': 0,
                'mean_followed_1': np.nan,
                'std_followed_1': np.nan,
                'mean_followed_0': np.nan,
                'std_followed_0': np.nan,
                'test_statistic': np.nan,
                'p_value': np.nan
            })
    
    return pd.DataFrame(test_results)
