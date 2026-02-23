"""Plotting functions for manuscript-ready figures."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def create_survival_figure(results_df, proportions_df, wilcoxon_results_df, output_path=None):
    """
    Create a two-panel manuscript-ready figure with survival fractions and p-values.
    
    Panel A: Proportion of samples following suggestion
    Panel B: Survivor fractions with suggestion status, annotated with p-values
    
    Args:
        results_df: DataFrame with embedding_size, suggestion_followed, survived_frac, file
        proportions_df: DataFrame with embedding_size, proportion_followed, file
        wilcoxon_results_df: DataFrame with test results including p-values
        output_path: path to save PDF (if None, only displays)
        
    Returns:
        matplotlib.figure.Figure: the figure object
    """
    if results_df.empty:
        print('No results to plot')
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: proportions of followed for 2019
    if not proportions_df.empty:
        sns.boxplot(data=proportions_df, x='embedding_size', y='proportion_followed', ax=axes[0])
        axes[0].set_xlabel('Embedding Size', fontsize=13)
        axes[0].set_ylabel('Proportion Followed (2019)', fontsize=13)
        axes[0].set_title('Proportion of Patients Following Suggestion', fontsize=14, fontweight='bold')
        axes[0].tick_params(labelsize=11)
    else:
        axes[0].text(0.5, 0.5, 'No proportion data', ha='center', va='center')
    
    # Panel B: survivor fractions
    sns.boxplot(data=results_df, x='embedding_size', y='survived_frac', hue='suggestion_followed', ax=axes[1])
    axes[1].set_xlabel('Embedding Size', fontsize=13)
    axes[1].set_ylabel('Survivor Fraction (2019)', fontsize=13)
    axes[1].set_title('Survivor Fractions by Embedding Size', fontsize=14, fontweight='bold')
    axes[1].legend(title='Suggestion Followed', loc='upper right', fontsize=11, title_fontsize=12)
    axes[1].tick_params(labelsize=11)
    
    # Add p-value annotations from Wilcoxon test below the boxplots
    if wilcoxon_results_df is not None and not wilcoxon_results_df.empty:
        embedding_sizes_sorted = sorted(results_df['embedding_size'].unique())
        y_min = results_df['survived_frac'].min()
        y_range = results_df['survived_frac'].max() - y_min
        
        # Adjust y-axis to make room for annotations below
        axes[1].set_ylim(y_min - y_range * 0.15, results_df['survived_frac'].max() + y_range * 0.05)
        
        y_pos = y_min - y_range * 0.08  # Position below the boxplots
        
        for idx, emb_size in enumerate(embedding_sizes_sorted):
            pval_row = wilcoxon_results_df[wilcoxon_results_df['embedding_size'] == emb_size]
            if not pval_row.empty:
                pval = pval_row.iloc[0]['p_value']
                if pd.notna(pval):
                    # Format p-value with appropriate precision
                    if pval < 0.001:
                        pval_text = r'$\it{P}$ < 0.001'
                    else:
                        pval_text = rf'$\it{{P}}$ = {pval:.3f}'
                    axes[1].text(idx, y_pos, pval_text, ha='center', fontsize=11)
    
    # Add panel labels
    axes[0].text(-0.1, 1.05, 'A', transform=axes[0].transAxes, fontsize=18, fontweight='bold')
    axes[1].text(-0.1, 1.05, 'B', transform=axes[1].transAxes, fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f'Figure saved to {output_path}')
    
    return fig
