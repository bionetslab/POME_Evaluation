"""Plotting functions for manuscript-ready figures."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def create_survival_figure(results_df, proportions_df, wilcoxon_results_df, modality_fractions_df, output_path=None):
    """
    Create a two-panel manuscript-ready figure with survival fractions and p-values.
    
    Panel A: Proportion of samples following suggestion
    Panel B: Survivor fractions with suggestion status, annotated with p-values
    
    Args:
        results_df: DataFrame with embedding_size, suggestion_followed, survived_frac, file
        proportions_df: DataFrame with embedding_size, proportion_followed, file
        wilcoxon_results_df: DataFrame with test results including p-values
        modality_fractions_df: DataFrame with modality fractions for 2019 and years prior
        output_path: path to save PDF (if None, only displays)
        
    Returns:
        matplotlib.figure.Figure: the figure object
    """
    if results_df.empty:
        print('No results to plot')
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    mod2description_map = {
        'Systemic and radiotherapy': 'AST & ART',
        'Only radiotherapy': 'ART only',
        'No therapy': 'no AT',
        'Only systemic therapy': 'AST only',
        'All': 'all AT modalities'
    }
    hue_order = ['all AT modalities', 'no AT', 'AST & ART','ART only', 'AST only']

    # Panel A: modality fractions for 2019 vs prior years
    modality_fractions_df.modality = modality_fractions_df.modality.map(mod2description_map)
    sns.barplot(data=modality_fractions_df, x='year', y='fraction', hue='modality', 
                hue_order=hue_order[1:], dodge=True, ax=axes[0, 0],
                palette=sns.color_palette("husl", 5)[1:])
    axes[0, 0].set_xlabel('Year', fontsize=13)
    axes[0, 0].set_ylabel('Fraction of patients', fontsize=13)
    axes[0, 0].set_title('Fractions of patients by AT modality and year', fontsize=14)
    axes[0, 0].legend(title='AT modality', loc='lower left', fontsize=11, title_fontsize=12)
    axes[0, 0].tick_params(labelsize=11)
    axes[0, 0].text(-0.1, 1.1, 'a', transform=axes[0, 0].transAxes, fontsize=18, fontweight='bold')  

    # Panel B: proportions of followed for 2019
    proportions_df.modality = proportions_df.modality.map(mod2description_map)
    sns.barplot(data=proportions_df, x='embedding_size', y='proportion_followed', hue='modality', 
                hue_order=hue_order[:-1], dodge=True, ax=axes[0, 1], 
                palette=sns.color_palette("husl", 5)[:-1])
    axes[0, 1].set_xlabel('Embedding size', fontsize=13)
    axes[0, 1].set_ylabel('Fraction POME-consistent', fontsize=13)
    axes[0, 1].set_title('Fractions of patients with\nPOME-consistent AT modality (2019)', fontsize=14)
    axes[0, 1].legend(title='AT modality', loc='lower left', fontsize=11, title_fontsize=12)
    axes[0, 1].tick_params(labelsize=11)
    axes[0, 1].text(-0.1, 1.1, 'b', transform=axes[0, 1].transAxes, fontsize=18, fontweight='bold')
    
    # Panel C–F: survivor fractions
    labels_row_2 = ['d', 'e', 'f']
    for ax_id_2, mod in enumerate(['No therapy', 'Systemic and radiotherapy', 'Only radiotherapy', 'All']):
        mod_subset = results_df[results_df['modality'] == mod]
        ax_id_1 = 1
        label = labels_row_2[ax_id_2] if ax_id_2 < 3 else 'c'
        if mod == 'All':
            ax_id_1 = 0
            ax_id_2 = 2
        sns.boxplot(data=mod_subset, x='embedding_size', y='survived_frac', hue='suggestion_followed', fill=False, ax=axes[ax_id_1, ax_id_2])
        axes[ax_id_1, ax_id_2].set_xlabel('Embedding size', fontsize=13)
        axes[ax_id_1, ax_id_2].set_ylabel('5-year survivor fraction)', fontsize=13)
        axes[ax_id_1, ax_id_2].set_title(f'Fractions of 5-year survivors\n(2019, {mod2description_map[mod]})', fontsize=14)
        axes[ax_id_1, ax_id_2].legend(title='AT modality\nPOME-consistent', fontsize=11, title_fontsize=12, loc='lower left')
        axes[ax_id_1, ax_id_2].tick_params(labelsize=11)
        
        # Add p-value annotations from Wilcoxon test below the boxplots
        if wilcoxon_results_df is not None and not wilcoxon_results_df.empty:
            embedding_sizes_sorted = sorted(mod_subset['embedding_size'].unique())
            y_min = mod_subset['survived_frac'].min()
            y_max = mod_subset['survived_frac'].max()
            y_range = y_max - y_min
            
            # Adjust y-axis to make room for annotations below
            axes[ax_id_1, ax_id_2].set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)
            
            y_pos = y_max + y_range * 0.08  # Position below the boxplots
            
            for idx, emb_size in enumerate(embedding_sizes_sorted):
                wilcoxon_results_df_mod = wilcoxon_results_df[wilcoxon_results_df['modality'] == mod]
                pval_row = wilcoxon_results_df_mod[wilcoxon_results_df_mod['embedding_size'] == str(emb_size)]
                if not pval_row.empty:
                    pval = pval_row.iloc[0]['p_value']
                    if pd.notna(pval):
                        # Format p-value with appropriate precision
                        if pval < 0.001:
                            pval_text = r'$\it{P}$ < 0.001'
                        else:
                            pval_text = rf'$\it{{P}}$ = {pval:.3f}'
                        axes[ax_id_1, ax_id_2].text(idx, y_pos, pval_text, ha='center', fontsize=11)
        
        # Add panel labels
        axes[ax_id_1, ax_id_2].text(-0.1, 1.1, label, transform=axes[ax_id_1, ax_id_2].transAxes, fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f'Figure saved to {output_path}')
    
    return fig
