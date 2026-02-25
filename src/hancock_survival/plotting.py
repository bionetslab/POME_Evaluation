"""Plotting functions for manuscript-ready figures."""
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
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
    
    mosaic = [
        ['a', 'c', 'd'],
        ['b', 'c', 'd'],
        ['e', 'f', 'g'],
        ['e', 'f', 'g']
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(16, 9), layout='constrained')
    mod2description_map = {
        'Systemic and radiotherapy': 'AST & ART',
        'Only radiotherapy': 'ART only',
        'No therapy': 'no AT',
        'Only systemic therapy': 'AST only',
        'All': 'all AT modalities'
    }
    mod2label_map = {
        'All': 'd',
        'No therapy': 'e',
        'Systemic and radiotherapy': 'f',
        'Only radiotherapy': 'g',
    }
    hue_order = ['all AT modalities', 'no AT', 'AST & ART','ART only', 'AST only']
    offset = transforms.ScaledTranslation(-12/72, 24/72, fig.dpi_scale_trans)

    # Panel a: AT modality fractions for 2019 vs prior years
    modality_fractions_df.modality = modality_fractions_df.modality.map(mod2description_map)
    sns.barplot(data=modality_fractions_df, x='year', y='modality_fraction', hue='modality', 
                hue_order=hue_order[1:], dodge=True, ax=axes['a'],
                palette=sns.color_palette("husl", 5)[1:], legend=False, gap=0.2)
    for patch in axes['a'].patches:
        patch.set_edgecolor(patch.get_facecolor())
        patch.set_linewidth(2.0)
    axes['a'].set_xlabel('Year', fontsize=13)
    axes['a'].set_ylabel('Fraction of patients', fontsize=13)
    axes['a'].set_yticks([0.0, 0.25, 0.5])
    axes['a'].set_ylim(bottom=0.0)
    axes['a'].set_title('Fractions of patients by AT modality and year', fontsize=14)
    axes['a'].tick_params(labelsize=11)
    axes['a'].text(0.0, 1.0, 'a', transform=axes['a'].transAxes + offset, fontsize=18, fontweight='bold')

    # Panel b: survival fractions by AT modality for 2019 vs prior years
    sns.barplot(data=modality_fractions_df, x='year', y='survival_fraction', hue='modality', 
                hue_order=hue_order[1:], dodge=True, ax=axes['b'],
                palette=sns.color_palette("husl", 5)[1:], gap=0.2)
    for patch in axes['b'].patches:
        patch.set_edgecolor(patch.get_facecolor())
        patch.set_linewidth(2.0)
    axes['b'].set_xlabel('Year', fontsize=13)
    axes['b'].set_ylabel('5-year survival fraction', fontsize=13)
    axes['b'].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axes['b'].set_title('5-year survival fractions by AT modality and year', fontsize=14)
    axes['b'].legend(title='AT modality', loc='lower center', bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=11, title_fontsize=12)
    axes['b'].tick_params(labelsize=11)
    axes['b'].text(0.0, 1.0, 'b', transform=axes['b'].transAxes + offset, fontsize=18, fontweight='bold')

    # Panel c: proportions of followed for 2019
    proportions_df.modality = proportions_df.modality.map(mod2description_map)
    sns.barplot(data=proportions_df, x='embedding_size', y='proportion_followed', hue='modality', 
                hue_order=hue_order[:-1], dodge=True, ax=axes['c'], 
                palette=sns.color_palette("husl", 5)[:-1], gap=0.2)
    axes['c'].set_xlabel('Embedding size', fontsize=13)
    axes['c'].set_ylabel('Fraction POME-consistent', fontsize=13)
    axes['c'].set_title('Fractions of patients with\nPOME-consistent AT modality (2019)', fontsize=14)
    axes['c'].legend(title='AT modality', loc='lower left', fontsize=11, title_fontsize=12)
    axes['c'].tick_params(labelsize=11)
    axes['c'].text(0.0, 1.0, 'c', transform=axes['c'].transAxes + offset, fontsize=18, fontweight='bold')
    
    # Panel d - g: survivor fractions by suggestion status for each modality, annotated with p-values
    results_df.suggestion_followed = results_df.suggestion_followed.map({1: 'True', 0: 'False'})
    for mod in ['No therapy', 'Systemic and radiotherapy', 'Only radiotherapy', 'All']:
        label = mod2label_map[mod]
        mod_subset = results_df[results_df['modality'] == mod]
        sns.boxplot(data=mod_subset, x='embedding_size', y='survived_frac', hue='suggestion_followed', fill=False, ax=axes[label], 
                    gap=0.2)
        axes[label].set_xlabel('Embedding size', fontsize=13)
        axes[label].set_ylabel('5-year survival fraction', fontsize=13)
        axes[label].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9,1.0])
        axes[label].set_title(f'5-year survival fractions\n(2019, {mod2description_map[mod]})', fontsize=14)
        axes[label].legend(title='AT modality\nPOME-consistent', fontsize=11, title_fontsize=12, loc='lower left')
        axes[label].tick_params(labelsize=11)
        
        # Add p-value annotations from Wilcoxon test below the boxplots
        if wilcoxon_results_df is not None and not wilcoxon_results_df.empty:
            embedding_sizes_sorted = sorted(mod_subset['embedding_size'].unique())
            
            # Adjust y-axis to make room for annotations below
            axes[label].set_ylim(0.45, 1.1)
            
            y_pos = 1.05  # Position above the boxplots
            
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
                        axes[label].text(idx, y_pos, pval_text, ha='center', fontsize=11)
        
        # Add panel labels
        axes[label].text(0.0, 1.0, label, transform=axes[label].transAxes + offset, fontsize=18, fontweight='bold')
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f'Figure saved to {output_path}')
    
    return fig
