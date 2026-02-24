"""Analysis functions for treatment recommendations."""
import pandas as pd
import numpy as np
import glob
from .data_processing import load_data, add_modality_column, compute_knn_graph


def compute_suggestions(df, knn_graph):
    """
    Generate treatment suggestions based on k-NN survival outcomes.
    
    For each sample, finds neighbors and groups them by modality.
    Recommends the modality with highest survival fraction among neighbors.
    
    Args:
        df: DataFrame with 'adjuvant_therapy_modality' and 'survival_status' columns
        knn_graph: sparse matrix of k-NN connectivity
        
    Returns:
        list: suggested modalities for each sample
    """
    modalities = df['adjuvant_therapy_modality']
    surv = df['survival_status']
    suggestions = []
    n_samples = df.shape[0]
    
    for i in range(n_samples):
        neigh = knn_graph[i].nonzero()[1]
        if len(neigh) == 0:
            suggestions.append(np.nan)
            continue
        
        neigh_mods = modalities.iloc[neigh]
        neigh_surv = surv.iloc[neigh]
        
        # compute survivor fraction per modality
        best_mod = np.nan
        best_frac = -1.0
        
        for mod in neigh_mods.dropna().unique():
            # Use .loc with boolean mask to get indices
            mask = (neigh_mods == mod) & (~neigh_mods.isna())
            idx = mask[mask].index
            if len(idx) > 0:
                frac = (neigh_surv.loc[idx].str.strip().str.lower() == 'living').mean()
                if frac > best_frac:
                    best_frac = frac
                    best_mod = mod
        
        suggestions.append(best_mod)
    
    return suggestions


def add_suggestion_column(df):
    """Add suggestion_followed binary column (1 if suggestion matches actual modality)."""
    if 'suggested_adjuvant_treatment_modality' in df.columns and 'adjuvant_therapy_modality' in df.columns:
        df['suggestion_followed'] = (
            df['suggested_adjuvant_treatment_modality'] == df['adjuvant_therapy_modality']
        ).astype(int)
    return df


def compute_adjuvant_therapy_modality_fractions(year_filter=2019):
    """
    Compute fractions of patients in each adjuvant therapy modality for a given year.
    
    Args:
        df: DataFrame with 'adjuvant_therapy_modality' and 'year_of_initial_diagnosis' columns
        year_filter: year to filter on (e.g., 2019)
    Returns:
        DataFrame with modality fractions  
    """
    df = load_data('data/HANCOCK_samples_16_0.tsv')
    df = add_modality_column(df)

    modality_fractions = []
    for mod in df['adjuvant_therapy_modality'].unique():
        mod_subset = df[df['year_of_initial_diagnosis'] < year_filter]
        total_count = len(mod_subset)
        if total_count > 0:
            fraction = (mod_subset['adjuvant_therapy_modality'] == mod).sum() / total_count
            modality_fractions.append({'modality': mod, 'fraction': fraction, 'year': f"< {year_filter}"})
        mod_subset = df[df['year_of_initial_diagnosis'] == year_filter]
        total_count = len(mod_subset)
        if total_count > 0:
            fraction = (mod_subset['adjuvant_therapy_modality'] == mod).sum() / total_count
            modality_fractions.append({'modality': mod, 'fraction': fraction, 'year': f"{year_filter}"})

    modality_fractions_df = pd.DataFrame(modality_fractions)
    
    return modality_fractions_df


def process_multi_file_analysis(glob_pattern, feature_cols, year_filter=2019, n_neighbors=10):
    """
    Process multiple data files and compute survival fractions by embedding size and suggestion_followed.
    
    Args:
        glob_pattern: glob pattern to match data files (e.g., 'data/HANCOCK_samples_*.tsv')
        feature_cols: list of feature column names
        year_filter: year to filter on (if None, use all data)
        n_neighbors: number of neighbors for k-NN
        
    Returns:
        tuple: (results_df, proportions_df)
    """
    results = []
    proportions = []
    
    for fpath in sorted(glob.glob(glob_pattern)):
        # parse embedding size from filename
        name = fpath.split('/')[-1]
        parts = name.replace('.tsv', '').split('_')
        emb = int(parts[2]) if len(parts) >= 3 else None
        
        # load file
        df_i = load_data(fpath)
        
        # compute modality column
        df_i = add_modality_column(df_i)
        
        # k-NN graph on feature dims
        if all(col in df_i.columns for col in feature_cols):
            X_i = df_i[feature_cols].values
            years_i = df_i['year_of_initial_diagnosis'].values if 'year_of_initial_diagnosis' in df_i.columns else None
            knn_i = compute_knn_graph(X_i, years_i, n_neighbors=n_neighbors)
            
            # compute suggestions
            df_i['suggested_adjuvant_treatment_modality'] = compute_suggestions(df_i, knn_i)
            df_i = add_suggestion_column(df_i)
        
        # filter by year if specified
        if year_filter is not None and 'year_of_initial_diagnosis' in df_i.columns:
            subset = df_i[df_i['year_of_initial_diagnosis'] == year_filter]
        else:
            subset = df_i
        
        # compute survival fractions by suggestion_followed overall and by modality
        if 'survival_status' in subset.columns and 'suggestion_followed' in subset.columns:
            total_count = subset.shape[0]
            
            for followed_val in [0, 1]:
                sub2 = subset[subset['suggestion_followed'] == followed_val]
                if sub2.shape[0] > 0:
                    frac = (sub2['survival_status'].astype(str).str.strip().str.lower() == 'living').mean()
                else:
                    frac = np.nan
                
                results.append({
                    'embedding_size': emb,
                    'suggestion_followed': followed_val,
                    'survived_frac': frac,
                    'file': name,
                    'modality': 'All'
                })

                for mod in subset['adjuvant_therapy_modality'].dropna().unique():
                    mod_sub = sub2[sub2['adjuvant_therapy_modality'] == mod]
                    if mod_sub.shape[0] > 0:
                        frac_mod = (mod_sub['survival_status'].astype(str).str.strip().str.lower() == 'living').mean()
                    else:
                        frac_mod = np.nan
                    
                    results.append({
                        'embedding_size': emb,
                        'suggestion_followed': followed_val,
                        'survived_frac': frac_mod,
                        'file': name,
                        'modality': mod
                    })

            # compute proportions of suggestion followed overall and by modality
            sub_followed_1 = subset[subset['suggestion_followed'] == 1]
            prop = sub_followed_1.shape[0] / total_count if total_count > 0 else np.nan
            proportions.append({'embedding_size': emb, 
                                'proportion_followed': prop, 
                                'file': name,
                                'modality': 'All'
            })

            for mod in subset['adjuvant_therapy_modality'].dropna().unique():
                mod_count = (subset['adjuvant_therapy_modality'] == mod).sum()
                if mod_count > 0:
                    prop = sub_followed_1[sub_followed_1['adjuvant_therapy_modality'] == mod].shape[0] / mod_count
                    proportions.append({
                        'embedding_size': emb,
                        'proportion_followed': prop,
                        'file': name,
                        'modality': mod
                    })
    
    results_df = pd.DataFrame(results)
    proportions_df = pd.DataFrame(proportions)
    
    return results_df, proportions_df
