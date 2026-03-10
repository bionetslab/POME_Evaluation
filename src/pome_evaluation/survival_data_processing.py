"""Data loading and preprocessing utilities."""
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import lil_matrix


def load_data(filepath, sep='\t'):
    """Load TSV data file."""
    return pd.read_csv(filepath, sep=sep)


def compute_modality(row):
    """
    Convert adjuvant therapy columns to modality category.
    
    Args:
        row: pandas Series with 'adjuvant_systemic_therapy' and 'adjuvant_radiotherapy'
        
    Returns:
        str: One of 'No therapy', 'Only systemic therapy', 'Only radiotherapy', 'Systemic and radiotherapy', or NaN
    """
    sys = row.get('adjuvant_systemic_therapy')
    rad = row.get('adjuvant_radiotherapy')
    
    def is_yes(val):
        return isinstance(val, str) and val.strip().lower() == 'yes'
    
    if pd.isna(sys) and pd.isna(rad):
        return np.nan
    
    sys_yes = is_yes(sys)
    rad_yes = is_yes(rad)
    
    if not sys_yes and not rad_yes:
        return 'No therapy'
    if sys_yes and not rad_yes:
        return 'Only systemic therapy'
    if not sys_yes and rad_yes:
        return 'Only radiotherapy'
    if sys_yes and rad_yes:
        return 'Systemic and radiotherapy'
    
    return np.nan


def add_modality_column(df):
    """Add adjuvant_therapy_modality column to dataframe."""
    if 'adjuvant_systemic_therapy' in df.columns and 'adjuvant_radiotherapy' in df.columns:
        df['adjuvant_therapy_modality'] = df.apply(compute_modality, axis=1)
    return df


def compute_knn_graph(X, years, n_neighbors=10):
    """
    Compute k-NN connectivity graph with temporal constraints.
    
    Only samples with earlier year_of_initial_diagnosis can be neighbors.
    
    Args:
        X: feature matrix (n_samples, n_features)
        years: array of year_of_initial_diagnosis
        n_neighbors: number of neighbors
        
    Returns:
        scipy.sparse.csr_matrix: sparse adjacency matrix
    """
    if years is None:
        from sklearn.neighbors import kneighbors_graph
        return kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', metric='sqeuclidean', include_self=False)
    
    dist = pairwise_distances(X, metric='sqeuclidean')
    n_samples = X.shape[0]
    
    # mask: do not consider j if year[j] >= year[i]
    mask = np.ones_like(dist, dtype=bool)
    for i in range(n_samples):
        mask[i, :] = years < years[i]
    
    # set distances to inf where mask is False (cannot be neighbor)
    filtered = np.where(mask, dist, np.inf)
    
    # for each row, find indices of k smallest distances
    neighbors = np.argsort(filtered, axis=1)[:, :n_neighbors]
    
    # build adjacency matrix
    knn_graph = lil_matrix((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        for j in neighbors[i]:
            if not np.isinf(filtered[i, j]):
                knn_graph[i, j] = 1
    
    return knn_graph.tocsr()
