import pandas as pd
import os
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder, normalize
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_X_y, _safe_indexing
import networkx as nx
from community import community_louvain
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

def gini_impurity(y):
    probs = y.value_counts(normalize=True)
    return 1 - np.sum(probs ** 2)

def normalized_cluster_entropy(labels, base=2):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    H = -np.sum(probs * np.log(probs) / np.log(base))
    k = len(counts)
    return H / np.log(k) * np.log(base) if k > 1 else 0.0

# Helper function for Dunn Index if validclust isn't installed
def compute_dunn(X, labels, metric):
    if len(set(labels)) < 2:
        return -1
    
    # Filter out noise (-1) for valid metric calculation
    mask = labels != -1
    if np.sum(mask) == 0: return -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    # Using a simplified diameter/linkage approach
    from validclust import dunn
    dist_matrix = pairwise_distances(X_filtered, metric=metric)
    return dunn(dist_matrix, labels_filtered)

def davies_bouldin_score_custom(X, labels, metric='euclidean', **kwds):
    # standard boilerplate for validation
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, n_features = X.shape
    n_labels = len(le.classes_)
    
    # ... (insert check_number_of_labels here)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, n_features))
    
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, np.where(labels == k)[0])
        centroid = np.mean(cluster_k, axis=0)
        centroids[k] = centroid
        
        # Pass the metric to intra-cluster distance calculation
        # Note: we compare cluster points to the single centroid
        intra_dists[k] = np.mean(
            pairwise_distances(cluster_k, [centroid], metric=metric, **kwds)
        )

    # Pass the metric to inter-centroid distance calculation
    centroid_distances = pairwise_distances(centroids, metric=metric, **kwds)

    # Handle division by zero/edge cases
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    # Fill diagonal with inf to ignore self-comparison in the max() step
    np.fill_diagonal(centroid_distances, np.inf)
    
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    
    return np.mean(scores)

NUM_EMBEDDINGS = 10

# Common input paths for MIMIC.
#dataset = "MIMIC"
#embedding_path = f"~/Projects/graph_based_embeddings.git/data/preprocessed/graph_based/{dataset}/embeddings"
#targets_path = f"../../data/input/MIMIC_targets.csv"
#targets_df = pd.read_csv(targets_path, index_col=0)

# Common input paths for TCGA.
dataset = "TCGA_LUAD"
embedding_path = f"~/Projects/graph_based_embeddings.git/data/preprocessed/graph_based/{dataset}/embeddings"
targets_path = f"../../data/input/TCGA_LUAD_targets.csv"
targets_df = pd.read_csv(targets_path, index_col=0)

# Common input paths for HANCOCK.
#dataset = "HANCOCK"
#embedding_path = f"~/Projects/graph_based_embeddings.git/data/preprocessed/graph_based/{dataset}/embeddings"
#targets_path = f"../../data/input/hancock_targets.csv"
#targets_df = pd.read_csv(targets_path, index_col=0)

# Common input paths for LUAD.
#dataset = "MIMIC"
#embedding_path = f"~/Projects/graph_based_embeddings.git/data/preprocessed/graph_based/{dataset}/embeddings"
#targets_path = f"../../data/input/MIMIC_targets.csv"
#targets_df = pd.read_csv(targets_path, index_col=0)

# Result dictionary for P-values.
pvalues_dict = {'run': [], 'k' : [], 'target' : [], 'pvalue': [], 'method' : [], 'dim': []}
silhouette_dict = {'run': [], 'k' : [], 'silhouette' : [], "davies_bouldin" : [], "db_custom": [], "calinski_harabasz": [], "dunn": [], 'method' : [], 'dim' : [], 'entropy': []}
dim_list = [16, 32, 64]
clustering_type = "leiden_knn"


for dim in dim_list:
    print(f"Running dimensions = ", dim)
    for i in range(NUM_EMBEDDINGS):
        print(f"    Processing embedding {i} from {NUM_EMBEDDINGS}...")
        # Read corresponding POME and UMAP embeddings. MIMIC: 64, HANCOCK: 16.
        embedding_file = os.path.join(embedding_path, f"{dataset}_samples_{dim}_{i}.tsv")
        umap_file = os.path.join(embedding_path, f"{dataset}_UMAP_{dim}_{i}.csv")
        embedding_df = pd.read_csv(embedding_file, index_col=0, sep='\t')
        umap_df = pd.read_csv(umap_file, index_col=0)
        
        embedding_df.index = embedding_df.index.astype(str)
        umap_df.index = umap_df.index.astype(str)
        targets_df.index = targets_df.index.astype(str)
        
        
        embedding_dist = squareform(pdist(embedding_df.values, metric="euclidean"))
        umap_dist = squareform(pdist(umap_df.values, metric="euclidean"))
        
        results = {}  # to store models, labels, and scores
        embedding_mat = embedding_df.to_numpy()
        umap_mat = umap_df.to_numpy()

        if clustering_type == "kmeans":
            
            for k in range(2, 11):
                # First cluster GAE embeddings.
                kmeans1 = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels_gae = kmeans1.fit_predict(embedding_mat)
                # Then cluster UMAP embeddings.
                kmeans2 = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels_umap = kmeans2.fit_predict(umap_mat)

                embedding_df[f'{k}_means'] = labels_gae 
                umap_df[f'{k}_means'] = labels_umap
                
                embedding_silhouette = silhouette_score(X=embedding_mat, labels=labels_gae)
                silhouette_dict['run'].append(i)
                silhouette_dict['k'].append(k)
                silhouette_dict['score'].append(embedding_silhouette)
                silhouette_dict['method'].append("POME")
                silhouette_dict['dim'].append(dim)
                umap_silhouette = silhouette_score(X=umap_mat, labels=labels_umap)
                silhouette_dict['run'].append(i)
                silhouette_dict['k'].append(k)
                silhouette_dict['score'].append(umap_silhouette)
                silhouette_dict['method'].append("UMAP")
                silhouette_dict['dim'].append(dim)
                
            # Merge targets df into embedding df and label-encode columns.
            cols_to_encode = []
            result_df = embedding_df.join(targets_df, how="inner")
            results_umap_df = umap_df.join(targets_df, how="inner")
            le = LabelEncoder()
            for col in cols_to_encode:
                result_df[col] = le.fit_transform(result_df[col])
                results_umap_df[col] = le.fit_transform(results_umap_df[col])
                
            # Analyze and plot separation of clusters regarding categorical target variable.
            if dataset == "TCGA_BRCA":
                cat_targets = ["vital_status.demographic"]
            elif dataset == "HANCOCK":
                cat_targets = ["recurrence", "survival_status", "rfs_event"]
            elif dataset == "MIMIC":
                cat_targets = ["label_aplasia", "label_nf"]
            elif dataset == "TCGA_LUAD":
                cat_targets = ["Disease Free Status", "Disease-specific Survival status", "Progression Free Status"]
            else:
                raise ValueError(f"Unkown dataset: {dataset}")
            for target in cat_targets:
                for k in range(2,11):
                    contingency = pd.crosstab(result_df[f'{k}_means'], result_df[target])
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    pvalues_dict['k'].append(k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(p)
                    pvalues_dict['method'].append("POME")
                    pvalues_dict['run'].append(i)
                    
                    # Compute Gini impurity per cluster.
                    summed_impurity = (
                        result_df
                        .groupby(f'{k}_means')[target]
                        .apply(gini_impurity)
                    ).sum()
                    pvalues_dict['summed_impurity'].append(summed_impurity)

                    contingency = pd.crosstab(results_umap_df[f'{k}_means'], results_umap_df[target])
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    pvalues_dict['k'].append(k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(p)
                    pvalues_dict['method'].append("UMAP")
                    pvalues_dict['run'].append(i)
                    summed_impurity = (
                        results_umap_df
                        .groupby(f'{k}_means')[target]
                        .apply(gini_impurity)
                    ).sum()
                    pvalues_dict['summed_impurity'].append(summed_impurity)

            # Analyze and plot separation of clusters regarding numeric target variable.
            if dataset == "TCGA_BRCA":
                num_targets = ["year_of_death.demographic", "days_to_death.demographic"]
            elif dataset == "HANCOCK":
                num_targets = ["days_to_rfs_event", "days_to_last_information"]
            elif dataset == "MIMIC":
                num_targets = []
            elif dataset == "TCGA_LUAD":
                num_targets = ["Disease Free (Months)", "Months of disease-specific survival", "Progress Free Survival (Months)"]
            
            for target in num_targets:
                for k in range(2, 11):
                    # Group target values by cluster label
                    groups = [
                        result_df[result_df[f'{k}_means'] == cluster][target].dropna()
                        for cluster in range(k)
                    ]
                    
                    variances = [
                        group.std(ddof=0)
                        for group in groups
                    ]

                    # Perform Kruskal-Wallis test
                    if all(len(g) > 1 for g in groups):  # Kruskal requires at least 2 values per group
                        stat, p = kruskal(*groups)
                        adj_p = p
                    else:
                        adj_p = np.nan

                    pvalues_dict['k'].append(k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(adj_p)
                    pvalues_dict['method'].append("POME")
                    pvalues_dict['run'].append(i)
                    pvalues_dict['summed_impurity'].append(sum(variances))

                    # Group target values by cluster label
                    groups = [
                        results_umap_df[results_umap_df[f'{k}_means'] == cluster][target].dropna()
                        for cluster in range(k)
                    ]
                    
                    variances = [
                        group.std(ddof=0)
                        for group in groups
                    ]

                    # Perform Kruskal-Wallis test
                    if all(len(g) > 1 for g in groups):  # Kruskal requires at least 2 values per group
                        stat, p = kruskal(*groups)
                        adj_p = p
                    else:
                        adj_p = np.nan  # fallback if any group is too small

                    pvalues_dict['k'].append(k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(adj_p)
                    pvalues_dict['method'].append("UMAP")
                    pvalues_dict['run'].append(i)
                    pvalues_dict['summed_impurity'].append(sum(variances))
            
        
        elif clustering_type == "louvain":
            
            # Run Louvain on POME embeddings.
            pome_dist_matrix = pairwise_distances(embedding_mat, metric='euclidean')

            # 2. Convert distance to similarity (Weight increases as distance decreases)
            # sigma controls how 'far' the influence of a point reaches
            sigma = np.median(pome_dist_matrix) 
            weights = np.exp(-pome_dist_matrix**2 / (2 * sigma**2))

            # 3. Create graph from the weighted adjacency matrix
            pome_graph = nx.from_numpy_array(weights)

            # 4. Run Louvain
            pome_partition = community_louvain.best_partition(pome_graph, weight='weight')
            pome_labels = np.array([pome_partition[i] for i in sorted(pome_partition.keys())])
            pome_k = len(set(pome_partition.values()))
            
            ## Run on UMAP embeddings.
            umap_dist_matrix = pairwise_distances(umap_mat, metric='euclidean')

            sigma = np.median(umap_dist_matrix) 
            weights = np.exp(-umap_dist_matrix**2 / (2 * sigma**2))

            # 3. Create graph from the weighted adjacency matrix
            umap_graph = nx.from_numpy_array(weights)

            # 4. Run Louvain
            umap_partition = community_louvain.best_partition(umap_graph, weight='weight')
            umap_labels = np.array([umap_partition[i] for i in sorted(umap_partition.keys())])
            umap_k = len(set(umap_partition.values()))
            
            embedding_silhouette = silhouette_score(X=embedding_mat, labels=pome_labels)
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(pome_k)
            silhouette_dict['score'].append(embedding_silhouette)
            silhouette_dict['method'].append("POME")
            silhouette_dict['dim'].append(dim)
            umap_silhouette = silhouette_score(X=umap_mat, labels=umap_labels)
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(umap_k)
            silhouette_dict['score'].append(umap_silhouette)
            silhouette_dict['method'].append("UMAP")
            silhouette_dict['dim'].append(dim)

        elif clustering_type == "louvain_knn":
            
            # --- Parameters ---
            n_neighbors = 30

            ## 1. Process POME Embeddings
            # 'connectivity' mode creates a binary (0 or 1) matrix
            pome_adj = kneighbors_graph(embedding_mat, n_neighbors=n_neighbors, 
                                        mode='connectivity', include_self=False)

            # Convert sparse matrix to undirected NetworkX graph
            pome_graph = nx.from_scipy_sparse_array(pome_adj).to_undirected()

            # Run Louvain (No weight parameter needed for unweighted)
            pome_partition = community_louvain.best_partition(pome_graph)
            pome_labels = np.array([pome_partition[i] for i in sorted(pome_partition.keys())])
            pome_k = len(set(pome_partition.values()))

            ## 2. Process UMAP Embeddings
            umap_adj = kneighbors_graph(umap_mat, n_neighbors=n_neighbors, 
                                        mode='connectivity', include_self=False)

            umap_graph = nx.from_scipy_sparse_array(umap_adj).to_undirected()

            # Run Louvain
            umap_partition = community_louvain.best_partition(umap_graph)
            umap_labels = np.array([umap_partition[i] for i in sorted(umap_partition.keys())])
            umap_k = len(set(umap_partition.values()))

            ## 3. Metrics (Keeping your original structure)
            embedding_silhouette = silhouette_score(X=embedding_mat, labels=pome_labels)
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(pome_k)
            silhouette_dict['score'].append(embedding_silhouette)
            silhouette_dict['method'].append("POME_KNN")
            silhouette_dict['dim'].append(dim)

            umap_silhouette = silhouette_score(X=umap_mat, labels=umap_labels)
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(umap_k)
            silhouette_dict['score'].append(umap_silhouette)
            silhouette_dict['method'].append("UMAP_KNN")
            silhouette_dict['dim'].append(dim)
        
        elif clustering_type == "leiden_knn":
            import igraph as ig
            import leidenalg as la
            import numpy as np
            from sklearn.neighbors import kneighbors_graph

            # --- Parameters ---
            n_neighbors = int(0.05 * len(embedding_df))

            ## 1. Process POME Embeddings
            pome_adj = kneighbors_graph(
                embedding_mat,
                n_neighbors=n_neighbors,
                mode='connectivity',
                include_self=False,
                metric='sqeuclidean'
            )

            # Convert sparse matrix to igraph
            sources, targets = pome_adj.nonzero()
            pome_ig = ig.Graph(
                n=pome_adj.shape[0],
                edges=list(zip(sources, targets)),
                directed=False
            )

            # Run Leiden
            pome_partition = la.find_partition(
                pome_ig,
                la.RBConfigurationVertexPartition
            )
            pome_labels = np.array(pome_partition.membership)
            pome_k = len(pome_partition)
            embedding_df["leiden_knn"] = pome_labels


            ## 2. Process UMAP Embeddings
            umap_adj = kneighbors_graph(
                umap_mat,
                n_neighbors=n_neighbors,
                mode='connectivity',
                include_self=False,
            )

            sources_u, targets_u = umap_adj.nonzero()
            umap_ig = ig.Graph(
                n=umap_adj.shape[0],
                edges=list(zip(sources_u, targets_u)),
                directed=False
            )

            # Run Leiden
            umap_partition = la.find_partition(
                umap_ig,
                la.RBConfigurationVertexPartition
            )
            umap_labels = np.array(umap_partition.membership)
            umap_df["leiden_knn"] = umap_labels
            umap_k = len(umap_partition)
            
            pome_entropy = normalized_cluster_entropy(pome_labels, base=2)
            umap_entropy = normalized_cluster_entropy(umap_labels, base=2)

            ### Compute clustering on normalized UMAP embeddings.
            umap_mat_norm = normalize(umap_mat, norm="l2", axis=1)
            umap_norm_adj = kneighbors_graph(
                umap_mat_norm,
                n_neighbors=n_neighbors,
                mode='connectivity',
                include_self=False,
            )

            sources_u, targets_u = umap_norm_adj.nonzero()

            umap_norm_ig = ig.Graph(
                n=umap_norm_adj.shape[0],
                edges=list(zip(sources_u, targets_u)),
                directed=False
            )

            # Run Leiden
            umap_norm_partition = la.find_partition(
                umap_norm_ig,
                la.RBConfigurationVertexPartition
            )

            umap_norm_labels = np.array(umap_norm_partition.membership)
            umap_df["leiden_norm"] = umap_norm_labels
            umap_norm_k = len(umap_norm_partition)

            ## 3. Metrics
            umap_db_custom = davies_bouldin_score_custom(umap_mat_norm, umap_norm_labels)
            pome_db_custom = davies_bouldin_score_custom(embedding_mat, pome_labels, metric="sqeuclidean")
            
            umap_db_score = davies_bouldin_score(umap_mat, umap_labels)
            pome_db_score = davies_bouldin_score(embedding_mat, pome_labels)
            
            umap_ch_score = calinski_harabasz_score(umap_mat_norm, umap_norm_labels)
            pome_ch_score = calinski_harabasz_score(embedding_mat, pome_labels)
            
            umap_dunn = compute_dunn(umap_mat_norm, umap_norm_labels, metric="euclidean")
            pome_dunn = compute_dunn(embedding_mat, pome_labels, metric="sqeuclidean")
            
            embedding_silhouette = silhouette_score(
                X=embedding_mat,
                labels=pome_labels,
                metric="sqeuclidean"
            )
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(pome_k)
            silhouette_dict['silhouette'].append(embedding_silhouette)
            silhouette_dict["davies_bouldin"].append(pome_db_score)
            silhouette_dict["calinski_harabasz"].append(pome_ch_score)
            silhouette_dict["db_custom"].append(pome_db_custom)
            silhouette_dict["dunn"].append(pome_dunn)
            silhouette_dict['entropy'].append(pome_entropy)
            silhouette_dict['method'].append("POME_KNN_Leiden")
            silhouette_dict['dim'].append(dim)

            umap_silhouette = silhouette_score(
                X=umap_mat, 
                labels=umap_labels,
            )
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(umap_k)
            silhouette_dict['silhouette'].append(umap_silhouette)
            silhouette_dict["davies_bouldin"].append(umap_db_score)
            silhouette_dict["calinski_harabasz"].append(umap_ch_score)
            silhouette_dict["db_custom"].append(umap_db_custom)
            silhouette_dict["dunn"].append(umap_dunn)
            silhouette_dict['entropy'].append(umap_entropy)
            silhouette_dict['method'].append("UMAP_KNN_Leiden")
            silhouette_dict['dim'].append(dim)

        
        elif clustering_type == "DBSCAN":
            
            # --- 1. Run DBSCAN on POME embeddings (Default Params) ---
            # Note: We don't need a distance matrix or graph here; DBSCAN handles it internally.
            db_pome = DBSCAN(metric="euclidean").fit(embedding_mat)
            pome_labels = db_pome.labels_

            # Count clusters (excluding noise -1)
            pome_k = len(set(pome_labels)) - (1 if -1 in pome_labels else 0)

            # --- 2. Run DBSCAN on UMAP embeddings (Default Params) ---
            db_umap = DBSCAN(metric="euclidean").fit(umap_mat)
            umap_labels = db_umap.labels_

            # Count clusters (excluding noise -1)
            umap_k = len(set(umap_labels)) - (1 if -1 in umap_labels else 0)

            # --- 3. Silhouette Metrics ---
            # We must check if at least 2 clusters were found (excluding noise) 
            # because silhouette_score requires at least 2 unique labels.
            if pome_k > 1:
                pome_sil = silhouette_score(embedding_mat, pome_labels)
            else:
                pome_sil = -1 # Flag indicating no clusters found with default eps

            if umap_k > 1:
                umap_sil = silhouette_score(umap_mat, umap_labels)
            else:
                umap_sil = -1

            # --- 4. Update Dictionary ---
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(pome_k)
            silhouette_dict['score'].append(pome_sil)
            silhouette_dict['method'].append("POME_DBSCAN")
            silhouette_dict['dim'].append(dim)

            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(umap_k)
            silhouette_dict['score'].append(umap_sil)
            silhouette_dict['method'].append("UMAP_DBSCAN")
            silhouette_dict['dim'].append(dim)            
        
        elif clustering_type == "OPTICS":
    
            # --- 1. Run OPTICS on POME embeddings ---
            # min_samples=5 is the default, similar to DBSCAN.
            optics_pome = OPTICS(metric="euclidean").fit(embedding_mat)
            pome_labels = optics_pome.labels_
            embedding_df["optics"] = pome_labels

            # Count clusters (excluding noise -1)
            pome_k = len(set(pome_labels)) - (1 if -1 in pome_labels else 0)

            # --- 2. Run OPTICS on UMAP embeddings ---
            optics_umap = OPTICS(metric="euclidean").fit(umap_mat)
            umap_labels = optics_umap.labels_
            umap_df["optics"] = umap_labels

            # Count clusters (excluding noise -1)
            umap_k = len(set(umap_labels)) - (1 if -1 in umap_labels else 0)

            # --- 3. Silhouette Metrics ---
            # Silhouette score still requires at least 2 clusters to be valid.
            if pome_k > 1:
                pome_sil = silhouette_score(embedding_mat, pome_labels)
            else:
                pome_sil = -1 

            if umap_k > 1:
                umap_sil = silhouette_score(umap_mat, umap_labels)
            else:
                umap_sil = -1
                
            pome_entropy = normalized_cluster_entropy(pome_labels, base=2)
            umap_entropy = normalized_cluster_entropy(umap_labels, base=2)

            # --- 4. Update Dictionary ---
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(pome_k)
            silhouette_dict['score'].append(pome_sil)
            silhouette_dict['entropy'].append(pome_entropy)
            silhouette_dict['method'].append("POME_OPTICS")
            silhouette_dict['dim'].append(dim)

            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(umap_k)
            silhouette_dict['score'].append(umap_sil)
            silhouette_dict['entropy'].append(umap_entropy)
            silhouette_dict['method'].append("UMAP_OPTICS")
            silhouette_dict['dim'].append(dim)
            
            # Pvalue computation for supervised stratification analysis.
            # Merge targets df into embedding df and label-encode columns.
            cols_to_encode = []
            result_df = embedding_df.join(targets_df, how="inner")
            results_umap_df = umap_df.join(targets_df, how="inner")
            le = LabelEncoder()
            for col in cols_to_encode:
                result_df[col] = le.fit_transform(result_df[col])
                results_umap_df[col] = le.fit_transform(results_umap_df[col])
                
            # Analyze and plot separation of clusters regarding categorical target variable.
            if dataset == "TCGA_BRCA":
                cat_targets = ["vital_status.demographic"]
            elif dataset == "HANCOCK":
                cat_targets = ["recurrence", "survival_status", "rfs_event"]
            elif dataset == "MIMIC":
                cat_targets = ["label_aplasia", "label_nf"]
            elif dataset == "TCGA_LUAD":
                cat_targets = ["Disease Free Status", "Disease-specific Survival status", "Progression Free Status"]
            else:
                raise ValueError(f"Unkown dataset: {dataset}")
            for target in cat_targets:
                for k in range(2,11):
                    contingency = pd.crosstab(result_df["optics"], result_df[target])
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    pvalues_dict['k'].append(pome_k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(p)
                    pvalues_dict['method'].append("POME")
                    pvalues_dict['run'].append(i)
                    pvalues_dict["dim"].append(dim)

                    contingency = pd.crosstab(results_umap_df[f'optics'], results_umap_df[target])
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    pvalues_dict['k'].append(umap_k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(p)
                    pvalues_dict['method'].append("UMAP")
                    pvalues_dict['run'].append(i)
                    pvalues_dict["dim"].append(dim)

            # Analyze and plot separation of clusters regarding numeric target variable.
            if dataset == "TCGA_BRCA":
                num_targets = ["year_of_death.demographic", "days_to_death.demographic"]
            elif dataset == "HANCOCK":
                num_targets = ["days_to_rfs_event", "days_to_last_information"]
            elif dataset == "MIMIC":
                num_targets = []
            elif dataset == "TCGA_LUAD":
                num_targets = ["Disease Free (Months)", "Months of disease-specific survival", "Progress Free Survival (Months)"]
            
            for target in num_targets:
                for k in range(2, 11):
                    # Group target values by cluster label
                    groups = [
                        result_df[result_df["optics"] == cluster][target].dropna()
                        for cluster in range(k)
                    ]

                    # Perform Kruskal-Wallis test
                    if all(len(g) > 1 for g in groups):  # Kruskal requires at least 2 values per group
                        stat, p = kruskal(*groups)
                        adj_p = p
                    else:
                        adj_p = np.nan

                    pvalues_dict['k'].append(pome_k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(adj_p)
                    pvalues_dict['method'].append("POME")
                    pvalues_dict['run'].append(i)
                    pvalues_dict["dim"].append(dim)

                    # Group target values by cluster label
                    groups = [
                        results_umap_df[results_umap_df[f'optics'] == cluster][target].dropna()
                        for cluster in range(k)
                    ]

                    # Perform Kruskal-Wallis test
                    if all(len(g) > 1 for g in groups):  # Kruskal requires at least 2 values per group
                        stat, p = kruskal(*groups)
                        adj_p = p
                    else:
                        adj_p = np.nan  # fallback if any group is too small

                    pvalues_dict['k'].append(umap_k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(adj_p)
                    pvalues_dict['method'].append("UMAP")
                    pvalues_dict['run'].append(i)
                    pvalues_dict["dim"].append(dim)
        
        elif clustering_type == "HDBSCAN":
            
            # --- 1. Run HDBSCAN on POME embeddings (Default Params) ---
            # min_cluster_size is the primary parameter for HDBSCAN
            hdb_pome = HDBSCAN(metric="euclidean").fit(embedding_mat)
            pome_labels = hdb_pome.labels_
            embedding_df["hdbscan"] = pome_labels

            # Count clusters (excluding noise -1)
            pome_k = len(set(pome_labels)) - (1 if -1 in pome_labels else 0)

            # --- 2. Run HDBSCAN on UMAP embeddings (Default Params) ---
            hdb_umap = HDBSCAN(metric="euclidean").fit(umap_mat)
            umap_labels = hdb_umap.labels_
            umap_df["hdbscan"] = umap_labels

            # Count clusters (excluding noise -1)
            umap_k = len(set(umap_labels)) - (1 if -1 in umap_labels else 0)

            # --- 3. Silhouette Metrics ---
            # Silhouette score requires at least 2 clusters (excluding noise)
            if pome_k > 1:
                pome_sil = silhouette_score(embedding_mat, pome_labels)
            else:
                pome_sil = -1 

            if umap_k > 1:
                umap_sil = silhouette_score(umap_mat, umap_labels)
            else:
                umap_sil = -1
                
            pome_entropy = normalized_cluster_entropy(pome_labels, base=2)
            umap_entropy = normalized_cluster_entropy(umap_labels, base=2)

            # --- 4. Update Dictionary ---
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(pome_k)
            silhouette_dict['score'].append(pome_sil)
            silhouette_dict['entropy'].append(pome_entropy)
            silhouette_dict['method'].append("POME_HDBSCAN")
            silhouette_dict['dim'].append(dim)

            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(umap_k)
            silhouette_dict['score'].append(umap_sil)
            silhouette_dict['entropy'].append(umap_entropy)
            silhouette_dict['method'].append("UMAP_HDBSCAN")
            silhouette_dict['dim'].append(dim)
            
            # Pvalue computation for supervised stratification analysis.
            # Merge targets df into embedding df and label-encode columns.
            cols_to_encode = []
            result_df = embedding_df.join(targets_df, how="inner")
            results_umap_df = umap_df.join(targets_df, how="inner")
            le = LabelEncoder()
            for col in cols_to_encode:
                result_df[col] = le.fit_transform(result_df[col])
                results_umap_df[col] = le.fit_transform(results_umap_df[col])
                
            # Analyze and plot separation of clusters regarding categorical target variable.
            if dataset == "TCGA_BRCA":
                cat_targets = ["vital_status.demographic"]
            elif dataset == "HANCOCK":
                cat_targets = ["recurrence", "survival_status", "rfs_event"]
            elif dataset == "MIMIC":
                cat_targets = ["label_aplasia", "label_nf"]
            elif dataset == "TCGA_LUAD":
                cat_targets = ["Disease Free Status", "Disease-specific Survival status", "Progression Free Status"]
            else:
                raise ValueError(f"Unkown dataset: {dataset}")
            for target in cat_targets:
                for k in range(2,11):
                    contingency = pd.crosstab(result_df["hdbscan"], result_df[target])
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    pvalues_dict['k'].append(pome_k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(p)
                    pvalues_dict['method'].append("POME")
                    pvalues_dict['run'].append(i)
                    pvalues_dict["dim"].append(dim)

                    contingency = pd.crosstab(results_umap_df[f'hdbscan'], results_umap_df[target])
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    pvalues_dict['k'].append(umap_k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(p)
                    pvalues_dict['method'].append("UMAP")
                    pvalues_dict['run'].append(i)
                    pvalues_dict["dim"].append(dim)

            # Analyze and plot separation of clusters regarding numeric target variable.
            if dataset == "TCGA_BRCA":
                num_targets = ["year_of_death.demographic", "days_to_death.demographic"]
            elif dataset == "HANCOCK":
                num_targets = ["days_to_rfs_event", "days_to_last_information"]
            elif dataset == "MIMIC":
                num_targets = []
            elif dataset == "TCGA_LUAD":
                num_targets = ["Disease Free (Months)", "Months of disease-specific survival", "Progress Free Survival (Months)"]
            
            for target in num_targets:
                for k in range(2, 11):
                    # Group target values by cluster label
                    groups = [
                        result_df[result_df["hdbscan"] == cluster][target].dropna()
                        for cluster in range(k)
                    ]

                    # Perform Kruskal-Wallis test
                    if all(len(g) > 1 for g in groups):  # Kruskal requires at least 2 values per group
                        stat, p = kruskal(*groups)
                        adj_p = p
                    else:
                        adj_p = np.nan

                    pvalues_dict['k'].append(pome_k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(adj_p)
                    pvalues_dict['method'].append("POME")
                    pvalues_dict['run'].append(i)
                    pvalues_dict["dim"].append(dim)

                    # Group target values by cluster label
                    groups = [
                        results_umap_df[results_umap_df[f'hdbscan'] == cluster][target].dropna()
                        for cluster in range(k)
                    ]

                    # Perform Kruskal-Wallis test
                    if all(len(g) > 1 for g in groups):  # Kruskal requires at least 2 values per group
                        stat, p = kruskal(*groups)
                        adj_p = p
                    else:
                        adj_p = np.nan  # fallback if any group is too small

                    pvalues_dict['k'].append(umap_k)
                    pvalues_dict['target'].append(target)
                    pvalues_dict['pvalue'].append(adj_p)
                    pvalues_dict['method'].append("UMAP")
                    pvalues_dict['run'].append(i)
                    pvalues_dict["dim"].append(dim)
        
        elif clustering_type == "HDBSCAN_Dunn":
            # --- 1. Run HDBSCAN on POME ---
            hdb_pome = HDBSCAN(metric="euclidean").fit(embedding_mat)
            pome_labels = hdb_pome.labels_
            pome_k = len(set(pome_labels)) - (1 if -1 in pome_labels else 0)

            # --- 2. Run HDBSCAN on UMAP ---
            hdb_umap = HDBSCAN(metric="euclidean").fit(umap_mat)
            umap_labels = hdb_umap.labels_
            umap_k = len(set(umap_labels)) - (1 if -1 in umap_labels else 0)

            # --- 3. Dunn Index Metrics ---
            # We filter out noise (-1) because Dunn index requires defined clusters
            if pome_k > 1:
                pome_mask = pome_labels != -1
                # Calculate only on non-noise points
                pome_dunn = compute_dunn(embedding_mat, pome_labels)
            else:
                pome_dunn = -1 

            if umap_k > 1:
                umap_mask = umap_labels != -1
                umap_dunn = compute_dunn(umap_mat, umap_labels)
            else:
                umap_dunn = -1

            # --- 4. Update Dictionary (Reusing your silhouette_dict structure) ---
            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(pome_k)
            silhouette_dict['score'].append(pome_dunn)
            silhouette_dict['method'].append("POME_HDBSCAN")
            silhouette_dict['dim'].append(dim)

            silhouette_dict['run'].append(i)
            silhouette_dict['k'].append(umap_k)
            silhouette_dict['score'].append(umap_dunn)
            silhouette_dict['method'].append("UMAP_HDBSCAN")
            silhouette_dict['dim'].append(dim)
            
#pvalues_df = pd.DataFrame(pvalues_dict)
silhouette_df = pd.DataFrame(silhouette_dict)
#print(pvalues_df)
#pvalues_df.to_csv(f"{dataset}_{clustering_type}_pvalues.csv", index=False)
silhouette_df.to_csv(f"{dataset}_{clustering_type}_clustering_scores.csv", index=False)