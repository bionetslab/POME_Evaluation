import pandas as pd
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder, normalize
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_X_y, _safe_indexing
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg as la

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

if __name__ == "__main__":
    
    os.chdir("../../data/input_datasets")
    
    NUM_EMBEDDINGS = 10
    dim_list = [16, 32, 64]
    dataset = "HANCOCK"

    if dataset == "HANCOCK":
        embedding_path = f"../embeddings/{dataset}/embeddings"
        targets_path = "hancock_targets.csv"
        targets_df = pd.read_csv(targets_path, index_col=0)

    elif dataset == "TCGA_LUAD":
        embedding_path = f"../embeddings/{dataset}/embeddings"
        targets_path = "TCGA_LUAD_targets.csv"
        targets_df = pd.read_csv(targets_path, index_col=0)

    elif dataset == "MIMIC":
        embedding_path = f"../embeddings/{dataset}/embeddings"
        targets_path = "mimic_targets.csv"
        targets_df = pd.read_csv(targets_path, index_col=0)

    # Result dictionary for P-values.
    silhouette_dict = {'run': [], 'k' : [], 'silhouette' : [], "davies_bouldin" : [], "db_custom": [], "calinski_harabasz": [], "dunn": [], 'method' : [], 'dim' : [], 'entropy': []}

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
            
            embedding_mat = embedding_df.to_numpy()
            umap_mat = umap_df.to_numpy()

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
    
    cluster_res_df = pd.DataFrame(silhouette_dict)
    cluster_res_df.to_csv(f"{dataset}_leiden_knn_clustering_scores.csv", index=False)