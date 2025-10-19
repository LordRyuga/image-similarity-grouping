import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize

def cluster_embeddings(embeddings, n_clusters=None, method='hierarchical'):
    """
    Cluster embeddings using specified method
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        n_clusters: number of clusters (None for auto-detection)
        method: 'hierarchical' or 'dbscan'
    
    Returns:
        cluster_labels: array of cluster assignments
    """
    # Normalize embeddings for better similarity comparison
    embeddings_normalized = normalize(embeddings)
    
    if method == 'dbscan':
        # DBSCAN - automatically determines number of clusters
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings_normalized)
        
    else:  # hierarchical
        # Determine number of clusters if not specified
        if n_clusters is None:
            # Use sqrt(n) as heuristic for number of clusters
            n_clusters = max(2, int(np.sqrt(len(embeddings))))
            # Cap at reasonable maximum
            n_clusters = min(n_clusters, len(embeddings) // 2)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings_normalized)
    
    return labels

def compute_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix"""
    embeddings_normalized = normalize(embeddings)
    similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)
    return similarity_matrix