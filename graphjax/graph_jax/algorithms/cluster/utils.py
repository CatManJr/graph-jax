"""
Clustering Utility Functions

Utility functions for clustering algorithms, including metrics computation,
data preprocessing, and visualization helpers.

Reference: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Union, Tuple, Optional
from functools import partial

from ..shortest_path import shortest_paths


# ============================================================================
# Clustering Metrics
# ============================================================================

@jax.jit
def silhouette_score_jax(data: jnp.ndarray, labels: jnp.ndarray) -> float:
    """
    Compute the silhouette score for a clustering.
    
    The silhouette score is a measure of how similar an object is to its own
    cluster compared to other clusters. Values range from -1 to 1.
    
    Args:
        data: Data matrix of shape (n_samples, n_features) or distance matrix
        labels: Cluster labels
        
    Returns:
        Mean silhouette score
    """
    n_samples = data.shape[0]
    unique_labels = jnp.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return 0.0
    
    # Compute pairwise distances (assume data is distance matrix if square)
    if data.shape[0] == data.shape[1]:
        distances = data
    else:
        # Compute Euclidean distances
        distances = jnp.sqrt(jnp.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=2))
    
    silhouette_scores = jnp.zeros(n_samples)
    
    for i in range(n_samples):
        # Same cluster distances
        same_cluster_mask = (labels == labels[i]) & (jnp.arange(n_samples) != i)
        
        if jnp.sum(same_cluster_mask) == 0:
            # Singleton cluster
            silhouette_scores = silhouette_scores.at[i].set(0.0)
            continue
        
        a_i = jnp.mean(distances[i, same_cluster_mask])
        
        # Nearest cluster distances
        b_i = jnp.inf
        for cluster in unique_labels:
            if cluster == labels[i]:
                continue
            
            other_cluster_mask = (labels == cluster)
            if jnp.sum(other_cluster_mask) > 0:
                cluster_dist = jnp.mean(distances[i, other_cluster_mask])
                b_i = jnp.minimum(b_i, cluster_dist)
        
        # Silhouette coefficient
        s_i = (b_i - a_i) / jnp.maximum(a_i, b_i)
        silhouette_scores = silhouette_scores.at[i].set(s_i)
    
    return jnp.mean(silhouette_scores)


@jax.jit
def adjusted_rand_score_jax(labels_true: jnp.ndarray, labels_pred: jnp.ndarray) -> float:
    """
    Compute the Adjusted Rand Index (ARI) between two clusterings.
    
    ARI is a measure of similarity between two clusterings, corrected for chance.
    Values range from -1 to 1, with 1 indicating perfect agreement.
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        Adjusted Rand Index
    """
    n_samples = len(labels_true)
    
    # Build contingency table
    unique_true = jnp.unique(labels_true)
    unique_pred = jnp.unique(labels_pred)
    
    contingency = jnp.zeros((len(unique_true), len(unique_pred)))
    
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            contingency = contingency.at[i, j].set(
                jnp.sum((labels_true == true_label) & (labels_pred == pred_label))
            )
    
    # Compute ARI
    sum_comb_c = jnp.sum(contingency * (contingency - 1) / 2)
    sum_comb_k = jnp.sum(jnp.sum(contingency, axis=1) * (jnp.sum(contingency, axis=1) - 1) / 2)
    sum_comb_r = jnp.sum(jnp.sum(contingency, axis=0) * (jnp.sum(contingency, axis=0) - 1) / 2)
    
    expected_index = sum_comb_k * sum_comb_r / (n_samples * (n_samples - 1) / 2)
    max_index = (sum_comb_k + sum_comb_r) / 2
    
    if max_index == expected_index:
        return 1.0
    
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    return ari


@jax.jit
def normalized_mutual_info_jax(labels_true: jnp.ndarray, labels_pred: jnp.ndarray) -> float:
    """
    Compute the Normalized Mutual Information (NMI) between two clusterings.
    
    NMI measures the mutual information between the two labelings,
    normalized by their entropies.
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        Normalized Mutual Information
    """
    n_samples = len(labels_true)
    
    # Build contingency table
    unique_true = jnp.unique(labels_true)
    unique_pred = jnp.unique(labels_pred)
    
    contingency = jnp.zeros((len(unique_true), len(unique_pred)))
    
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            contingency = contingency.at[i, j].set(
                jnp.sum((labels_true == true_label) & (labels_pred == pred_label))
            )
    
    # Compute marginals
    pi = jnp.sum(contingency, axis=1) / n_samples
    pj = jnp.sum(contingency, axis=0) / n_samples
    
    # Compute mutual information
    mi = 0.0
    for i in range(len(unique_true)):
        for j in range(len(unique_pred)):
            if contingency[i, j] > 0:
                mi += (contingency[i, j] / n_samples) * jnp.log(
                    (contingency[i, j] / n_samples) / (pi[i] * pj[j])
                )
    
    # Compute entropies
    h_true = -jnp.sum(pi * jnp.log(pi + 1e-10))
    h_pred = -jnp.sum(pj * jnp.log(pj + 1e-10))
    
    # Normalized mutual information
    if h_true + h_pred == 0:
        return 1.0
    
    nmi = 2 * mi / (h_true + h_pred)
    return nmi


def compute_cluster_metrics(
    similarity_matrix: jnp.ndarray,
    cluster_labels: jnp.ndarray,
    true_labels: Optional[jnp.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive clustering evaluation metrics.
    
    Args:
        similarity_matrix: Similarity matrix or distance matrix
        cluster_labels: Predicted cluster labels
        true_labels: Ground truth labels (optional)
        
    Returns:
        Dictionary of clustering metrics
    """
    metrics = {}
    
    # Internal metrics (don't require ground truth)
    n_nodes = similarity_matrix.shape[0]
    n_clusters = len(jnp.unique(cluster_labels))
    
    # Silhouette score
    try:
        silhouette = silhouette_score_jax(similarity_matrix, cluster_labels)
        metrics['silhouette'] = float(silhouette)
    except:
        metrics['silhouette'] = 0.0
    

    
    # Intra-cluster vs inter-cluster similarity
    community_matrix = jnp.equal(cluster_labels[:, None], cluster_labels[None, :])
    within_similarity = jnp.sum(similarity_matrix * community_matrix) / (jnp.sum(community_matrix) + 1e-8)
    between_similarity = jnp.sum(similarity_matrix * (1 - community_matrix)) / (jnp.sum(1 - community_matrix) + 1e-8)
    
    metrics['within_similarity'] = float(within_similarity)
    metrics['between_similarity'] = float(between_similarity)
    metrics['n_clusters'] = int(n_clusters)
    
    # External metrics (require ground truth)
    if true_labels is not None:
        try:
            ari = adjusted_rand_score_jax(true_labels, cluster_labels)
            metrics['adjusted_rand_score'] = float(ari)
        except:
            metrics['adjusted_rand_score'] = 0.0
        
        try:
            nmi = normalized_mutual_info_jax(true_labels, cluster_labels)
            metrics['normalized_mutual_info'] = float(nmi)
        except:
            metrics['normalized_mutual_info'] = 0.0
    
    return metrics


# ============================================================================
# Path-based Similarity Functions
# ============================================================================

def normalize_path_lengths(path_matrix: jnp.ndarray, method: str = "min_max") -> jnp.ndarray:
    """
    Normalize path length matrix using different strategies.
    
    Args:
        path_matrix: Matrix of shortest path lengths
        method: Normalization method ("min_max", "z_score", "robust")
        
    Returns:
        Normalized path matrix
    """
    # Convert to numpy for easier handling of boolean indexing
    path_matrix_np = np.array(path_matrix)
    
    # Handle infinite values
    finite_mask = np.isfinite(path_matrix_np)
    finite_values = path_matrix_np[finite_mask]
    
    if len(finite_values) == 0:
        return jnp.zeros_like(path_matrix)
    
    if method == "min_max":
        min_val = np.min(finite_values)
        max_val = np.max(finite_values)
        if max_val > min_val:
            normalized = (path_matrix_np - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(path_matrix_np)
        # Set infinite values to 1
        normalized[~finite_mask] = 1.0
    
    elif method == "z_score":
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values)
        if std_val > 0:
            normalized = (path_matrix_np - mean_val) / std_val
        else:
            normalized = np.zeros_like(path_matrix_np)
        # Set infinite values to max z-score
        max_z = np.max(np.abs(normalized[finite_mask])) if np.sum(finite_mask) > 0 else 0
        normalized[~finite_mask] = max_z
    
    elif method == "robust":
        # Use median and MAD for robust normalization
        median_val = np.median(finite_values)
        mad = np.median(np.abs(finite_values - median_val))
        if mad > 0:
            normalized = (path_matrix_np - median_val) / (1.4826 * mad)  # 1.4826 for normal distribution
        else:
            normalized = np.zeros_like(path_matrix_np)
        # Clip extreme values and set infinite to max
        normalized = np.clip(normalized, -10.0, 10.0)
        max_robust = 10.0
        normalized[~finite_mask] = max_robust
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return jnp.array(normalized)


@partial(jax.jit, static_argnums=(1,))
def path_similarity_matrix(path_matrix: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """
    Convert path length matrix to similarity matrix using Gaussian kernel.
    
    Args:
        path_matrix: Matrix of path lengths or distances
        sigma: Bandwidth parameter for Gaussian kernel
        
    Returns:
        Similarity matrix
    """
    # Apply Gaussian kernel: exp(-d^2 / (2*sigma^2))
    similarity = jnp.exp(-path_matrix**2 / (2 * sigma**2))
    
    # Ensure diagonal is 1 (self-similarity)
    n_nodes = path_matrix.shape[0]
    similarity = similarity.at[jnp.diag_indices(n_nodes)].set(1.0)
    
    return similarity


def create_path_based_similarity(
    graph,
    normalization: str = "min_max",
    sigma: float = 1.0
) -> jnp.ndarray:
    """
    Create similarity matrix based on shortest path distances.
    
    Args:
        graph: Graph object
        normalization: Path normalization method
        sigma: Gaussian kernel bandwidth
        
    Returns:
        Path-based similarity matrix
    """
    # Compute shortest paths
    path_matrix = shortest_paths(graph)
    
    # Normalize path lengths
    normalized_paths = normalize_path_lengths(path_matrix, normalization)
    
    # Convert to similarity
    similarity_matrix = path_similarity_matrix(normalized_paths, sigma)
    
    return similarity_matrix


# ============================================================================
# Clustering Validation
# ============================================================================

def validate_clustering_input(data: jnp.ndarray, n_clusters: int) -> Tuple[jnp.ndarray, int]:
    """
    Validate and preprocess clustering input.
    
    Args:
        data: Input data matrix
        n_clusters: Number of clusters
        
    Returns:
        Tuple of (validated_data, validated_n_clusters)
    """
    # Convert to JAX array
    data = jnp.asarray(data)
    
    # Check dimensions
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got {data.ndim}D")
    
    n_samples = data.shape[0]
    
    # Validate n_clusters
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    
    if n_clusters > n_samples:
        n_clusters = n_samples
    
    return data, n_clusters


def check_clustering_convergence(old_labels: jnp.ndarray, 
                                new_labels: jnp.ndarray,
                                tol: float = 1e-4) -> bool:
    """
    Check if clustering has converged.
    
    Args:
        old_labels: Previous cluster labels
        new_labels: Current cluster labels
        tol: Tolerance for convergence
        
    Returns:
        True if converged
    """
    # Check if labels are identical
    if jnp.array_equal(old_labels, new_labels):
        return True
    
    # Check fraction of changed labels
    changed_fraction = jnp.mean(old_labels != new_labels)
    return changed_fraction < tol


# ============================================================================
# Clustering Post-processing
# ============================================================================

def relabel_clusters_consecutively(labels: jnp.ndarray) -> jnp.ndarray:
    """
    Relabel clusters to use consecutive integers starting from 0.
    
    Args:
        labels: Cluster labels
        
    Returns:
        Relabeled clusters
    """
    unique_labels = jnp.unique(labels)
    n_clusters = len(unique_labels)
    
    # Create mapping from old to new labels
    label_map = jnp.zeros(jnp.max(unique_labels) + 1, dtype=jnp.int32)
    for i, old_label in enumerate(unique_labels):
        label_map = label_map.at[old_label].set(i)
    
    # Apply mapping
    new_labels = label_map[labels]
    
    return new_labels


def merge_small_clusters(labels: jnp.ndarray, 
                        similarity_matrix: jnp.ndarray,
                        min_cluster_size: int = 2) -> jnp.ndarray:
    """
    Merge clusters that are smaller than minimum size.
    
    Args:
        labels: Cluster labels
        similarity_matrix: Similarity matrix
        min_cluster_size: Minimum cluster size
        
    Returns:
        Modified cluster labels
    """
    unique_labels = jnp.unique(labels)
    modified_labels = labels.copy()
    
    for cluster_id in unique_labels:
        cluster_mask = (labels == cluster_id)
        cluster_size = jnp.sum(cluster_mask)
        
        if cluster_size < min_cluster_size:
            # Find most similar cluster to merge with
            cluster_indices = jnp.where(cluster_mask)[0]
            
            best_target = cluster_id
            best_similarity = -jnp.inf
            
            for other_cluster_id in unique_labels:
                if other_cluster_id == cluster_id:
                    continue
                
                other_mask = (labels == other_cluster_id)
                other_indices = jnp.where(other_mask)[0]
                
                # Compute average similarity between clusters
                inter_similarity = jnp.mean(
                    similarity_matrix[jnp.ix_(cluster_indices, other_indices)]
                )
                
                if inter_similarity > best_similarity:
                    best_similarity = inter_similarity
                    best_target = other_cluster_id
            
            # Merge with best target
            if best_target != cluster_id:
                modified_labels = jnp.where(cluster_mask, best_target, modified_labels)
    
    # Relabel consecutively
    return relabel_clusters_consecutively(modified_labels)


# ============================================================================
# Distance Metrics
# ============================================================================

@jax.jit
def euclidean_distance_matrix(X: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise Euclidean distances."""
    X_norm_sq = jnp.sum(X**2, axis=1, keepdims=True)
    distances_sq = X_norm_sq + X_norm_sq.T - 2 * X @ X.T
    return jnp.sqrt(jnp.maximum(distances_sq, 0))


@jax.jit
def cosine_similarity_matrix(X: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise cosine similarities."""
    X_normalized = X / (jnp.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X_normalized @ X_normalized.T


@jax.jit
def rbf_similarity_matrix(X: jnp.ndarray, gamma: float = 1.0) -> jnp.ndarray:
    """Compute RBF (Gaussian) similarity matrix."""
    distances_sq = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    return jnp.exp(-gamma * distances_sq)
