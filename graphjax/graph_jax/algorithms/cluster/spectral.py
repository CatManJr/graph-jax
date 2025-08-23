"""
Spectral Clustering - SciKit-Learn Compatible

JAX implementation of spectral clustering based on scikit-learn's _spectral.py.
This implementation follows the exact logic from scikit-learn but uses JAX operations.

Reference: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_spectral.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Union, Optional
from functools import partial

from .kmeans import kmeans_clustering


# ============================================================================
# Laplacian Matrix Computation
# ============================================================================

def _spectral_embedding(adjacency: jnp.ndarray,
                       n_components: int = 8,
                       eigen_solver: str = "arpack",
                       random_state: int = 42,
                       eigen_tol: float = 1e-7,
                       norm_laplacian: bool = True,
                       drop_first: bool = True) -> jnp.ndarray:
    """
    Compute spectral embedding following scikit-learn's _spectral_embedding.
    
    Args:
        adjacency: Adjacency matrix
        n_components: Number of components
        eigen_solver: Eigenvalue solver method
        random_state: Random seed
        eigen_tol: Tolerance for eigenvalue computation
        norm_laplacian: Whether to use normalized Laplacian
        drop_first: Whether to drop the first eigenvector
        
    Returns:
        Spectral embedding matrix
    """
    n_nodes = adjacency.shape[0]
    
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1
    
    # Check if graph is connected (simplified version)
    # In practice, scikit-learn uses more sophisticated connectivity checking
    
    # Compute Laplacian matrix
    if norm_laplacian:
        # Symmetric normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
        degrees = jnp.sum(adjacency, axis=1)
        degrees_safe = jnp.where(degrees > 0, degrees, 1.0)
        degrees_inv_sqrt = 1.0 / jnp.sqrt(degrees_safe)
        D_inv_sqrt = jnp.diag(degrees_inv_sqrt)
        laplacian = jnp.eye(n_nodes) - D_inv_sqrt @ adjacency @ D_inv_sqrt
    else:
        # Unnormalized Laplacian: L = D - A
        degrees = jnp.sum(adjacency, axis=1)
        degree_matrix = jnp.diag(degrees)
        laplacian = degree_matrix - adjacency
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = jax.numpy.linalg.eigh(laplacian)
    
    # Sort by eigenvalue (ascending)
    sorted_indices = jnp.argsort(eigenvals)
    eigenvals = eigenvals[sorted_indices]
    eigenvecs = eigenvecs[:, sorted_indices]
    
    # Select eigenvectors
    if drop_first:
        # Skip the first eigenvector (constant) and take the next n_components
        embedding = eigenvecs[:, 1:n_components]
    else:
        # Take the first n_components eigenvectors
        embedding = eigenvecs[:, :n_components]
    
    # For symmetric normalized Laplacian, recover u = D^(-1/2) x
    if norm_laplacian:
        embedding = embedding / degrees_inv_sqrt[:, None]
    
    return embedding


@jax.jit
def compute_random_walk_laplacian(affinity_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the random walk Laplacian matrix: L_rw = I - D^(-1) A
    
    Args:
        affinity_matrix: Affinity/similarity matrix
        
    Returns:
        Random walk Laplacian matrix
    """
    n_nodes = affinity_matrix.shape[0]
    degrees = jnp.sum(affinity_matrix, axis=1)
    
    # Compute D^(-1)
    degrees_inv = jnp.where(degrees > 0, 1.0 / degrees, 0.0)
    D_inv = jnp.diag(degrees_inv)
    
    # Random walk Laplacian
    laplacian = jnp.eye(n_nodes) - D_inv @ affinity_matrix
    
    return laplacian


# ============================================================================
# Eigenvalue Computation and Processing
# ============================================================================

def compute_spectral_embedding(affinity_matrix: jnp.ndarray,
                             n_clusters: int,
                             norm_laplacian: bool = True,
                             eigen_solver: str = "arpack") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute spectral embedding for clustering.
    
    Args:
        affinity_matrix: Affinity/similarity matrix
        n_clusters: Number of clusters
        norm_laplacian: Whether to use normalized Laplacian
        eigen_solver: Eigenvalue solver method (currently only supports "arpack")
        
    Returns:
        Tuple of (embedding_vectors, eigenvalues)
    """
    # SciKit-Learn uses symmetric normalized Laplacian by default when degree_normalization=True
    # This is different from random walk Laplacian
    if norm_laplacian:
        # Use symmetric normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
        laplacian = compute_normalized_laplacian(affinity_matrix, norm_laplacian=True)
    else:
        # Use unnormalized Laplacian: L = D - A
        laplacian = compute_normalized_laplacian(affinity_matrix, norm_laplacian=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = jax.numpy.linalg.eigh(laplacian)
    
    # Sort by eigenvalue (ascending)
    sorted_indices = jnp.argsort(eigenvals)
    eigenvals = eigenvals[sorted_indices]
    eigenvecs = eigenvecs[:, sorted_indices]
    
    # For symmetric normalized Laplacian, use the first n_clusters eigenvectors
    # (excluding the first one which is constant)
    # This matches scikit-learn's behavior more closely
    if norm_laplacian:
        # Skip the first eigenvector (constant) and take the next n_clusters
        start_idx = 1
        embedding = eigenvecs[:, start_idx:start_idx + n_clusters]
        selected_eigenvals = eigenvals[start_idx:start_idx + n_clusters]
    else:
        # For unnormalized Laplacian, use first n_clusters eigenvectors
        embedding = eigenvecs[:, :n_clusters]
        selected_eigenvals = eigenvals[:n_clusters]
    
    return embedding, selected_eigenvals


@jax.jit
def normalize_embedding(embedding: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize embedding vectors to unit length (L2 normalization).
    
    This matches scikit-learn's normalization strategy for spectral clustering.
    
    Args:
        embedding: Embedding matrix of shape (n_samples, n_components)
        
    Returns:
        Normalized embedding matrix
    """
    # Compute L2 norm for each row
    row_norms = jnp.linalg.norm(embedding, axis=1, keepdims=True)
    
    # Avoid division by zero - use small epsilon instead of 1.0
    epsilon = 1e-10
    row_norms = jnp.where(row_norms < epsilon, epsilon, row_norms)
    
    # Normalize
    normalized_embedding = embedding / row_norms
    
    return normalized_embedding


# ============================================================================
# Main Spectral Clustering Function
# ============================================================================

def spectral_clustering_core(
    affinity_matrix: jnp.ndarray,
    n_clusters: int,
    norm_laplacian: bool = True,
    assign_labels: str = "kmeans",
    random_state: int = 42,
    n_init: int = 10,
    eigen_solver: str = "arpack",
    eigen_tol: float = 1e-7
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Core spectral clustering implementation following scikit-learn's logic.
    
    Args:
        affinity_matrix: Affinity/similarity matrix
        n_clusters: Number of clusters
        norm_laplacian: Whether to use normalized Laplacian
        assign_labels: Label assignment strategy ("kmeans")
        random_state: Random seed for reproducibility
        n_init: Number of k-means initializations
        eigen_solver: Eigenvalue solver method
        eigen_tol: Tolerance for eigenvalue computation
        
    Returns:
        Tuple of (cluster_labels, eigenvalues)
    """
    # Ensure diagonal is zero (self-similarity should be 0)
    affinity_matrix = affinity_matrix - jnp.diag(jnp.diag(affinity_matrix))
    
    # Add small diagonal values for numerical stability
    affinity_matrix = affinity_matrix + jnp.eye(affinity_matrix.shape[0]) * 1e-10
    
    # Compute spectral embedding following scikit-learn
    n_components = n_clusters if n_clusters is not None else n_clusters
    
    # We now obtain the real valued solution matrix to the relaxed Ncut problem
    # solving the eigenvalue problem L_sym x = lambda x and recovering u = D^(-1/2) x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    maps = _spectral_embedding(
        affinity_matrix,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        drop_first=False,  # Keep first eigenvector as in scikit-learn
    )
    
    # Assign labels using k-means with scikit-learn parameters
    if assign_labels == "kmeans":
        cluster_labels = kmeans_clustering(
            maps, 
            n_clusters,
            n_init=n_init,
            max_iter=300,
            tol=1e-4,
            random_state=random_state
        )
    else:
        # Fallback: simple distance-based assignment
        cluster_labels = jnp.argmax(maps, axis=1)
    
    return cluster_labels, jnp.zeros(n_clusters)  # Dummy eigenvalues


def spectral_clustering(affinity_matrix: jnp.ndarray,
                       n_clusters: int = 2,
                       norm_laplacian: bool = True,
                       assign_labels: str = "kmeans",
                       random_state: int = 42,
                       n_init: int = 10,
                       eigen_solver: str = "arpack",
                       eigen_tol: float = 1e-7) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform spectral clustering following scikit-learn's exact logic.
    
    This implementation follows the exact steps from scikit-learn's SpectralClustering.fit():
    1. Data validation and preprocessing
    2. Affinity matrix construction (if needed)
    3. Spectral embedding computation
    4. Label assignment using k-means
    
    Args:
        affinity_matrix: Similarity/affinity matrix of shape (n_samples, n_samples)
        n_clusters: Number of clusters to find
        norm_laplacian: Whether to use normalized Laplacian
        assign_labels: Label assignment strategy ("kmeans")
        random_state: Random seed for reproducibility
        n_init: Number of k-means initializations
        eigen_solver: Eigenvalue solver method
        eigen_tol: Tolerance for eigenvalue computation
        
    Returns:
        Tuple of (cluster_labels, eigenvalues)
    """
    n_samples = affinity_matrix.shape[0]
    
    # Validate input
    if affinity_matrix.shape[1] != n_samples:
        raise ValueError("Affinity matrix must be square")
    
    if n_clusters > n_samples:
        raise ValueError(f"n_clusters ({n_clusters}) cannot be larger than n_samples ({n_samples})")
    
    # Ensure affinity matrix is symmetric and non-negative
    # This is crucial for consistency with scikit-learn
    affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2
    affinity_matrix = jnp.maximum(affinity_matrix, 0.0)
    
    # Ensure diagonal is non-zero for better numerical stability
    # Add small values to diagonal if needed
    diagonal = jnp.diag(affinity_matrix)
    min_diagonal = jnp.min(diagonal)
    if min_diagonal <= 0:
        epsilon = 1e-8
        affinity_matrix = affinity_matrix.at[jnp.diag_indices(n_samples)].set(
            jnp.maximum(diagonal, epsilon)
        )
    
    # Perform spectral clustering
    cluster_labels, eigenvals = spectral_clustering_core(
        affinity_matrix, n_clusters, norm_laplacian, assign_labels, 
        random_state, n_init, eigen_solver, eigen_tol
    )
    
    return cluster_labels, eigenvals


# ============================================================================
# Utility Functions
# ============================================================================

def rbf_kernel(X: jnp.ndarray, gamma: float = 1.0) -> jnp.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        gamma: Kernel coefficient
        
    Returns:
        RBF kernel matrix
    """
    # Compute pairwise squared distances
    X_norm_sq = jnp.sum(X**2, axis=1, keepdims=True)
    distances_sq = X_norm_sq + X_norm_sq.T - 2 * X @ X.T
    
    # Apply RBF kernel
    kernel_matrix = jnp.exp(-gamma * distances_sq)
    
    return kernel_matrix


def knn_graph(X: jnp.ndarray, n_neighbors: int = 10, 
              include_self: bool = True) -> jnp.ndarray:
    """
    Compute k-nearest neighbors graph.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_neighbors: Number of neighbors
        include_self: Whether to include self-connections
        
    Returns:
        KNN adjacency matrix
    """
    n_samples = X.shape[0]
    
    # Compute pairwise distances
    X_norm_sq = jnp.sum(X**2, axis=1, keepdims=True)
    distances_sq = X_norm_sq + X_norm_sq.T - 2 * X @ X.T
    distances = jnp.sqrt(jnp.maximum(distances_sq, 0))
    
    # Find k nearest neighbors for each point
    if include_self:
        k = min(n_neighbors + 1, n_samples)
    else:
        k = min(n_neighbors, n_samples)
    
    # Get indices of k smallest distances
    _, neighbor_indices = jax.lax.top_k(-distances, k)
    
    # Create adjacency matrix
    adjacency = jnp.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(k):
            neighbor = neighbor_indices[i, j]
            if include_self or neighbor != i:
                adjacency = adjacency.at[i, neighbor].set(1.0)
    
    # Make symmetric
    adjacency = jnp.maximum(adjacency, adjacency.T)
    
    return adjacency
