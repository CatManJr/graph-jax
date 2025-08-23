"""
K-Means Clustering - SciKit-Learn Compatible

JAX implementation of k-means clustering based on scikit-learn's _kmeans.py.
This implementation follows the exact logic from scikit-learn but uses JAX operations.

Reference: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple, Optional
from functools import partial


# ============================================================================
# SciKit-Learn Compatible K-means Implementation
# ============================================================================

def _kmeans_single_lloyd(data: jnp.ndarray, sample_weight: jnp.ndarray,
                        centers_init: jnp.ndarray, max_iter: int = 300,
                        tol: float = 1e-4, verbose: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, float, int]:
    """
    Single run of k-means using Lloyd's algorithm.
    
    This follows scikit-learn's _kmeans_single_lloyd implementation.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        sample_weight: Sample weights of shape (n_samples,)
        centers_init: Initial centers of shape (n_clusters, n_features)
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        verbose: Whether to print progress
        
    Returns:
        Tuple of (labels, centers, inertia, n_iter)
    """
    n_samples, n_features = data.shape
    n_clusters = centers_init.shape[0]
    
    # Initialize
    centers = centers_init.copy()
    labels = jnp.zeros(n_samples, dtype=jnp.int32)
    inertia = jnp.inf
    
    # Precompute squared norms of data points (scikit-learn optimization)
    x_squared_norms = jnp.sum(data ** 2, axis=1)
    
    for iteration in range(max_iter):
        prev_inertia = inertia
        
        # Assign points to clusters
        # Compute distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
        centers_squared_norms = jnp.sum(centers ** 2, axis=1)
        distances = (x_squared_norms[:, None] + centers_squared_norms[None, :] - 
                    2 * data @ centers.T)
        
        # Assign to nearest center
        labels = jnp.argmin(distances, axis=1)
        
        # Update centers
        for k in range(n_clusters):
            mask = labels == k
            if jnp.any(mask):
                # Weighted mean of points in cluster
                cluster_weights = sample_weight[mask]
                total_weight = jnp.sum(cluster_weights)
                if total_weight > 0:
                    centers = centers.at[k].set(
                        jnp.sum(data[mask] * cluster_weights[:, None], axis=0) / total_weight
                    )
        
        # Compute inertia
        inertia = jnp.sum(sample_weight * jnp.min(distances, axis=1))
        
        # Check convergence (scikit-learn style)
        if jnp.abs(inertia - prev_inertia) < tol * jnp.abs(prev_inertia):
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
    
    return labels, centers, inertia, iteration + 1


def _init_centroids(data: jnp.ndarray, n_clusters: int, x_squared_norms: jnp.ndarray,
                   init: str = "k-means++", random_state: int = 42) -> jnp.ndarray:
    """
    Initialize centroids following scikit-learn's logic.
    
    Args:
        data: Data matrix
        n_clusters: Number of clusters
        x_squared_norms: Precomputed squared norms of data points
        init: Initialization method ("k-means++" or "random")
        random_state: Random seed
        
    Returns:
        Initial centroids
    """
    n_samples = data.shape[0]
    key = jax.random.PRNGKey(random_state)
    
    if init == "k-means++":
        # K-means++ initialization
        centers = jnp.zeros((n_clusters, data.shape[1]))
        
        # Choose first center randomly
        key, subkey = jax.random.split(key)
        first_center_idx = jax.random.choice(subkey, n_samples)
        centers = centers.at[0].set(data[first_center_idx])
        
        # Choose remaining centers
        for c in range(1, n_clusters):
            # Compute distances to existing centers
            centers_squared_norms = jnp.sum(centers[:c] ** 2, axis=1)
            distances = (x_squared_norms[:, None] + centers_squared_norms[None, :] - 
                        2 * data @ centers[:c].T)
            min_distances = jnp.min(distances, axis=1)
            
            # Choose next center with probability proportional to squared distance
            probabilities = min_distances / jnp.sum(min_distances)
            
            key, subkey = jax.random.split(key)
            next_center_idx = jax.random.choice(subkey, n_samples, p=probabilities)
            centers = centers.at[c].set(data[next_center_idx])
        
        return centers
    
    elif init == "random":
        # Random initialization
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(subkey, n_samples, shape=(n_clusters,), replace=False)
        return data[indices]
    
    else:
        raise ValueError(f"Unknown initialization method: {init}")


def _is_same_clustering(labels1: jnp.ndarray, labels2: jnp.ndarray, n_clusters: int) -> bool:
    """
    Check if two clusterings are the same (up to permutation).
    
    Args:
        labels1: First clustering labels
        labels2: Second clustering labels
        n_clusters: Number of clusters
        
    Returns:
        True if clusterings are the same
    """
    if labels1 is None or labels2 is None:
        return False
    
    # Convert to numpy for easier manipulation
    labels1_np = np.array(labels1)
    labels2_np = np.array(labels2)
    
    # Check if they have the same number of clusters
    if len(np.unique(labels1_np)) != len(np.unique(labels2_np)):
        return False
    
    # Try to find a permutation that makes them equal
    from itertools import permutations
    for perm in permutations(range(n_clusters)):
        permuted_labels2 = np.array([perm[label] for label in labels2_np])
        if np.array_equal(labels1_np, permuted_labels2):
            return True
    
    return False


@partial(jax.jit, static_argnums=(1, 2))
def random_init(data: jnp.ndarray, n_clusters: int, 
                random_state: int = 42) -> jnp.ndarray:
    """
    Random initialization of centroids.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Initial centroids of shape (n_clusters, n_features)
    """
    n_samples, n_features = data.shape
    key = jax.random.PRNGKey(random_state)
    
    # Randomly select n_clusters points as initial centroids
    indices = jax.random.choice(key, n_samples, shape=(n_clusters,), replace=False)
    centroids = data[indices]
    
    return centroids


# ============================================================================
# K-means Core Algorithm - Vectorized
# ============================================================================

@jax.jit
def assign_clusters(data: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized cluster assignment.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        centroids: Centroids matrix of shape (n_clusters, n_features)
        
    Returns:
        Cluster assignments of shape (n_samples,)
    """
    # Compute squared distances to all centroids
    # Shape: (n_samples, n_clusters)
    distances = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    
    # Assign to nearest centroid
    labels = jnp.argmin(distances, axis=1)
    
    return labels


def update_centroids_vectorized(data: jnp.ndarray, labels: jnp.ndarray, 
                               n_clusters: int) -> jnp.ndarray:
    """
    Vectorized centroid update using linear algebra.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        labels: Cluster assignments of shape (n_samples,)
        n_clusters: Number of clusters
        
    Returns:
        Updated centroids of shape (n_clusters, n_features)
    """
    n_samples, n_features = data.shape
    
    # Create one-hot encoding of labels
    # Shape: (n_samples, n_clusters)
    one_hot = jax.nn.one_hot(labels, n_clusters)
    
    # Count points in each cluster
    # Shape: (n_clusters,)
    cluster_sizes = jnp.sum(one_hot, axis=0)
    
    # Compute weighted sum for each cluster
    # Shape: (n_clusters, n_features)
    weighted_sums = jnp.dot(one_hot.T, data)
    
    # Compute centroids, handling empty clusters
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    safe_sizes = jnp.maximum(cluster_sizes[:, None], eps)
    centroids = weighted_sums / safe_sizes
    
    return centroids


@jax.jit
def compute_inertia(data: jnp.ndarray, labels: jnp.ndarray, 
                   centroids: jnp.ndarray) -> float:
    """
    Vectorized inertia computation.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        labels: Cluster assignments of shape (n_samples,)
        centroids: Centroids matrix of shape (n_clusters, n_features)
        
    Returns:
        Inertia (sum of squared distances to centroids)
    """
    # Get centroid for each point
    assigned_centroids = centroids[labels]
    
    # Compute squared distances
    squared_distances = jnp.sum((data - assigned_centroids) ** 2, axis=1)
    
    # Sum all squared distances
    inertia = jnp.sum(squared_distances)
    
    return inertia


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def kmeans_single_run(data: jnp.ndarray, n_clusters: int, 
                     max_iter: int = 300, tol: float = 1e-4,
                     random_state: int = 42) -> Tuple[jnp.ndarray, jnp.ndarray, float, int]:
    """
    Single run of k-means algorithm with vectorized operations.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (labels, centroids, inertia, n_iterations)
    """
    # Initialize centroids using k-means++
    centroids = kmeans_plus_plus_init(data, n_clusters, random_state)
    
    def kmeans_iteration(carry):
        centroids, prev_inertia, n_iter = carry
        
        # Assign points to clusters
        labels = assign_clusters(data, centroids)
        
        # Update centroids using vectorized operation
        new_centroids = update_centroids_vectorized(data, labels, n_clusters)
        
        # Compute new inertia
        new_inertia = compute_inertia(data, labels, new_centroids)
        
        # Check convergence (relative change in inertia) - match scikit-learn exactly
        # scikit-learn uses: |inertia - prev_inertia| < tol * prev_inertia
        converged = jnp.abs(new_inertia - prev_inertia) < tol * jnp.abs(prev_inertia)
        
        return (new_centroids, new_inertia, n_iter + 1), (labels, converged)
    
    # Initialize loop variables
    prev_inertia = jnp.inf
    n_iter = 0
    
    # Run iterations
    init_carry = (centroids, prev_inertia, n_iter)
    
    def cond_fn(carry):
        centroids, prev_inertia, n_iter = carry
        return (n_iter < max_iter) & (prev_inertia == jnp.inf)
    
    def body_fn(carry):
        (centroids, prev_inertia, n_iter), (labels, converged) = kmeans_iteration(carry)
        # If converged, set prev_inertia to a finite value to stop
        prev_inertia = jnp.where(converged, 0.0, prev_inertia)
        return (centroids, prev_inertia, n_iter)
    
    # Run the loop
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    final_centroids, _, final_n_iter = final_carry
    
    # Get final labels and inertia
    final_labels = assign_clusters(data, final_centroids)
    final_inertia = compute_inertia(data, final_labels, final_centroids)
    
    return final_labels, final_centroids, final_inertia, final_n_iter


# ============================================================================
# Main K-means Function - SciKit-Learn Compatible
# ============================================================================

def kmeans_clustering(data: jnp.ndarray,
                     n_clusters: int,
                     n_init: int = 10,  # Match scikit-learn default
                     max_iter: int = 300,
                     tol: float = 1e-4,
                     init: str = "k-means++",
                     random_state: int = 42,
                     sample_weight: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Perform k-means clustering following scikit-learn's exact logic.
    
    This implementation follows the exact steps from scikit-learn's KMeans.fit():
    1. Data validation and preprocessing
    2. Data centering for better distance computations
    3. Multiple initializations with best result selection
    4. Lloyd's algorithm with scikit-learn's convergence criteria
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters
        n_init: Number of random initializations (default: 10)
        max_iter: Maximum number of iterations per run
        tol: Tolerance for convergence
        init: Initialization method ("k-means++" or "random")
        random_state: Random seed for reproducibility
        sample_weight: Sample weights (default: uniform weights)
        
    Returns:
        Cluster labels array of shape (n_samples,)
    """
    # Convert to JAX array
    data = jnp.asarray(data)
    
    # Validate input
    if data.ndim != 2:
        raise ValueError("Data must be 2-dimensional")
    
    n_samples, n_features = data.shape
    
    if n_clusters > n_samples:
        raise ValueError(f"n_clusters ({n_clusters}) cannot be larger than n_samples ({n_samples})")
    
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    
    # Handle trivial cases
    if n_clusters == 1:
        return jnp.zeros(n_samples, dtype=jnp.int32)
    
    if n_clusters == n_samples:
        return jnp.arange(n_samples, dtype=jnp.int32)
    
    # Set up sample weights (scikit-learn style)
    if sample_weight is None:
        sample_weight = jnp.ones(n_samples)
    else:
        sample_weight = jnp.asarray(sample_weight)
        if sample_weight.shape != (n_samples,):
            raise ValueError("sample_weight must have shape (n_samples,)")
    
    # Normalize sample weights
    sample_weight = sample_weight / jnp.sum(sample_weight) * n_samples
    
    # Data centering for more accurate distance computations (scikit-learn optimization)
    data_mean = jnp.mean(data, axis=0)
    data_centered = data - data_mean
    
    # Precompute squared norms of data points (scikit-learn optimization)
    x_squared_norms = jnp.sum(data_centered ** 2, axis=1)
    
    # Run multiple initializations
    best_labels = None
    best_centers = None
    best_inertia = jnp.inf
    best_n_iter = 0
    
    for i in range(n_init):
        # Initialize centers
        centers_init = _init_centroids(data_centered, n_clusters, x_squared_norms, init, random_state + i)
        
        # Run a k-means once
        labels, centers, inertia, n_iter = _kmeans_single_lloyd(
            data_centered, sample_weight, centers_init, max_iter, tol, verbose=False
        )
        
        # Determine if these results are the best so far
        # We choose a new run if it has a better inertia and the clustering is different
        if (best_inertia is None or 
            (inertia < best_inertia and 
             not _is_same_clustering(labels, best_labels, n_clusters))):
            best_labels = labels
            best_centers = centers
            best_inertia = inertia
            best_n_iter = n_iter
    
    # Restore data mean to centers (scikit-learn style)
    if best_centers is not None:
        best_centers = best_centers + data_mean
    
    # Check for degenerate clustering
    distinct_clusters = len(jnp.unique(best_labels))
    if distinct_clusters < n_clusters:
        import warnings
        warnings.warn(
            f"Number of distinct clusters ({distinct_clusters}) found smaller than "
            f"n_clusters ({n_clusters}). Possibly due to duplicate points in X.",
            UserWarning,
            stacklevel=2
        )
    
    return best_labels


# ============================================================================
# Additional K-means Variants
# ============================================================================

def mini_batch_kmeans(
    data: Union[jnp.ndarray, np.ndarray],
    n_clusters: int,
    batch_size: int = 100,
    max_iter: int = 100,
    random_state: Optional[int] = None
) -> jnp.ndarray:
    """
    Mini-batch K-means clustering for large datasets.
    
    This is a faster variant of k-means that uses small random batches
    of data for centroid updates, making it suitable for large datasets.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        batch_size: Size of mini-batches
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Cluster labels for each data point
    """
    data = jnp.asarray(data)
    n_samples, n_features = data.shape
    
    if random_state is None:
        random_state = 42
    
    key = jax.random.PRNGKey(random_state)
    
    # Initialize centroids
    centroids = kmeans_plus_plus_init(data, n_clusters, random_state)
    
    # Mini-batch k-means iterations
    for iteration in range(max_iter):
        # Sample random batch
        key, subkey = jax.random.split(key)
        batch_indices = jax.random.choice(
            subkey, n_samples, shape=(batch_size,), replace=False
        )
        batch_data = data[batch_indices]
        
        # Assign clusters for batch
        batch_labels = assign_clusters(batch_data, centroids)
        
        # Update centroids using batch
        centroids = update_centroids_vectorized(batch_data, batch_labels, n_clusters)
    
    # Final assignment for all data
    final_labels = assign_clusters(data, centroids)
    
    return final_labels


def fuzzy_kmeans(
    data: Union[jnp.ndarray, np.ndarray],
    n_clusters: int,
    fuzziness: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fuzzy C-means clustering.
    
    This variant allows points to belong to multiple clusters with
    different membership degrees.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        fuzziness: Fuzziness coefficient (> 1)
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (hard_labels, membership_matrix)
    """
    data = jnp.asarray(data)
    n_samples, n_features = data.shape
    
    if random_state is None:
        random_state = 42
    
    key = jax.random.PRNGKey(random_state)
    
    # Initialize membership matrix randomly
    membership = jax.random.uniform(key, (n_samples, n_clusters))
    membership = membership / jnp.sum(membership, axis=1, keepdims=True)
    
    for iteration in range(max_iter):
        # Update centroids using fuzzy membership
        weighted_membership = membership ** fuzziness
        centroids = (weighted_membership.T @ data) / jnp.sum(weighted_membership, axis=0, keepdims=True).T
        
        # Update membership matrix
        distances = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        distances = jnp.maximum(distances, 1e-10)  # Avoid division by zero
        
        new_membership = jnp.zeros_like(membership)
        for i in range(n_clusters):
            for j in range(n_clusters):
                new_membership = new_membership.at[:, i].add(
                    (distances[:, i] / distances[:, j]) ** (1 / (fuzziness - 1))
                )
        new_membership = 1.0 / new_membership
        
        # Check convergence
        if jnp.max(jnp.abs(new_membership - membership)) < tol:
            break
        
        membership = new_membership
    
    # Convert to hard labels
    hard_labels = jnp.argmax(membership, axis=1)
    
    return hard_labels, membership
