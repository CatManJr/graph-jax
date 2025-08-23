"""
JAX Implementation of Fuzzy C-Means (FCM) Clustering

This implementation follows the standard FCM algorithm using JAX for automatic differentiation
and GPU acceleration while maintaining numerical accuracy.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any, Optional
from functools import partial


@jax.jit
def fcm_objective(centroids: jnp.ndarray, 
                  membership: jnp.ndarray, 
                  data: jnp.ndarray, 
                  m: float) -> jnp.ndarray:
    """
    Compute FCM objective function.
    
    Args:
        centroids: Cluster centroids (n_clusters, n_features)
        membership: Membership matrix (n_samples, n_clusters)  
        data: Data matrix (n_samples, n_features)
        m: Fuzziness parameter
        
    Returns:
        Objective function value
    """
    n_clusters = centroids.shape[0]
    um = membership ** m  # Apply fuzziness
    
    objective = 0.0
    for i in range(n_clusters):
        # Squared distances from data points to centroid i
        distances_sq = jnp.sum((data - centroids[i]) ** 2, axis=1)
        # Weighted sum by fuzzy membership
        objective += jnp.sum(um[:, i] * distances_sq)
    
    return objective


@jax.jit 
def update_centroids(data: jnp.ndarray, 
                    membership: jnp.ndarray, 
                    m: float) -> jnp.ndarray:
    """
    Update cluster centroids using FCM formula.
    
    Args:
        data: Data matrix (n_samples, n_features)
        membership: Membership matrix (n_samples, n_clusters)
        m: Fuzziness parameter
        
    Returns:
        Updated centroids (n_clusters, n_features)
    """
    um = membership ** m  # Apply fuzziness parameter
    
    # Compute weighted centroids
    numerator = jnp.dot(um.T, data)  # (n_clusters, n_features)
    denominator = jnp.sum(um, axis=0, keepdims=True).T  # (n_clusters, 1)
    
    # Avoid division by zero
    denominator = jnp.maximum(denominator, 1e-10)
    
    centroids = numerator / denominator
    return centroids


@jax.jit
def update_membership(data: jnp.ndarray, 
                     centroids: jnp.ndarray, 
                     m: float) -> jnp.ndarray:
    """
    Update membership matrix using FCM formula.
    
    Args:
        data: Data matrix (n_samples, n_features)
        centroids: Current centroids (n_clusters, n_features)
        m: Fuzziness parameter
        
    Returns:
        Updated membership matrix (n_samples, n_clusters)
    """
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    
    # Compute distances from each point to each centroid
    distances = jnp.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        distances = distances.at[:, i].set(
            jnp.sqrt(jnp.sum((data - centroids[i]) ** 2, axis=1))
        )
    
    # Avoid zero distances
    distances = jnp.maximum(distances, 1e-10)
    
    # FCM membership update formula
    power = 2.0 / (m - 1.0)
    membership = jnp.zeros((n_samples, n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            ratio = distances[:, i] / distances[:, j]
            membership = membership.at[:, i].add(ratio ** power)
    
    membership = 1.0 / membership
    
    # Normalize to ensure rows sum to 1
    row_sums = jnp.sum(membership, axis=1, keepdims=True)
    row_sums = jnp.maximum(row_sums, 1e-10)
    membership = membership / row_sums
    
    return membership


def initialize_membership(n_samples: int, 
                         n_clusters: int, 
                         random_state: int = 42) -> jnp.ndarray:
    """
    Initialize membership matrix randomly.
    
    Args:
        n_samples: Number of data points
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Initial membership matrix (n_samples, n_clusters)
    """
    key = jax.random.PRNGKey(random_state)
    u = jax.random.uniform(key, (n_samples, n_clusters))
    
    # Normalize so each row sums to 1
    u = u / jnp.sum(u, axis=1, keepdims=True)
    
    return u


def initialize_centroids_plus_plus(data: jnp.ndarray, 
                                  n_clusters: int, 
                                  random_state: int = 42) -> jnp.ndarray:
    """
    Initialize centroids using k-means++ strategy.
    
    Args:
        data: Data matrix (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Initial centroids (n_clusters, n_features)
    """
    n_samples, n_features = data.shape
    centroids = jnp.zeros((n_clusters, n_features))
    
    # Convert to numpy for easier indexing
    data_np = np.array(data)
    centroids_np = np.array(centroids)
    
    np.random.seed(random_state)
    
    # Choose first centroid randomly
    first_idx = np.random.choice(n_samples)
    centroids_np[0] = data_np[first_idx]
    
    # Choose remaining centroids
    for i in range(1, n_clusters):
        # Compute distances to nearest centroid
        distances = np.full(n_samples, np.inf)
        for j in range(i):
            dists = np.sum((data_np - centroids_np[j]) ** 2, axis=1)
            distances = np.minimum(distances, dists)
        
        # Choose next centroid with probability proportional to squared distance
        probabilities = distances / np.sum(distances)
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids_np[i] = data_np[next_idx]
    
    return jnp.array(centroids_np)


def fuzzy_c_means_jax(data: jnp.ndarray,
                     n_clusters: int,
                     m: float = 2.0,
                     max_iter: int = 300,
                     tol: float = 1e-4,
                     init: str = 'k-means++',
                     random_state: int = 42) -> Dict[str, Any]:
    """
    JAX implementation of Fuzzy C-Means clustering.
    
    Args:
        data: Data matrix (n_samples, n_features)
        n_clusters: Number of clusters
        m: Fuzziness parameter (> 1.0)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        init: Initialization method ('random' or 'k-means++')
        random_state: Random seed
        
    Returns:
        Dictionary containing clustering results
    """
    n_samples, n_features = data.shape
    
    # Initialize membership matrix
    if init == 'k-means++':
        centroids = initialize_centroids_plus_plus(data, n_clusters, random_state)
        membership = update_membership(data, centroids, m)
    else:
        membership = initialize_membership(n_samples, n_clusters, random_state)
        centroids = update_centroids(data, membership, m)
    
    loss_history = []
    
    for iteration in range(max_iter):
        # Store previous membership for convergence check
        membership_prev = membership
        
        # Update centroids and membership alternately
        centroids = update_centroids(data, membership, m)
        membership = update_membership(data, centroids, m)
        
        # Compute objective function
        objective = fcm_objective(centroids, membership, data, m)
        loss_history.append(float(objective))
        
        # Check convergence
        membership_change = jnp.max(jnp.abs(membership - membership_prev))
        if membership_change < tol:
            break
    
    # Hard assignments
    hard_labels = jnp.argmax(membership, axis=1)
    
    return {
        'labels': hard_labels,
        'membership': membership,
        'centroids': centroids,
        'loss_history': jnp.array(loss_history),
        'final_loss': loss_history[-1] if loss_history else 0.0,
        'iterations': iteration + 1,
        'converged': membership_change < tol
    }


def compute_clustering_metrics(membership: jnp.ndarray) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        membership: Membership matrix (n_samples, n_clusters)
        
    Returns:
        Dictionary of metrics
    """
    # Entropy (measure of fuzziness)
    entropy = -jnp.mean(jnp.sum(membership * jnp.log(membership + 1e-10), axis=1))
    
    # Partition coefficient (measure of crispness)
    pc = jnp.mean(jnp.sum(membership ** 2, axis=1))
    
    # Average maximum membership
    max_membership = jnp.mean(jnp.max(membership, axis=1))
    
    # Cluster balance (how evenly distributed cluster sizes are)
    cluster_sizes = jnp.sum(membership, axis=0)
    balance = jnp.min(cluster_sizes) / jnp.max(cluster_sizes)
    
    return {
        'entropy': float(entropy),
        'partition_coefficient': float(pc),
        'max_membership': float(max_membership),
        'balance': float(balance)
    }
