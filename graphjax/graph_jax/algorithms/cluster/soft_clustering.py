"""
Soft Clustering with Gradient Optimization

JAX implementation of soft clustering using automatic differentiation and PyTree.
This implementation leverages JAX's unique advantages for differentiable clustering.

Features:
- Automatic differentiation for gradient-based optimization
- PyTree support for complex data structures
- Soft assignments with membership probabilities
- Differentiable loss functions
- GPU acceleration support
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from collections import namedtuple
from functools import partial
import optax


# ============================================================================
# JAX-Compatible Data Structures
# ============================================================================

# Use named tuples for JAX compatibility
SoftClusteringState = namedtuple('SoftClusteringState', 
                                ['centroids', 'membership', 'temperature', 'iteration', 'loss_history'])

SoftClusteringConfig = namedtuple('SoftClusteringConfig',
                                 ['n_clusters', 'n_features', 'temperature', 'max_iter', 
                                  'learning_rate', 'convergence_tol', 'regularization', 
                                  'entropy_weight', 'sparsity_weight'])


# ============================================================================
# Differentiable Loss Functions
# ============================================================================

def soft_clustering_loss(state: SoftClusteringState, 
                        data: jnp.ndarray,
                        config: SoftClusteringConfig) -> jnp.ndarray:
    """
    FCM-based loss function for soft clustering.
    
    Args:
        state: Current clustering state
        data: Data matrix of shape (n_samples, n_features)
        config: Clustering configuration
        
    Returns:
        FCM objective function value
    """
    m = config.temperature  # Use temperature as fuzziness parameter
    
    # FCM objective: sum of fuzzy membership weighted squared distances
    um = state.membership ** m  # Apply fuzziness parameter (n_samples, n_clusters)
    
    # Vectorized computation of distances
    # data: (n_samples, n_features), centroids: (n_clusters, n_features)
    # Expand data to (n_samples, 1, n_features) and centroids to (1, n_clusters, n_features)
    data_expanded = data[:, None, :]  # (n_samples, 1, n_features)
    centroids_expanded = state.centroids[None, :, :]  # (1, n_clusters, n_features)
    
    # Compute squared distances: (n_samples, n_clusters)
    distances_sq = jnp.sum((data_expanded - centroids_expanded) ** 2, axis=2)
    
    # Weighted sum by fuzzy membership
    objective = jnp.sum(um * distances_sq)
    
    return objective


def graph_aware_loss(state: SoftClusteringState,
                    data: jnp.ndarray,
                    adjacency: jnp.ndarray,
                    config: SoftClusteringConfig) -> jnp.ndarray:
    """
    Graph-aware loss function that considers network structure.
    
    Args:
        state: Current clustering state
        data: Data matrix of shape (n_samples, n_features)
        adjacency: Adjacency matrix of shape (n_samples, n_samples)
        config: Clustering configuration
        
    Returns:
        Total loss value
    """
    # Base clustering loss
    base_loss = soft_clustering_loss(state, data, config)
    
    # Graph smoothness loss
    # Encourage connected nodes to have similar cluster assignments
    membership_diff = state.membership[:, None, :] - state.membership[None, :, :]  # (n_samples, n_samples, n_clusters)
    membership_diff_sq = jnp.sum(membership_diff ** 2, axis=2)  # (n_samples, n_samples)
    
    # Only consider connected pairs
    graph_smoothness = jnp.mean(adjacency * membership_diff_sq)
    
    # Modularity-inspired loss
    # Encourage dense connections within clusters
    cluster_connectivity = state.membership.T @ adjacency @ state.membership  # (n_clusters, n_clusters)
    cluster_sizes = jnp.sum(state.membership, axis=0, keepdims=True)  # (1, n_clusters)
    expected_connectivity = cluster_sizes.T @ cluster_sizes / jnp.sum(adjacency)
    modularity_loss = -jnp.mean(cluster_connectivity - expected_connectivity)
    
    # Total loss
    total_loss = base_loss + 0.1 * graph_smoothness + 0.05 * modularity_loss
    
    return total_loss


# ============================================================================
# Soft Assignment Functions
# ============================================================================

def compute_soft_assignments(data: jnp.ndarray, 
                           centroids: jnp.ndarray,
                           temperature: float = 2.0) -> jnp.ndarray:
    """
    Compute soft cluster assignments using FCM formula.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        centroids: Centroids matrix of shape (n_clusters, n_features)
        temperature: Fuzziness parameter (m in FCM, should be > 1.0)
        
    Returns:
        Soft membership matrix of shape (n_samples, n_clusters)
    """
    m = temperature  # Fuzziness parameter
    
    # Vectorized computation of distances
    # data: (n_samples, n_features), centroids: (n_clusters, n_features)
    data_expanded = data[:, None, :]  # (n_samples, 1, n_features)
    centroids_expanded = centroids[None, :, :]  # (1, n_clusters, n_features)
    
    # Compute distances: (n_samples, n_clusters)
    distances = jnp.sqrt(jnp.sum((data_expanded - centroids_expanded) ** 2, axis=2))
    
    # Avoid zero distances
    distances = jnp.maximum(distances, 1e-10)
    
    # FCM membership update formula (vectorized)
    power = 2.0 / (m - 1.0)
    
    # For each cluster i, compute sum over j of (d_i / d_j)^power
    # distances: (n_samples, n_clusters)
    # We need to compute ratio matrix: (n_samples, n_clusters, n_clusters)
    distances_i = distances[:, :, None]  # (n_samples, n_clusters, 1)
    distances_j = distances[:, None, :]  # (n_samples, 1, n_clusters)
    
    # Compute ratios and sum over j dimension
    ratios = (distances_i / distances_j) ** power  # (n_samples, n_clusters, n_clusters)
    membership = 1.0 / jnp.sum(ratios, axis=2)  # (n_samples, n_clusters)
    
    # Normalize to ensure rows sum to 1 (should be automatic with FCM formula)
    row_sums = jnp.sum(membership, axis=1, keepdims=True)
    row_sums = jnp.maximum(row_sums, 1e-10)
    membership = membership / row_sums
    
    return membership


def compute_graph_aware_soft_assignments(data: jnp.ndarray, 
                                        centroids: jnp.ndarray,
                                        adjacency: jnp.ndarray,
                                        temperature: float = 2.0,
                                        graph_weight: float = 0.1) -> jnp.ndarray:
    """
    Compute graph-aware soft cluster assignments using modified FCM formula.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        centroids: Centroids matrix of shape (n_clusters, n_features)
        adjacency: Adjacency matrix of shape (n_samples, n_samples)
        temperature: Fuzziness parameter (m in FCM, should be > 1.0)
        graph_weight: Weight for graph regularization
        
    Returns:
        Soft membership matrix of shape (n_samples, n_clusters)
    """
    m = temperature  # Fuzziness parameter
    
    # Vectorized computation of distances
    data_expanded = data[:, None, :]  # (n_samples, 1, n_features)
    centroids_expanded = centroids[None, :, :]  # (1, n_clusters, n_features)
    
    # Compute distances: (n_samples, n_clusters)
    distances = jnp.sqrt(jnp.sum((data_expanded - centroids_expanded) ** 2, axis=2))
    distances = jnp.maximum(distances, 1e-10)
    
    # Standard FCM membership computation
    power = 2.0 / (m - 1.0)
    distances_i = distances[:, :, None]  # (n_samples, n_clusters, 1)
    distances_j = distances[:, None, :]  # (n_samples, 1, n_clusters)
    
    ratios = (distances_i / distances_j) ** power  # (n_samples, n_clusters, n_clusters)
    membership = 1.0 / jnp.sum(ratios, axis=2)  # (n_samples, n_clusters)
    
    # Graph regularization: encourage connected nodes to have similar memberships
    if graph_weight > 0:
        # Compute neighborhood influence
        # adjacency: (n_samples, n_samples), membership: (n_samples, n_clusters)
        neighbor_membership = jnp.dot(adjacency, membership)  # (n_samples, n_clusters)
        
        # Normalize by degree
        degree = jnp.sum(adjacency, axis=1, keepdims=True)  # (n_samples, 1)
        degree = jnp.maximum(degree, 1e-10)
        neighbor_membership = neighbor_membership / degree
        
        # Combine standard FCM with graph regularization
        membership = (1 - graph_weight) * membership + graph_weight * neighbor_membership
    
    # Normalize to ensure rows sum to 1
    row_sums = jnp.sum(membership, axis=1, keepdims=True)
    row_sums = jnp.maximum(row_sums, 1e-10)
    membership = membership / row_sums
    
    return membership


def compute_hard_assignments(membership: jnp.ndarray) -> jnp.ndarray:
    """
    Convert soft assignments to hard assignments.
    
    Args:
        membership: Soft membership matrix of shape (n_samples, n_clusters)
        
    Returns:
        Hard cluster labels of shape (n_samples,)
    """
    return jnp.argmax(membership, axis=1)


# ============================================================================
# Gradient-Based Optimization
# ============================================================================

@jax.jit
def update_step(state: SoftClusteringState,
                data: jnp.ndarray,
                config: SoftClusteringConfig) -> Tuple[SoftClusteringState, float]:
    """
    Single FCM optimization step.
    
    Args:
        state: Current clustering state
        data: Data matrix
        config: Clustering configuration
        
    Returns:
        Updated state and loss value
    """
    m = state.temperature  # Fuzziness parameter
    
    # Update centroids using FCM formula
    um = state.membership ** m  # Apply fuzziness parameter
    
    # Compute weighted centroids
    numerator = jnp.dot(um.T, data)  # (n_clusters, n_features)
    denominator = jnp.sum(um, axis=0, keepdims=True).T  # (n_clusters, 1)
    
    # Avoid division by zero
    denominator = jnp.maximum(denominator, 1e-10)
    
    new_centroids = numerator / denominator
    
    # Update membership probabilities using FCM formula
    new_membership = compute_soft_assignments(data, new_centroids, state.temperature)
    
    # Compute loss
    temp_state = SoftClusteringState(
        centroids=new_centroids,
        membership=new_membership,
        temperature=state.temperature,
        iteration=state.iteration,
        loss_history=state.loss_history
    )
    loss = soft_clustering_loss(temp_state, data, config)
    
    # Update loss history
    new_loss_history = state.loss_history.at[state.iteration].set(loss)
    
    # Create new state using named tuple
    new_state = SoftClusteringState(
        centroids=new_centroids,
        membership=new_membership,
        temperature=state.temperature,
        iteration=state.iteration + 1,
        loss_history=new_loss_history
    )
    
    return new_state, loss


@jax.jit
def graph_aware_update_step(state: SoftClusteringState,
                           data: jnp.ndarray,
                           adjacency: jnp.ndarray,
                           config: SoftClusteringConfig) -> Tuple[SoftClusteringState, float]:
    """
    Single optimization step with graph-aware FCM.
    
    Args:
        state: Current clustering state
        data: Data matrix
        adjacency: Adjacency matrix
        config: Clustering configuration
        
    Returns:
        Updated state and loss value
    """
    m = state.temperature  # Fuzziness parameter
    
    # Update centroids using FCM formula (same as standard FCM)
    um = state.membership ** m  # Apply fuzziness parameter
    
    # Compute weighted centroids
    numerator = jnp.dot(um.T, data)  # (n_clusters, n_features)
    denominator = jnp.sum(um, axis=0, keepdims=True).T  # (n_clusters, 1)
    
    # Avoid division by zero
    denominator = jnp.maximum(denominator, 1e-10)
    
    new_centroids = numerator / denominator
    
    # Update membership using graph-aware FCM formula
    new_membership = compute_graph_aware_soft_assignments(
        data, new_centroids, adjacency, state.temperature
    )
    
    # Compute loss
    temp_state = SoftClusteringState(
        centroids=new_centroids,
        membership=new_membership,
        temperature=state.temperature,
        iteration=state.iteration,
        loss_history=state.loss_history
    )
    loss = graph_aware_loss(temp_state, data, adjacency, config)
    
    # Update loss history
    new_loss_history = state.loss_history.at[state.iteration].set(loss)
    
    # Create new state using named tuple
    new_state = SoftClusteringState(
        centroids=new_centroids,
        membership=new_membership,
        temperature=state.temperature,
        iteration=state.iteration + 1,
        loss_history=new_loss_history
    )
    
    return new_state, loss


# ============================================================================
# Main Clustering Functions
# ============================================================================

def soft_clustering(data: jnp.ndarray,
                   n_clusters: int,
                   temperature: float = 2.0,
                   max_iter: int = 300,
                   learning_rate: float = 0.01,
                   convergence_tol: float = 1e-4,
                   random_state: int = 42) -> Dict[str, Any]:
    """
    Perform soft clustering using FCM algorithm.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters
        temperature: Fuzziness parameter (m in FCM, should be > 1.0)
        max_iter: Maximum number of iterations
        learning_rate: Learning rate (not used in FCM, kept for compatibility)
        convergence_tol: Convergence tolerance
        random_state: Random seed
        
    Returns:
        Dictionary containing clustering results
    """
    n_samples, n_features = data.shape
    
    # Initialize configuration with default values
    config = SoftClusteringConfig(
        n_clusters=n_clusters,
        n_features=n_features,
        temperature=temperature,
        max_iter=max_iter,
        learning_rate=learning_rate,
        convergence_tol=convergence_tol,
        regularization=0.01,
        entropy_weight=0.1,
        sparsity_weight=0.05
    )
    
    # Initialize membership matrix randomly
    key = jax.random.PRNGKey(random_state)
    u = jax.random.uniform(key, (n_samples, n_clusters))
    u = u / jnp.sum(u, axis=1, keepdims=True)  # Normalize rows
    
    # Initialize centroids using FCM formula
    um = u ** temperature
    numerator = jnp.dot(um.T, data)
    denominator = jnp.sum(um, axis=0, keepdims=True).T
    denominator = jnp.maximum(denominator, 1e-10)
    centroids = numerator / denominator
    
    # Initialize membership using proper FCM formula
    membership = compute_soft_assignments(data, centroids, temperature)
    
    # Initialize state using named tuple
    state = SoftClusteringState(
        centroids=centroids,
        membership=membership,
        temperature=temperature,
        iteration=0,
        loss_history=jnp.zeros(max_iter)
    )
    
    # Optimization loop with membership-based convergence
    def cond_fn(carry):
        state, prev_membership = carry
        iteration_check = state.iteration < max_iter
        
        # Check membership change for convergence
        def get_membership_change():
            return jnp.max(jnp.abs(state.membership - prev_membership))
        
        def get_initial_change():
            return jnp.inf  # Large value to ensure first iteration runs
        
        membership_change = jax.lax.cond(
            state.iteration > 0,
            get_membership_change,
            get_initial_change
        )
        
        convergence_check = membership_change > convergence_tol
        return jnp.logical_and(iteration_check, convergence_check)
    
    def body_fn(carry):
        state, prev_membership = carry
        new_state, loss = update_step(state, data, config)
        return new_state, state.membership  # Return current membership for next iteration
    
    # Run optimization
    final_state, final_membership = jax.lax.while_loop(
        cond_fn, body_fn, (state, jnp.zeros_like(state.membership))
    )
    
    # Compute final loss
    final_config = SoftClusteringConfig(
        n_clusters=n_clusters,
        n_features=n_features,
        temperature=temperature,
        max_iter=max_iter,
        learning_rate=learning_rate,
        convergence_tol=convergence_tol,
        regularization=0.01,
        entropy_weight=0.1,
        sparsity_weight=0.05
    )
    final_loss = soft_clustering_loss(final_state, data, final_config)
    
    # Convert to hard assignments
    hard_labels = compute_hard_assignments(final_state.membership)
    
    return {
        'labels': hard_labels,
        'membership': final_state.membership,
        'centroids': final_state.centroids,
        'loss_history': final_state.loss_history[:final_state.iteration],
        'final_loss': float(final_loss),
        'iterations': final_state.iteration
    }


def graph_aware_soft_clustering(data: jnp.ndarray,
                               adjacency: jnp.ndarray,
                               n_clusters: int,
                               temperature: float = 2.0,
                               max_iter: int = 300,
                               learning_rate: float = 0.01,
                               convergence_tol: float = 1e-4,
                               random_state: int = 42) -> Dict[str, Any]:
    """
    Perform graph-aware soft clustering using FCM algorithm.
    
    Args:
        data: Data matrix of shape (n_samples, n_features)
        adjacency: Adjacency matrix of shape (n_samples, n_samples)
        n_clusters: Number of clusters
        temperature: Fuzziness parameter (m in FCM, should be > 1.0)
        max_iter: Maximum number of iterations
        learning_rate: Learning rate (not used in FCM, kept for compatibility)
        convergence_tol: Convergence tolerance
        random_state: Random seed
        
    Returns:
        Dictionary containing clustering results
    """
    n_samples, n_features = data.shape
    
    # Initialize configuration with default values
    config = SoftClusteringConfig(
        n_clusters=n_clusters,
        n_features=n_features,
        temperature=temperature,
        max_iter=max_iter,
        learning_rate=learning_rate,
        convergence_tol=convergence_tol,
        regularization=0.01,
        entropy_weight=0.1,
        sparsity_weight=0.05
    )
    
    # Initialize membership matrix randomly (same as standard FCM)
    key = jax.random.PRNGKey(random_state)
    u = jax.random.uniform(key, (n_samples, n_clusters))
    u = u / jnp.sum(u, axis=1, keepdims=True)  # Normalize rows
    
    # Initialize centroids using FCM formula
    um = u ** temperature
    numerator = jnp.dot(um.T, data)
    denominator = jnp.sum(um, axis=0, keepdims=True).T
    denominator = jnp.maximum(denominator, 1e-10)
    centroids = numerator / denominator
    
    # Initialize membership using graph-aware FCM formula
    membership = compute_graph_aware_soft_assignments(data, centroids, adjacency, temperature)
    
    # Initialize state using named tuple
    state = SoftClusteringState(
        centroids=centroids,
        membership=membership,
        temperature=temperature,
        iteration=0,
        loss_history=jnp.zeros(max_iter)
    )
    
    # Optimization loop with membership-based convergence
    def cond_fn(carry):
        state, prev_membership = carry
        iteration_check = state.iteration < max_iter
        
        # Check membership change for convergence
        def get_membership_change():
            return jnp.max(jnp.abs(state.membership - prev_membership))
        
        def get_initial_change():
            return jnp.inf  # Large value to ensure first iteration runs
        
        membership_change = jax.lax.cond(
            state.iteration > 0,
            get_membership_change,
            get_initial_change
        )
        
        convergence_check = membership_change > convergence_tol
        return jnp.logical_and(iteration_check, convergence_check)
    
    def body_fn(carry):
        state, prev_membership = carry
        new_state, loss = graph_aware_update_step(state, data, adjacency, config)
        return new_state, state.membership  # Return current membership for next iteration
    
    # Run optimization
    final_state, final_membership = jax.lax.while_loop(
        cond_fn, body_fn, (state, jnp.zeros_like(state.membership))
    )
    
    # Compute final loss
    final_config = SoftClusteringConfig(
        n_clusters=n_clusters,
        n_features=n_features,
        temperature=temperature,
        max_iter=max_iter,
        learning_rate=learning_rate,
        convergence_tol=convergence_tol,
        regularization=0.01,
        entropy_weight=0.1,
        sparsity_weight=0.05
    )
    final_loss = graph_aware_loss(final_state, data, adjacency, final_config)
    
    # Convert to hard assignments
    hard_labels = compute_hard_assignments(final_state.membership)
    
    return {
        'labels': hard_labels,
        'membership': final_state.membership,
        'centroids': final_state.centroids,
        'loss_history': final_state.loss_history[:final_state.iteration],
        'final_loss': float(final_loss),
        'iterations': final_state.iteration
    }


# ============================================================================
# Advanced Features
# ============================================================================

def adaptive_temperature_scheduling(initial_temp: float = 2.0,
                                  final_temp: float = 0.1,
                                  decay_rate: float = 0.95) -> callable:
    """
    Create adaptive temperature scheduling function.
    
    Args:
        initial_temp: Initial temperature
        final_temp: Final temperature
        decay_rate: Temperature decay rate
        
    Returns:
        Function that returns temperature for given iteration
    """
    def get_temperature(iteration: int) -> float:
        temp = initial_temp * (decay_rate ** iteration)
        return max(temp, final_temp)
    
    return get_temperature


def compute_cluster_quality(membership: jnp.ndarray,
                          adjacency: Optional[jnp.ndarray] = None) -> Dict[str, float]:
    """
    Compute various quality metrics for soft clustering.
    
    Args:
        membership: Soft membership matrix
        adjacency: Optional adjacency matrix for graph-aware metrics
        
    Returns:
        Dictionary of quality metrics
    """
    n_samples, n_clusters = membership.shape
    
    # Entropy of assignments
    log_membership = jnp.log(membership + 1e-8)
    entropy = -jnp.mean(membership * log_membership)
    
    # Sparsity (how clear the assignments are)
    sparsity = jnp.mean(membership ** 2)
    
    # Balance (how evenly distributed the clusters are)
    cluster_sizes = jnp.sum(membership, axis=0)
    balance = jnp.std(cluster_sizes) / (jnp.mean(cluster_sizes) + 1e-8)
    
    metrics = {
        'entropy': float(entropy),
        'sparsity': float(sparsity),
        'balance': float(balance)
    }
    
    # Graph-aware metrics if adjacency is provided
    if adjacency is not None:
        # Modularity
        hard_labels = compute_hard_assignments(membership)
        modularity = compute_modularity(adjacency, hard_labels)
        metrics['modularity'] = float(modularity)
        
        # Conductance
        conductance = compute_conductance(adjacency, hard_labels)
        metrics['conductance'] = float(conductance)
    
    return metrics


def compute_modularity(adjacency: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Compute modularity for given clustering."""
    n_nodes = adjacency.shape[0]
    total_edges = jnp.sum(adjacency)
    
    modularity = 0.0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if labels[i] == labels[j]:
                expected = jnp.sum(adjacency[i]) * jnp.sum(adjacency[j]) / (2 * total_edges)
                modularity += adjacency[i, j] - expected
    
    return float(modularity / (2 * total_edges))


def compute_conductance(adjacency: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Compute conductance for given clustering."""
    unique_labels = jnp.unique(labels)
    conductances = []
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_size = jnp.sum(cluster_mask)
        
        if cluster_size == 0 or cluster_size == adjacency.shape[0]:
            continue
        
        # Internal edges
        internal_edges = jnp.sum(adjacency[cluster_mask][:, cluster_mask])
        
        # External edges
        external_edges = jnp.sum(adjacency[cluster_mask]) - internal_edges
        
        # Conductance
        conductance = external_edges / (2 * internal_edges + external_edges + 1e-8)
        conductances.append(float(conductance))
    
    return float(jnp.mean(jnp.array(conductances))) if conductances else 0.0
