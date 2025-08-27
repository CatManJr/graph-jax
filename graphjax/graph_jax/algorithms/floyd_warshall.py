#!/usr/bin/env python3
"""
Floyd-Warshall algorithm implementations for Graph-JAX.

This module provides various shortest path computations based on the Floyd-Warshall algorithm,
which computes all-pairs shortest paths in O(n³) time complexity.
"""

import jax
import jax.numpy as jnp
from typing import Optional
from ..graphs import Graph


def shortest_paths(
    g: Graph, 
    max_distance: Optional[float] = None
) -> jnp.ndarray:
    """
    Compute shortest paths between all pairs of nodes using Floyd-Warshall algorithm.
    
    Args:
        g: Graph object
        max_distance: Maximum distance to consider (for disconnected components)
    
    Returns:
        Distance matrix of shape (n_nodes, n_nodes)
    """
    n_nodes = g.n_nodes
    
    # Algorithm: Floyd-Warshall
    # Initialize distance matrix
    if g.edge_weights is not None:
        weights = g.edge_weights
    else:
        weights = jnp.ones(g.n_edges, dtype=jnp.float32)
    
    # Create adjacency matrix with edge weights
    adj_matrix = jnp.full((n_nodes, n_nodes), jnp.inf, dtype=jnp.float32)
    
    # Set diagonal to 0 (distance to self is 0)
    adj_matrix = adj_matrix.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)
    
    # Set edge weights
    adj_matrix = adj_matrix.at[g.senders, g.receivers].set(weights)
    
    # Floyd-Warshall algorithm
    for k in range(n_nodes):
        # Update distances through intermediate node k
        # Algorithm: Floyd-Warshall update step
        dist_through_k = adj_matrix[:, k:k+1] + adj_matrix[k:k+1, :]
        adj_matrix = jnp.minimum(adj_matrix, dist_through_k)
    
    # Handle disconnected components
    if max_distance is not None:
        adj_matrix = jnp.where(adj_matrix > max_distance, max_distance, adj_matrix)
    
    return adj_matrix


def single_source_shortest_paths(
    g: Graph,
    source: int,
    max_distance: Optional[float] = None
) -> jnp.ndarray:
    """
    Compute shortest paths from a single source node to all other nodes.
    Algorithm: Uses all-pairs Floyd-Warshall and selects source row.
    
    Args:
        g: Graph object
        source: Source node index
        max_distance: Maximum distance to consider
    
    Returns:
        Distance array of shape (n_nodes,)
    """
    all_distances = shortest_paths(g, max_distance)
    return all_distances[source]


def path_exists(
    g: Graph,
    source: int,
    target: int,
    max_distance: Optional[float] = None
) -> bool:
    """
    Check if a path exists between two nodes.
    Algorithm: Uses Floyd-Warshall all-pairs shortest paths.
    
    Args:
        g: Graph object
        source: Source node index
        target: Target node index
        max_distance: Maximum distance to consider
    
    Returns:
        True if path exists, False otherwise
    """
    distances = shortest_paths(g, max_distance)
    return distances[source, target] < jnp.inf


def diameter(
    g: Graph,
    max_distance: Optional[float] = None
) -> float:
    """
    Compute the diameter of the graph (maximum shortest path length).
    Algorithm: Uses Floyd-Warshall all-pairs shortest paths.
    
    Args:
        g: Graph object
        max_distance: Maximum distance to consider
    
    Returns:
        Graph diameter
    """
    distances = shortest_paths(g, max_distance)
    # Exclude infinite distances and self-distances
    finite_distances = jnp.where(
        (distances < jnp.inf) & (distances > 0),
        distances,
        -jnp.inf
    )
    return jnp.max(finite_distances)


def average_shortest_path_length(
    g: Graph,
    max_distance: Optional[float] = None
) -> float:
    """
    Compute the average shortest path length in the graph.
    Algorithm: Uses Floyd-Warshall all-pairs shortest paths.
    
    Args:
        g: Graph object
        max_distance: Maximum distance to consider
    
    Returns:
        Average shortest path length
    """
    distances = shortest_paths(g, max_distance)
    # Exclude infinite distances and self-distances
    finite_mask = (distances < jnp.inf) & (distances > 0)
    finite_distances = jnp.where(finite_mask, distances, 0.0)
    
    n_finite_paths = jnp.sum(finite_mask)
    total_distance = jnp.sum(finite_distances)
    
    return total_distance / (n_finite_paths + 1e-8)
