"""
Algebraic Shortest Path Algorithms using Sparse Matrix Multiplication.

This module implements modern shortest path algorithms that leverage
sparse matrix operations for improved performance on large graphs.
It utilizes Graph-JAX's existing matrix operators for optimal performance.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from functools import partial
from ..graphs import Graph
from ..kernels import spgemm, parallel_spgemm
from ..kernels.parallel_spgemm import simplified_shortest_path_spgemm
from ..kernels.matrix import degree_matrix


@partial(jax.jit, static_argnums=())
def min_plus_multiply(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Perform min-plus matrix multiplication: (A ⊗ B)_ij = min_k(A_ik + B_kj)
    
    This function is JIT-compiled for optimal performance.
    
    Args:
        A: First matrix
        B: Second matrix
        
    Returns:
        Result of min-plus multiplication
    """
    # Reshape for broadcasting
    A_expanded = A[:, :, None]  # (n, m, 1)
    B_expanded = B[None, :, :]  # (1, m, p)
    
    # Compute A_ik + B_kj for all k
    sums = A_expanded + B_expanded  # (n, m, p)
    
    # Take minimum over k dimension
    result = jnp.min(sums, axis=1)  # (n, p)
    
    return result


def algebraic_all_pairs_shortest_paths(
    g: Graph,
    max_iterations: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute all-pairs shortest paths using algebraic approach.
    
    This implementation uses JIT-compiled min-plus multiplication
    for optimal performance. It follows the Floyd-Warshall algorithm
    structure to ensure correctness.
    
    Args:
        g: Graph object
        max_iterations: Maximum number of iterations (default: n-1 for Floyd-Warshall)
        
    Returns:
        Distance matrix of shape (n_nodes, n_nodes)
    """
    n_nodes = g.n_nodes
    
    # Initialize distance matrix
    if g.edge_weights is not None:
        weights = g.edge_weights
    else:
        weights = jnp.ones(g.n_edges, dtype=jnp.float32)
    
    # Create adjacency matrix
    adj_matrix = jnp.full((n_nodes, n_nodes), jnp.inf, dtype=jnp.float32)
    adj_matrix = adj_matrix.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)
    adj_matrix = adj_matrix.at[g.senders, g.receivers].set(weights)
    
    # Initialize distance matrix
    D = adj_matrix.copy()
    
    # Set maximum iterations if not provided
    # Use n-1 iterations to match Floyd-Warshall algorithm
    if max_iterations is None:
        max_iterations = n_nodes - 1
    
    # JIT-compiled update step for better performance
    @partial(jax.jit)
    def update_step(D_current, adj_matrix):
        """Single update step using JIT-compiled min-plus multiplication."""
        return min_plus_multiply(D_current, adj_matrix)
    
    # Algebraic shortest path algorithm with JIT optimization
    # This follows the Floyd-Warshall structure: D[i,j] = min(D[i,j], D[i,k] + D[k,j])
    for iteration in range(max_iterations):
        D_old = D.copy()
        D = update_step(D, adj_matrix)
        
        # Check for convergence (optional, but helps with early termination)
        if jnp.allclose(D, D_old, rtol=1e-6, atol=1e-6):
            break
    
    return D





def min_plus_shortest_paths(
    g: Graph,
    max_power: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute shortest paths using min-plus matrix multiplication.
    
    This method uses repeated min-plus matrix squaring: D^(2^k) represents
    shortest paths with at most 2^k edges using min-plus algebra.
    
    This implementation is fully JIT-compiled for optimal performance.
    
    Args:
        g: Graph object
        max_power: Maximum power to compute (default: log2(n))
        
    Returns:
        Distance matrix of shape (n_nodes, n_nodes)
    """
    n_nodes = g.n_nodes
    
    # Initialize adjacency matrix
    if g.edge_weights is not None:
        weights = g.edge_weights
    else:
        weights = jnp.ones(g.n_edges, dtype=jnp.float32)
    
    adj_matrix = jnp.full((n_nodes, n_nodes), jnp.inf, dtype=jnp.float32)
    adj_matrix = adj_matrix.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)
    adj_matrix = adj_matrix.at[g.senders, g.receivers].set(weights)
    
    # Min-plus matrix power approach - use repeated squaring
    if max_power is None:
        max_power = int(jnp.ceil(jnp.log2(n_nodes)))
    
    D = adj_matrix.copy()
    
    # JIT-compiled squaring step for optimal performance
    @partial(jax.jit)
    def square_step(D_current):
        """Single squaring step using JIT-compiled min-plus multiplication."""
        return min_plus_multiply(D_current, D_current)
    
    for power in range(max_power):
        # Use JIT-compiled min-plus matrix squaring
        # This computes shortest paths with at most 2^(power+1) edges
        D = square_step(D)
    
    return D

@partial(jax.jit, static_argnums=(3, 4))
def _jit_min_plus_shortest_paths(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_weights: jnp.ndarray,
    n_nodes: int,
    max_power: int
) -> jnp.ndarray:
    """
    JIT-compiled min-plus shortest paths implementation.
    
    This is the core implementation that gets JIT-compiled for maximum performance.
    
    Args:
        senders: Source node indices
        receivers: Target node indices
        edge_weights: Edge weights
        n_nodes: Number of nodes
        max_power: Maximum power for matrix squaring
        
    Returns:
        Distance matrix of shape (n_nodes, n_nodes)
    """
    # Create adjacency matrix
    adj_matrix = jnp.full((n_nodes, n_nodes), jnp.inf, dtype=jnp.float32)
    adj_matrix = adj_matrix.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)
    adj_matrix = adj_matrix.at[senders, receivers].set(edge_weights)
    
    # Initialize distance matrix
    D = adj_matrix.copy()
    
    # JIT-compiled squaring step
    def square_step(D_current):
        return min_plus_multiply(D_current, D_current)
    
    # Apply repeated squaring
    for power in range(max_power):
        D = square_step(D)
    
    return D