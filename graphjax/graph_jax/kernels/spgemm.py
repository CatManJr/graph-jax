import jax
import jax.numpy as jnp
from functools import partial
from ..graphs import Graph, batch_graphs
from typing import List, Optional

# Optimized internal function with comprehensive JIT compilation
@partial(jax.jit, static_argnames=('n_nodes', 'feature_dim'))
def _spgemm_single_optimized(
    graph: Graph, 
    node_features: jnp.ndarray, 
    n_nodes: int,
    feature_dim: int
) -> jnp.ndarray:
    """
    Optimized internal function: Execute SpGEMM (message passing) on a single graph.
    
    This function is fully JIT-compiled for maximum performance.
    """
    # Optimized message extraction with proper indexing
    messages = node_features.at[graph.senders].get()
    
    # Optimized weight application with proper broadcasting
    if graph.edge_weights is not None:
        # Use broadcast_in_dim for more efficient broadcasting
        weights = jax.lax.broadcast_in_dim(
            graph.edge_weights, 
            shape=messages.shape, 
            broadcast_dimensions=(0,)
        )
        messages *= weights

    # Optimized aggregation using scatter_add
    aggregated_features = jnp.zeros((n_nodes, feature_dim))
    aggregated_features = aggregated_features.at[graph.receivers].add(messages)
    
    return aggregated_features

# JIT-compiled function for handling edge weight broadcasting
@partial(jax.jit)
def _apply_edge_weights(messages: jnp.ndarray, edge_weights: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled function to apply edge weights to messages.
    """
    if edge_weights is not None:
        weights = jax.lax.broadcast_in_dim(
            edge_weights, 
            shape=messages.shape, 
            broadcast_dimensions=(0,)
        )
        return messages * weights
    return messages

# JIT-compiled function for final aggregation
@partial(jax.jit, static_argnames=('n_nodes', 'feature_dim'))
def _aggregate_messages(
    messages: jnp.ndarray, 
    receivers: jnp.ndarray, 
    n_nodes: int, 
    feature_dim: int
) -> jnp.ndarray:
    """
    JIT-compiled function for final message aggregation.
    """
    aggregated_features = jnp.zeros((n_nodes, feature_dim))
    return aggregated_features.at[receivers].add(messages)

# Main optimized SpGEMM function
@partial(jax.jit, static_argnames=('n_nodes', 'feature_dim'))
def _spgemm_single(
    graph: Graph, 
    node_features: jnp.ndarray, 
    n_nodes: int,
    feature_dim: int
) -> jnp.ndarray:
    """
    Internal function: Execute SpGEMM (message passing) on a single graph.
    This version uses the optimized implementation.
    """
    return _spgemm_single_optimized(graph, node_features, n_nodes, feature_dim)

# JIT-compiled batch processing function
@partial(jax.jit, static_argnames=('n_nodes', 'feature_dim'))
def _spgemm_batch_optimized(
    graph: Graph, 
    node_features: jnp.ndarray, 
    n_nodes: int,
    feature_dim: int
) -> jnp.ndarray:
    """
    Optimized batch SpGEMM processing with JIT compilation.
    """
    # Use vmap for efficient batch processing
    vmapped_spgemm = jax.vmap(
        partial(_spgemm_single_optimized, n_nodes=n_nodes, feature_dim=feature_dim),
        in_axes=(0, 0)
    )
    result = vmapped_spgemm(graph, node_features)

    # Apply node mask if available
    if graph.node_mask is not None:
        result *= graph.node_mask[..., None]

    return result

def spgemm(graph: Graph | List[Graph], node_features: jnp.ndarray) -> jnp.ndarray:
    """
    Execute SpGEMM (message passing) on a graph or batch of graphs.
    This function can automatically handle single graphs, a list of graphs, or a pre-batched graph.
    
    Optimized with comprehensive JIT compilation for maximum performance.
    """
    if isinstance(graph, list):
        graph = batch_graphs(graph)
    
    is_batched = graph.senders.ndim > 1

    if not is_batched:
        if node_features.ndim != 2:
            raise ValueError("For single graph, node_features must be 2D (n_nodes, feature_dim).")
        return _spgemm_single(graph, node_features, graph.n_nodes, node_features.shape[1])
    else:
        if node_features.ndim != 3:
            raise ValueError("For batched graphs, node_features must be 3D (batch, n_nodes, feature_dim).")
        
        # Use optimized batch processing
        return _spgemm_batch_optimized(graph, node_features, graph.n_nodes, node_features.shape[2])

# Additional optimized utility functions

@partial(jax.jit)
def spgemm_with_mask(
    graph: Graph, 
    node_features: jnp.ndarray, 
    node_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    JIT-compiled SpGEMM with optional node masking.
    
    Args:
        graph: Graph object
        node_features: Node feature matrix
        node_mask: Optional boolean mask for nodes
        
    Returns:
        Aggregated features with mask applied
    """
    result = _spgemm_single_optimized(graph, node_features, graph.n_nodes, node_features.shape[1])
    
    if node_mask is not None:
        result *= node_mask[..., None]
    
    return result

@partial(jax.jit, static_argnames=('n_nodes', 'feature_dim'))
def spgemm_dense_adjacency(
    adjacency_matrix: jnp.ndarray,
    node_features: jnp.ndarray,
    n_nodes: int,
    feature_dim: int
) -> jnp.ndarray:
    """
    JIT-compiled SpGEMM using dense adjacency matrix representation.
    
    This is useful when the graph is dense or when adjacency matrix is already computed.
    
    Args:
        adjacency_matrix: Dense adjacency matrix (n_nodes, n_nodes)
        node_features: Node feature matrix (n_nodes, feature_dim)
        n_nodes: Number of nodes
        feature_dim: Feature dimension
        
    Returns:
        Aggregated features
    """
    # Use matrix multiplication for dense adjacency
    return jnp.matmul(adjacency_matrix, node_features)