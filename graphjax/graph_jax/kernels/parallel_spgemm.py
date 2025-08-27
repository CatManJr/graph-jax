import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional
from ..graphs import Graph

@partial(jax.jit, static_argnames=('n_nodes', 'feature_dim'))
def _spgemm_single_graph(
    graph: Graph, 
    node_features: jnp.ndarray, 
    n_nodes: int,
    feature_dim: int
) -> jnp.ndarray:
    """
    Internal function: Execute SpGEMM on a single graph with given features.
    This is our basic computation unit for parallelization.
    """
    messages = node_features.at[graph.receivers].get()
    
    if graph.edge_weights is not None:
        # Ensure weights are properly broadcasted
        weights = jax.lax.broadcast_in_dim(
            graph.edge_weights, 
            shape=messages.shape, 
            broadcast_dimensions=(0,)
        )
        messages *= weights

    aggregated_features = jnp.zeros((n_nodes, feature_dim))
    aggregated_features = aggregated_features.at[graph.senders].add(messages)
    
    return aggregated_features

def spgemm_pmap(graph: Graph, node_features: jnp.ndarray) -> jnp.ndarray:
    """
    Parallelize SpGEMM computation in feature dimension using pmap.
    
    Args:
        graph (Graph): A single, potentially large graph.
        node_features (jnp.ndarray): Node feature matrix (n_nodes, feature_dim).

    Returns:
        jnp.ndarray: Aggregated new node features.
    """
    n_devices = jax.local_device_count()
    n_nodes, feature_dim = node_features.shape

    # 1. Check if feature dimension can be divided by device count
    if feature_dim % n_devices != 0:
        raise ValueError(
            f"Feature dimension ({feature_dim}) must be divisible by device count ({n_devices})."
        )
    
    # 2. Split feature matrix along feature axis (axis=1) and reshape to match devices
    #    (n_nodes, n_devices, feature_dim // n_devices)
    features_sharded = node_features.reshape(
        n_nodes, n_devices, feature_dim // n_devices
    )
    #    (n_devices, n_nodes, feature_dim // n_devices)
    features_sharded = jnp.transpose(features_sharded, (1, 0, 2))

    # 3. Define pmap version of computation function
    #    in_axes: (None, 0) -> graph is broadcast to all devices, features distributed on first axis
    pmapped_spgemm = jax.pmap(
        partial(_spgemm_single_graph, n_nodes=n_nodes, feature_dim=feature_dim // n_devices),
        in_axes=(None, 0)
    )

    # 4. Execute parallel computation
    result_sharded = pmapped_spgemm(graph, features_sharded)

    # 5. Collect results from all devices and concatenate
    #    (n_devices, n_nodes, feature_dim // n_devices) -> (n_nodes, n_devices, feature_dim // n_devices)
    result_sharded = jnp.transpose(result_sharded, (1, 0, 2))
    #    (n_nodes, n_devices, feature_dim // n_devices) -> (n_nodes, feature_dim)
    result = result_sharded.reshape(n_nodes, feature_dim)

    return result


# Simplified min-plus multiplication relying on JIT/XLA optimization
@partial(jax.jit)
def optimized_min_plus_multiply(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Optimized min-plus matrix multiplication relying on JIT/XLA automatic optimization.
    
    Let XLA automatically choose the best parallelization strategy based on
    the hardware platform and matrix sizes.
    
    Args:
        A: First matrix, shape (n, n)
        B: Second matrix, shape (n, n)
        
    Returns:
        Result of min-plus multiplication, shape (n, n)
    """
    # Let XLA optimize this operation automatically
    A_expanded = A[:, :, None]  # (n, n, 1)
    B_expanded = B[None, :, :]  # (1, n, n)
    sums = A_expanded + B_expanded  # (n, n, n)
    return jnp.min(sums, axis=1)  # (n, n)


def simplified_shortest_path_spgemm(
    g: Graph,
    max_iterations: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute shortest paths using simplified SpGEMM relying on JIT/XLA optimization.
    
    This implementation trusts JAX's JIT compilation and XLA backend to
    automatically choose the best parallelization strategy for the given
    hardware platform and problem size.
    
    Args:
        g: Graph object
        max_iterations: Maximum number of iterations
        
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
    if max_iterations is None:
        max_iterations = int(jnp.ceil(jnp.log2(n_nodes)))
    
    # JIT-compiled iterative updates
    @partial(jax.jit)
    def update_step(D_current, adj_matrix):
        """Single update step using optimized min-plus multiplication."""
        return optimized_min_plus_multiply(D_current, adj_matrix)
    
    # Iterative updates relying on XLA optimization
    for iteration in range(max_iterations):
        D_old = D.copy()
        D = update_step(D, adj_matrix)
        
        # Check for convergence
        if jnp.allclose(D, D_old, rtol=1e-6, atol=1e-6):
            break
    
    return D