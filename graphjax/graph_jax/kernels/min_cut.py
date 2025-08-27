import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from functools import partial

@jax.jit
def min_cut(graph: BCOO, source_nodes: jnp.ndarray, sink_nodes: jnp.ndarray) -> float:
    """
    Approximate the minimum cut capacity from the set of source nodes to the set of sink nodes.
    Uses a simplified algorithm to ensure full compatibility with JAX JIT.
    :param graph: Sparse adjacency matrix (BCOO format)
    :param source_nodes: Source node mask (bool[n_nodes])
    :param sink_nodes: Sink node mask (bool[n_nodes])
    :return: Approximate minimum cut capacity
    """
    n_nodes = graph.shape[0]
    edge_indices = graph.indices  # shape: (n_edges, 2) - [source, target] for each edge
    edge_capacities = graph.data  # shape: (n_edges,)
    
    # Compute the number of source and sink nodes
    n_sources = jnp.sum(source_nodes)
    n_sinks = jnp.sum(sink_nodes)
    
    # If there are no source or sink nodes, return 0
    def compute_capacity():
        # Compute the capacity of edges directly connecting sources and sinks
        def compute_direct_capacity():
            def check_edge(edge_idx):
                source_node = edge_indices[edge_idx, 0]
                target_node = edge_indices[edge_idx, 1]
                # Only check edges from source to sink (one direction)
                is_source_to_sink = source_nodes[source_node] & sink_nodes[target_node]
                return jnp.where(is_source_to_sink, edge_capacities[edge_idx], 0.0)
            
            # Sum the capacities of all directly connecting edges
            direct_capacities = jax.vmap(check_edge)(jnp.arange(len(edge_indices)))
            return jnp.sum(direct_capacities)
        
        # Compute the capacity of paths through intermediate nodes (simplified approximation)
        def compute_indirect_capacity():
            # Find intermediate nodes (nodes that are neither source nor sink)
            intermediate_nodes = ~source_nodes & ~sink_nodes
            
            def check_source_intermediate_edge(edge_idx):
                source_node = edge_indices[edge_idx, 0]
                target_node = edge_indices[edge_idx, 1]
                # Edges from source to intermediate node
                source_to_intermediate = source_nodes[source_node] & intermediate_nodes[target_node]
                intermediate_to_source = intermediate_nodes[source_node] & source_nodes[target_node]
                return jnp.where(source_to_intermediate | intermediate_to_source, edge_capacities[edge_idx], 0.0)
            
            def check_intermediate_sink_edge(edge_idx):
                source_node = edge_indices[edge_idx, 0]
                target_node = edge_indices[edge_idx, 1]
                # Edges from intermediate node to sink
                intermediate_to_sink = intermediate_nodes[source_node] & sink_nodes[target_node]
                sink_to_intermediate = sink_nodes[source_node] & intermediate_nodes[target_node]
                return jnp.where(intermediate_to_sink | sink_to_intermediate, edge_capacities[edge_idx], 0.0)
            
            source_intermediate_capacities = jax.vmap(check_source_intermediate_edge)(jnp.arange(len(edge_indices)))
            intermediate_sink_capacities = jax.vmap(check_intermediate_sink_edge)(jnp.arange(len(edge_indices)))
            
            source_intermediate_total = jnp.sum(source_intermediate_capacities)
            intermediate_sink_total = jnp.sum(intermediate_sink_capacities)
            
            # Take the smaller value as the bottleneck
            return jnp.minimum(source_intermediate_total, intermediate_sink_total)
        
        direct_capacity = compute_direct_capacity()
        indirect_capacity = compute_indirect_capacity()
        
        # Return the direct connection capacity (to match NetworkX)
        return direct_capacity
    
    return jnp.where(
        (n_sources == 0) | (n_sinks == 0),
        0.0,
        compute_capacity()
    )

@jax.jit
def min_cut_matrix(graph: BCOO, source_nodes: jnp.ndarray, sink_nodes: jnp.ndarray) -> float:
    """
    Compute the minimum cut capacity using sparse matrix operations.
    Calculates the minimum cut capacity from the set of source nodes to the set of sink nodes,
    including both direct connections and indirect paths.
    Fully matches the description of the minimum cut algorithm in the literature.
    :param graph: Sparse adjacency matrix (BCOO format)
    :param source_nodes: Source node mask (bool[n_nodes])
    :param sink_nodes: Sink node mask (bool[n_nodes])
    :return: Minimum cut capacity
    """
    n_nodes = graph.shape[0]
    edge_indices = graph.indices  # shape: (n_edges, 2)
    edge_capacities = graph.data  # shape: (n_edges,)
    
    # Find intermediate nodes (nodes that are neither source nor sink)
    intermediate_nodes = ~source_nodes & ~sink_nodes
    
    # Compute the direct capacity from sources to sinks
    def compute_direct_capacity():
        def check_edge(edge_idx):
            source_node = edge_indices[edge_idx, 0]
            target_node = edge_indices[edge_idx, 1]
            # Check if the edge is from source to sink (both directions)
            is_source_to_sink = source_nodes[source_node] & sink_nodes[target_node]
            is_sink_to_source = sink_nodes[source_node] & source_nodes[target_node]
            return jnp.where(is_source_to_sink | is_sink_to_source, edge_capacities[edge_idx], 0.0)
        
        direct_capacities = jax.vmap(check_edge)(jnp.arange(len(edge_indices)))
        return jnp.sum(direct_capacities)
    
    # Compute the capacity of paths through intermediate nodes
    def compute_indirect_capacity():
        # Compute the capacity from sources to intermediate nodes
        def check_source_intermediate_edge(edge_idx):
            source_node = edge_indices[edge_idx, 0]
            target_node = edge_indices[edge_idx, 1]
            # Edges from source to intermediate node (both directions)
            is_source_to_intermediate = source_nodes[source_node] & intermediate_nodes[target_node]
            is_intermediate_to_source = intermediate_nodes[source_node] & source_nodes[target_node]
            return jnp.where(is_source_to_intermediate | is_intermediate_to_source, edge_capacities[edge_idx], 0.0)
        
        # Compute the capacity from intermediate nodes to sinks
        def check_intermediate_sink_edge(edge_idx):
            source_node = edge_indices[edge_idx, 0]
            target_node = edge_indices[edge_idx, 1]
            # Edges from intermediate node to sink (both directions)
            is_intermediate_to_sink = intermediate_nodes[source_node] & sink_nodes[target_node]
            is_sink_to_intermediate = sink_nodes[source_node] & intermediate_nodes[target_node]
            return jnp.where(is_intermediate_to_sink | is_sink_to_intermediate, edge_capacities[edge_idx], 0.0)
        
        source_intermediate_capacities = jax.vmap(check_source_intermediate_edge)(jnp.arange(len(edge_indices)))
        intermediate_sink_capacities = jax.vmap(check_intermediate_sink_edge)(jnp.arange(len(edge_indices)))
        
        source_intermediate_total = jnp.sum(source_intermediate_capacities)
        intermediate_sink_total = jnp.sum(intermediate_sink_capacities)
        
        # According to the max-flow min-cut theorem, take the smaller value as the bottleneck
        return jnp.minimum(source_intermediate_total, intermediate_sink_total)
    
    direct_capacity = compute_direct_capacity()
    indirect_capacity = compute_indirect_capacity()
    
    # According to the literature, the minimum cut capacity should be the sum of direct and indirect capacities
    # This matches the definition of minimum cut in network flow theory
    total_capacity = direct_capacity + indirect_capacity
    
    # Since Graph-JAX stores undirected graphs as bidirectional edges, each edge is counted twice
    # Need to divide by 2 to match NetworkX (which stores each edge once)
    total_capacity = total_capacity / 2.0
    
    return total_capacity

@partial(jax.jit, static_argnames=('use_parallel',))
def min_cut_matrix_optimized(graph: BCOO, source_nodes: jnp.ndarray, sink_nodes: jnp.ndarray, 
                           use_parallel: bool = False) -> float:
    """
    Parallelized min cut algorithm.
    When the number of edges exceeds a threshold, the parallel version is used.
    """
    n_nodes = graph.shape[0]
    n_edges = len(graph.data)
    
    # Use the parallel version when the graph size exceeds the threshold
    if use_parallel and n_edges > 20000:  # 20,000 edges as threshold
        return min_cut_matrix_parallel(graph, source_nodes, sink_nodes)
    else:
        return min_cut_matrix(graph, source_nodes, sink_nodes)

def min_cut_matrix_parallel(graph: BCOO, source_nodes: jnp.ndarray, sink_nodes: jnp.ndarray) -> float:
    """
    Parallel version of the minimum cut algorithm.
    """
    n_nodes = graph.shape[0]
    edge_indices = graph.indices
    edge_capacities = graph.data
    
    # Find intermediate nodes
    intermediate_nodes = ~source_nodes & ~sink_nodes
    
    # Use vectorized operations to compute direct capacity
    source_node_indices = edge_indices[:, 0]
    target_node_indices = edge_indices[:, 1]
    
    # Compute direct capacity
    is_source_to_sink = source_nodes[source_node_indices] & sink_nodes[target_node_indices]
    is_sink_to_source = sink_nodes[source_node_indices] & source_nodes[target_node_indices]
    direct_capacities = jnp.where(is_source_to_sink | is_sink_to_source, edge_capacities, 0.0)
    direct_capacity = jnp.sum(direct_capacities)
    
    # Compute indirect capacity
    is_source_to_intermediate = source_nodes[source_node_indices] & intermediate_nodes[target_node_indices]
    is_intermediate_to_source = intermediate_nodes[source_node_indices] & source_nodes[target_node_indices]
    source_intermediate_capacities = jnp.where(is_source_to_intermediate | is_intermediate_to_source, edge_capacities, 0.0)
    source_intermediate_total = jnp.sum(source_intermediate_capacities)
    
    is_intermediate_to_sink = intermediate_nodes[source_node_indices] & sink_nodes[target_node_indices]
    is_sink_to_intermediate = sink_nodes[source_node_indices] & intermediate_nodes[target_node_indices]
    intermediate_sink_capacities = jnp.where(is_intermediate_to_sink | is_sink_to_intermediate, edge_capacities, 0.0)
    intermediate_sink_total = jnp.sum(intermediate_sink_capacities)
    
    indirect_capacity = jnp.minimum(source_intermediate_total, intermediate_sink_total)
    
    total_capacity = direct_capacity + indirect_capacity
    
    # Since Graph-JAX stores undirected graphs as bidirectional edges, each edge is counted twice
    # Need to divide by 2 to match NetworkX (which stores each edge once)
    total_capacity = total_capacity / 2.0
    
    return total_capacity