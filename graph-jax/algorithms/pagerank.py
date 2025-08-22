import jax
import jax.numpy as jnp
from functools import partial
from graphs import Graph
from kernels.spgemm import spgemm

@partial(jax.jit, static_argnames=('max_iterations', 'damping_factor', 'tolerance'))
def pagerank(
    graph: Graph, 
    damping_factor: float = 0.85, 
    max_iterations: int = 100, 
    tolerance: float = 1e-06
) -> jnp.ndarray:
    """
    Computes the PageRank of nodes in a graph using the power iteration method.
    This implementation uses the standard PageRank formulation where each node
    distributes its rank equally among its outgoing neighbors.
    """
    n_nodes = graph.n_nodes
    if n_nodes == 0:
        return jnp.array([])

    # Ensure calculations are done in float64 to match NetworkX precision
    dtype = jnp.float64
    
    # 1. Initialize PageRank scores uniformly
    pr = jnp.full(n_nodes, 1.0 / n_nodes, dtype=dtype)

    # 2. Calculate out-degrees for normalization
    # For PageRank, out-degree should be the sum of outgoing edge weights per node
    edge_weights = graph.edge_weights if graph.edge_weights is not None else jnp.ones(graph.n_edges, dtype=dtype)
    edge_weights = edge_weights.astype(dtype)
    
    # Calculate weighted out-degree: sum of weights of outgoing edges for each node
    out_degree = jnp.zeros(n_nodes, dtype=dtype).at[graph.senders].add(edge_weights)
    
    # Handle dangling nodes (nodes with no outgoing edges)
    dangling_nodes_mask = (out_degree == 0)
    
    # The personalization vector (uniform for standard PageRank)
    personalization_vector = jnp.full(n_nodes, 1.0 / n_nodes, dtype=dtype)

    # Cast damping factor to the specified dtype
    damping_factor = jnp.asarray(damping_factor, dtype=dtype)

    def iteration_body(loop_val):
        """A single step of the PageRank power iteration."""
        prev_pr, current_pr, i = loop_val
        
        # --- Calculate contributions from dangling nodes ---
        # Dangling nodes distribute their rank equally to all nodes
        dangling_rank_sum = jnp.sum(jnp.where(dangling_nodes_mask, current_pr, 0))
        
        # --- Calculate contributions from regular nodes ---
        # For weighted PageRank: each edge (u->v) with weight w contributes
        # (PageRank[u] * w) / (sum of weights of outgoing edges from u) to PageRank[v]
        
        # Calculate contribution of each edge
        # sender_contributions = PageRank[sender] * edge_weight / out_degree[sender]
        sender_contributions = jnp.where(
            out_degree[graph.senders] > 0, 
            current_pr[graph.senders] * edge_weights / out_degree[graph.senders], 
            0
        )
        
        # Aggregate contributions to receivers
        distributed_rank = jnp.zeros(n_nodes, dtype=dtype).at[graph.receivers].add(sender_contributions)

        # --- Combine all contributions ---
        # New PageRank = (1-d) * personalization + d * (incoming_rank + dangling_contribution)
        new_pr = (
            (1 - damping_factor) * personalization_vector + 
            damping_factor * (distributed_rank + dangling_rank_sum * personalization_vector)
        )
        
        # Normalize to ensure it's a probability distribution
        new_pr = new_pr / jnp.sum(new_pr)
        
        return (current_pr, new_pr, i + 1)

    def convergence_check(loop_val):
        """Check if the algorithm has converged or max iterations reached."""
        prev_pr, current_pr, i = loop_val
        # Use L1 norm for convergence check (same as NetworkX)
        err = jnp.sum(jnp.abs(current_pr - prev_pr))
        return (err > n_nodes * tolerance) & (i < max_iterations)

    # Run the iterative algorithm
    final_state = jax.lax.while_loop(
        convergence_check,
        iteration_body,
        (jnp.zeros_like(pr), pr, 0)
    )
    
    return final_state[1]