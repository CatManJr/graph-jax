import jax
import jax.numpy as jnp
from functools import partial
from ..graphs import Graph, batch_graphs
from typing import List

# We no longer need to mark 'graph' as static, JAX handles it as a Pytree.
@partial(jax.jit, static_argnames=('n_nodes', 'feature_dim'))
def _spgemm_single(
    graph: Graph, 
    node_features: jnp.ndarray, 
    n_nodes: int,
    feature_dim: int
) -> jnp.ndarray:
    """Internal function: Execute SpGEMM (message passing) on a single graph."""
    # Fix: Take features from senders, not receivers
    messages = node_features.at[graph.senders].get()
    
    if graph.edge_weights is not None:
        messages *= graph.edge_weights[:, None]

    aggregated_features = jnp.zeros((n_nodes, feature_dim))
    # Fix: Add messages to receivers, not senders
    aggregated_features = aggregated_features.at[graph.receivers].add(messages)
    
    return aggregated_features

def spgemm(graph: Graph | List[Graph], node_features: jnp.ndarray) -> jnp.ndarray:
    """
    Execute SpGEMM (message passing) on a graph or batch of graphs.
    This function can automatically handle single graphs, a list of graphs, or a pre-batched graph.
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
        
        # vmap now works directly on the Pytree, which is much cleaner.
        vmapped_spgemm = jax.vmap(
            partial(_spgemm_single, n_nodes=graph.n_nodes, feature_dim=node_features.shape[2]),
            in_axes=(0, 0)
        )
        result = vmapped_spgemm(graph, node_features)

        if graph.node_mask is not None:
            result *= graph.node_mask[..., None]

        return result