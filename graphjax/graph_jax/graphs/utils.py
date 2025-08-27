import jax.numpy as jnp
from typing import List
from .graph import Graph

def batch_graphs(graph_list: List[Graph]) -> Graph:
    """
    Batch a list of graphs into a single, padded graph object for vmap processing.

    This function automatically calculates the maximum number of nodes and edges in the batch,
    then pads all graphs to this size and generates corresponding masks.

    Args:
        graph_list (List[Graph]): A Python list containing Graph objects.

    Returns:
        Graph: A single Graph object representing the entire batch.
    """
    if not graph_list:
        raise ValueError("Graph list cannot be empty.")

    # 1. Determine batch dimensions and padding sizes
    batch_size = len(graph_list)
    max_n_nodes = max(g.n_nodes for g in graph_list)
    max_n_edges = max(g.n_edges for g in graph_list)
    
    # Assume all graphs have the same feature dimensions
    has_features = graph_list[0].node_features is not None
    if has_features:
        feature_dim = graph_list[0].node_features.shape[1]

    # 2. Initialize lists to store padded data
    all_senders, all_receivers, all_weights, all_features = [], [], [], []
    all_node_masks, all_edge_masks = [], []

    # 3. Iterate through each graph, pad and create masks
    for g in graph_list:
        n_edge_pad = max_n_edges - g.n_edges
        all_senders.append(jnp.pad(g.senders, (0, n_edge_pad)))
        all_receivers.append(jnp.pad(g.receivers, (0, n_edge_pad)))
        
        weights = g.edge_weights if g.edge_weights is not None else jnp.ones_like(g.senders)
        all_weights.append(jnp.pad(weights, (0, n_edge_pad)))
        
        n_node_pad = max_n_nodes - g.n_nodes
        if has_features:
            all_features.append(jnp.pad(g.node_features, ((0, n_node_pad), (0, 0))))
        
        all_node_masks.append(jnp.arange(max_n_nodes) < g.n_nodes)
        all_edge_masks.append(jnp.arange(max_n_edges) < g.n_edges)

    # 4. Stack the padded data into a batch
    return Graph(
        senders=jnp.stack(all_senders),
        receivers=jnp.stack(all_receivers),
        edge_weights=jnp.stack(all_weights),
        node_features=jnp.stack(all_features) if has_features else None,
        n_nodes=max_n_nodes,
        n_edges=max_n_edges,
        node_mask=jnp.stack(all_node_masks),
        edge_mask=jnp.stack(all_edge_masks)
    )