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
    """内部函数：对单个图执行 SpGEMM (消息传递)。"""
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
    在图或一批图上执行 SpGEMM (消息传递)。
    这个函数可以自动处理单个图、一个图的列表或一个预先批处理好的图。
    """
    if isinstance(graph, list):
        graph = batch_graphs(graph)
    
    is_batched = graph.senders.ndim > 1

    if not is_batched:
        if node_features.ndim != 2:
            raise ValueError("对于单个图, node_features 必须是2D的 (n_nodes, feature_dim)。")
        return _spgemm_single(graph, node_features, graph.n_nodes, node_features.shape[1])
    else:
        if node_features.ndim != 3:
            raise ValueError("对于批处理图, node_features 必须是3D的 (batch, n_nodes, feature_dim)。")
        
        # vmap now works directly on the Pytree, which is much cleaner.
        vmapped_spgemm = jax.vmap(
            partial(_spgemm_single, n_nodes=graph.n_nodes, feature_dim=node_features.shape[2]),
            in_axes=(0, 0)
        )
        result = vmapped_spgemm(graph, node_features)

        if graph.node_mask is not None:
            result *= graph.node_mask[..., None]

        return result