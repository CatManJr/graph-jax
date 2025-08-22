import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Optional
from dataclasses import dataclass
import jax.tree_util

# --- 内部 JIT 编译的纯函数 ---

@partial(jax.jit, static_argnames=('n_nodes', 'n_edges'))
def _to_adjacency_matrix_pure(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_weights: Optional[jnp.ndarray],
    n_nodes: int,
    n_edges: int
) -> jnp.ndarray:
    """将稀疏图数据转换为稠密邻接矩阵的纯函数。"""
    if edge_weights is not None:
        weights = edge_weights
    else:
        # 如果没有提供权重，则假定所有边的权重为 1.0
        weights = jnp.ones(n_edges, dtype=jnp.float32)
    
    adj = jnp.zeros((n_nodes, n_nodes), dtype=weights.dtype)
    adj = adj.at[senders, receivers].set(weights)
    return adj

# --- Graph 数据结构 ---

@dataclass
class Graph:
    """
    A JAX-compatible sparse graph data structure, registered as a JAX Pytree.

    This allows the Graph object to be passed directly into jax.jit, jax.vmap,
    and other JAX transformations. JAX will automatically handle flattening
    and unflattening the graph's array components (leaves).
    """
    senders: jnp.ndarray
    receivers: jnp.ndarray
    edge_weights: jnp.ndarray | None
    node_features: jnp.ndarray | None
    n_nodes: int
    n_edges: int
    node_mask: jnp.ndarray | None = None
    edge_mask: jnp.ndarray | None = None

    def tree_flatten(self):
        """
        Flattens the Graph into a list of dynamic array components (children)
        and a dictionary of static, non-array data (aux_data).
        """
        children = (self.senders, self.receivers, self.edge_weights, self.node_features, self.node_mask, self.edge_mask)
        aux_data = {'n_nodes': self.n_nodes, 'n_edges': self.n_edges}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a Graph from its dynamic children and static aux_data.
        """
        senders, receivers, edge_weights, node_features, node_mask, edge_mask = children
        return cls(
            senders=senders,
            receivers=receivers,
            edge_weights=edge_weights,
            node_features=node_features,
            n_nodes=aux_data['n_nodes'],
            n_edges=aux_data['n_edges'],
            node_mask=node_mask,
            edge_mask=edge_mask
        )

    def to_adjacency_matrix(self) -> jnp.ndarray:
        """将稀疏图转换为稠密的 JAX 邻接矩阵。"""
        return _to_adjacency_matrix_pure(
            senders=self.senders,
            receivers=self.receivers,
            edge_weights=self.edge_weights,
            n_nodes=self.n_nodes,
            n_edges=self.n_edges
        )

    def to_unweighted_adjacency_matrix(self) -> jnp.ndarray:
        """将稀疏图转换为稠密的、无权重的 JAX 邻接矩阵 (所有边权重为1)。"""
        # 强制 edge_weights 为 None，这样纯函数会使用全1的权重
        return _to_adjacency_matrix_pure(
            senders=self.senders,
            receivers=self.receivers,
            edge_weights=None,
            n_nodes=self.n_nodes,
            n_edges=self.n_edges
        )

# Register the Graph class as a custom Pytree node for JAX
jax.tree_util.register_pytree_node_class(Graph)