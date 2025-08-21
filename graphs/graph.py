import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Optional

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

@struct.dataclass
class Graph:
    """
    一个表示图结构的 JAX 兼容数据类。

    使用稀疏的 COO 格式 (senders, receivers) 存储边。
    所有属性都应该是 JAX 数组，以便与 JIT 兼容。
    图的形状信息 (n_nodes, n_edges) 作为静态元数据存储。
    """
    # 修正: 将没有默认值的属性放在前面
    senders: jnp.ndarray
    receivers: jnp.ndarray
    edge_weights: Optional[jnp.ndarray]
    n_nodes: int = struct.field(pytree_node=False)
    n_edges: int = struct.field(pytree_node=False)
    
    # 修正: 将有默认值的属性放在最后
    node_features: Optional[jnp.ndarray] = None

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