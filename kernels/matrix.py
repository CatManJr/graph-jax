import jax
import jax.numpy as jnp
from graphs import Graph
from functools import partial
from typing import Optional

# --- 内部 JIT 编译的纯函数 ---

# 修改: 将 n_edges 也添加到 static_argnames
@partial(jax.jit, static_argnames=('n_nodes', 'n_edges', 'as_diagonal'))
def _degree_matrix_pure(
    receivers: jnp.ndarray, 
    edge_weights: Optional[jnp.ndarray], # 参数保留，但函数体内不使用
    n_nodes: int, 
    n_edges: int,
    as_diagonal: bool
) -> jnp.ndarray:
    """
    纯函数版本的度矩阵计算。
    修改：此版本计算无权重的度，以匹配 networkx 的默认行为。
    """
    # 忽略 edge_weights，为每条边加 1 来计算无权重的度
    ones = jnp.ones(n_edges, dtype=jnp.float32)
    degrees = jnp.zeros(n_nodes, dtype=jnp.float32).at[receivers].add(ones)
    return jnp.diag(degrees) if as_diagonal else degrees

@partial(jax.jit, static_argnames=('n_nodes',))
def _laplacian_matrix_pure(
    adj: jnp.ndarray,
    n_nodes: int
) -> jnp.ndarray:
    """纯函数版本的拉普拉斯矩阵计算。"""
    degrees = jnp.sum(adj, axis=1)
    deg_matrix = jnp.diag(degrees)
    return deg_matrix - adj

@partial(jax.jit, static_argnames=('n_nodes', 'add_self_loops'))
def _normalized_laplacian_sym_pure(adj: jnp.ndarray, n_nodes: int, add_self_loops: bool) -> jnp.ndarray:
    """
    纯函数：计算对称归一化拉普拉斯矩阵。
    遵循 NetworkX 的标准定义。
    """
    # 修正: 始终使用原始邻接矩阵计算度，这是标准定义
    degrees = jnp.sum(adj, axis=1)
    
    # 如果需要，为最终的邻接矩阵添加自环
    if add_self_loops:
        adj_processed = adj + jnp.eye(n_nodes)
    else:
        adj_processed = adj

    # 防止除以零
    degrees_inv_sqrt = jnp.where(degrees > 0, 1.0 / jnp.sqrt(degrees), 0)
    D_inv_sqrt = jnp.diag(degrees_inv_sqrt)
    
    # L_sym = I - D^(-1/2) A' D^(-1/2)
    # 其中 A' 是可能添加了自环的邻接矩阵
    l_sym = jnp.eye(n_nodes) - D_inv_sqrt @ adj_processed @ D_inv_sqrt
    return l_sym

@partial(jax.jit, static_argnames=('n_nodes', 'add_self_loops'))
def _random_walk_normalized_laplacian_pure(
    adj: jnp.ndarray,
    n_nodes: int,
    add_self_loops: bool
) -> jnp.ndarray:
    """纯函数版本的随机游走归一化拉普拉斯矩阵计算 (L_rw = I - D⁻¹A)。"""
    if add_self_loops:
        adj += jnp.eye(n_nodes)
    
    degrees = jnp.sum(adj, axis=1)
    # 防止除以零
    inv_deg = jnp.where(degrees > 0, 1.0 / degrees, 0)
    d_inv = jnp.diag(inv_deg)

    return jnp.eye(n_nodes) - d_inv @ adj

@partial(jax.jit, static_argnames=('k', 'force_float64'))
def _laplacian_eigensystem_pure(l_sym: jnp.ndarray, k: int, force_float64: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    """纯函数：计算对称矩阵的特征分解。"""
    # 优化: 根据参数决定是否转换为 float64
    if force_float64:
        l_sym = l_sym.astype(jnp.float64)
    
    # 使用 jax.scipy.linalg.eigh 进行高效计算
    # eigh 假定矩阵是厄米特矩阵 (对于实数矩阵即对称矩阵)
    # 它返回的特征值是排序好的
    vals, vecs = jnp.linalg.eigh(l_sym)
    return vals[:k], vecs[:, :k]


# --- 外部调用的封装函数 ---

def degree_matrix(graph: Graph, as_diagonal: bool = True) -> jnp.ndarray:
    """计算图的度。"""
    return _degree_matrix_pure(
        receivers=graph.receivers,
        edge_weights=graph.edge_weights,
        n_nodes=graph.n_nodes,
        n_edges=graph.n_edges,
        as_diagonal=as_diagonal
    )

def laplacian_matrix(graph: Graph) -> jnp.ndarray:
    """计算图的组合拉普拉斯矩阵 (L = D - A)。"""
    adj = graph.to_adjacency_matrix()
    return _laplacian_matrix_pure(adj, n_nodes=graph.n_nodes)

def normalized_laplacian_sym(graph: Graph, add_self_loops: bool = True, use_weights: bool = False) -> jnp.ndarray:
    """
    计算对称归一化拉普拉斯矩阵。

    Args:
        graph (Graph): 输入图。
        add_self_loops (bool): 是否在计算前添加自环。
        use_weights (bool): 是否使用边的权重。默认为 False。

    Returns:
        jnp.ndarray: 对称归一化拉普拉斯矩阵。
    """
    # 修正: 根据 use_weights 参数选择正确的邻接矩阵
    if use_weights:
        adj = graph.to_adjacency_matrix()
    else:
        adj = graph.to_unweighted_adjacency_matrix()
    return _normalized_laplacian_sym_pure(adj, graph.n_nodes, add_self_loops)

def random_walk_normalized_laplacian(graph: Graph, add_self_loops: bool = True) -> jnp.ndarray:
    """
    计算随机游走归一化拉普拉斯矩阵 (L_rw = I - D⁻¹A)。

    Args:
        graph (Graph): 输入图。
        add_self_loops (bool): 是否在计算度之前添加自环。
    
    Returns:
        jnp.ndarray: 随机游走归一化拉普拉斯矩阵。
    """
    adj = graph.to_adjacency_matrix()
    return _random_walk_normalized_laplacian_pure(adj, n_nodes=graph.n_nodes, add_self_loops=add_self_loops)

def laplacian_eigensystem(graph: Graph, k: int, use_weights: bool = False, force_float64: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    计算对称归一化拉普拉斯矩阵的前 k 个特征值和特征向量。

    这对于谱聚类、图傅里叶变换等任务至关重要。

    Args:
        graph (Graph): 输入图。
        k (int): 需要计算的最小特征值/向量的数量。
        use_weights (bool): 是否在计算中使用边的权重。默认为 False。
        force_float64 (bool): 是否强制使用 float64 进行计算以保证数值精度。
                              默认为 True 以匹配 SciPy/NumPy 的结果。
                              在性能敏感的应用中可以设为 False。

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: (特征值, 特征向量)
    """
    # 谱分析通常在不带自环的对称归一化拉普拉斯上进行
    l_sym = normalized_laplacian_sym(graph, add_self_loops=False, use_weights=use_weights)
    return _laplacian_eigensystem_pure(l_sym, k=k, force_float64=force_float64)