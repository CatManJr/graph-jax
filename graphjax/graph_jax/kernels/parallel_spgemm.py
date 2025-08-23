import jax
import jax.numpy as jnp
from functools import partial
from ..graphs import Graph

@partial(jax.jit, static_argnames=('n_nodes', 'feature_dim'))
def _spgemm_single_graph(
    graph: Graph, 
    node_features: jnp.ndarray, 
    n_nodes: int,
    feature_dim: int
) -> jnp.ndarray:
    """
    内部函数：对单个图和给定的特征执行 SpGEMM。
    这是我们并行化的基本计算单元。
    """
    messages = node_features.at[graph.receivers].get()
    
    if graph.edge_weights is not None:
        # 确保权重被正确广播
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
    使用 pmap 在特征维度上并行化 SpGEMM 计算。
    
    Args:
        graph (Graph): 一个单一的、可能很大的图。
        node_features (jnp.ndarray): 节点特征矩阵 (n_nodes, feature_dim)。

    Returns:
        jnp.ndarray: 聚合后的新节点特征。
    """
    n_devices = jax.local_device_count()
    n_nodes, feature_dim = node_features.shape

    # 1. 检查特征维度是否可以被设备数量整除
    if feature_dim % n_devices != 0:
        raise ValueError(
            f"特征维度 ({feature_dim}) 必须能被设备数 ({n_devices}) 整除。"
        )
    
    # 2. 将特征矩阵在特征轴(axis=1)上切分，并重塑以匹配设备
    #    (n_nodes, n_devices, feature_dim // n_devices)
    features_sharded = node_features.reshape(
        n_nodes, n_devices, feature_dim // n_devices
    )
    #    (n_devices, n_nodes, feature_dim // n_devices)
    features_sharded = jnp.transpose(features_sharded, (1, 0, 2))

    # 3. 定义 pmap 版本的计算函数
    #    in_axes: (None, 0) -> graph被广播到所有设备，features在第一个轴上分发
    pmapped_spgemm = jax.pmap(
        partial(_spgemm_single_graph, n_nodes=n_nodes, feature_dim=feature_dim // n_devices),
        in_axes=(None, 0)
    )

    # 4. 执行并行计算
    result_sharded = pmapped_spgemm(graph, features_sharded)

    # 5. 将结果从各个设备收集回来并拼接
    #    (n_devices, n_nodes, feature_dim // n_devices) -> (n_nodes, n_devices, feature_dim // n_devices)
    result_sharded = jnp.transpose(result_sharded, (1, 0, 2))
    #    (n_nodes, n_devices, feature_dim // n_devices) -> (n_nodes, feature_dim)
    result = result_sharded.reshape(n_nodes, feature_dim)

    return result