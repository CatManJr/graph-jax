import jax.numpy as jnp
from typing import List
from .graph import Graph

def batch_graphs(graph_list: List[Graph]) -> Graph:
    """
    将一个图的列表批处理成一个单一的、填充过的图对象，以便于 vmap 处理。

    这个函数会自动计算批中最大的节点和边的数量，然后将所有图
    填充到这个大小，并生成相应的掩码。

    Args:
        graph_list (List[Graph]): 一个包含 Graph 对象的 Python 列表。

    Returns:
        Graph: 一个代表整个批次的单一 Graph 对象。
    """
    if not graph_list:
        raise ValueError("图列表不能为空。")

    # 1. 确定批处理维度和填充大小
    batch_size = len(graph_list)
    max_n_nodes = max(g.n_nodes for g in graph_list)
    max_n_edges = max(g.n_edges for g in graph_list)
    
    # 假设所有图的特征维度都相同
    has_features = graph_list[0].node_features is not None
    if has_features:
        feature_dim = graph_list[0].node_features.shape[1]

    # 2. 初始化用于存储填充后数据的列表
    all_senders, all_receivers, all_weights, all_features = [], [], [], []
    all_node_masks, all_edge_masks = [], []

    # 3. 遍历每个图，进行填充并创建掩码
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

    # 4. 将填充后的数据堆叠成一个批处理
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