import jax.numpy as jnp
import networkx as nx
import json
import pandas as pd
from typing import Dict, List, Optional, Type
import warnings
import numpy as np
import csv

from .graph import Graph

def from_networkx(g: nx.Graph, node_feature_key: Optional[str] = None) -> Graph:
    """从 NetworkX 图对象创建 Graph。"""
    if not isinstance(g, (nx.Graph, nx.DiGraph)):
        raise TypeError(f"只支持 nx.Graph 和 nx.DiGraph，但得到的是 {type(g)}")

    n_nodes = g.number_of_nodes()
    
    # 提取边列表和权重
    if g.is_directed():
        # 对于有向图，直接使用边
        edges = list(g.edges(data=True))
        senders = [u for u, v, d in edges]
        receivers = [v for u, v, d in edges]
        weights_list = [d.get('weight', 1.0) for u, v, d in edges]
    else:
        # 对于无向图，为每条边创建两个方向的边
        senders = []
        receivers = []
        weights_list = []
        for u, v, data in g.edges(data=True):
            senders.extend([u, v])
            receivers.extend([v, u])
            weight = data.get('weight', 1.0)
            weights_list.extend([weight, weight])

    # 转换列表为 JAX 数组
    senders = jnp.array(senders, dtype=jnp.int32)
    receivers = jnp.array(receivers, dtype=jnp.int32)
    
    # 修正: n_edges 必须是稀疏表示中边的实际数量
    n_edges = len(senders)

    # 处理权重
    if any(w != 1.0 for w in weights_list):
        weights = jnp.array(weights_list, dtype=jnp.float32)
    else:
        weights = None

    # 提取节点特征
    node_features = None
    if node_feature_key and g.nodes:
        first_node_data = next(iter(g.nodes(data=True)))[1]
        if node_feature_key in first_node_data:
            try:
                node_features_list = [g.nodes[i][node_feature_key] for i in range(n_nodes)]
                
                # 修正: 检查特征是否为字符串，如果是则进行编码
                if node_features_list and isinstance(node_features_list[0], str):
                    # 检测到字符串特征，执行标签编码
                    unique_labels = sorted(list(set(node_features_list)))
                    label_map = {label: i for i, label in enumerate(unique_labels)}
                    warnings.warn(
                        f"字符串特征 '{node_feature_key}' 被自动编码为整数: {label_map}"
                    )
                    encoded_features = [label_map[label] for label in node_features_list]
                    node_features = jnp.array(encoded_features)
                else:
                    # 否则，假定为数值特征
                    node_features = jnp.array(node_features_list)

                if node_features.ndim == 1:
                    node_features = node_features[:, None]
            except (KeyError, TypeError) as e:
                warnings.warn(f"警告: 提取或转换特征时出错: {e}。未提取节点特征。")
        else:
            warnings.warn(f"警告: 节点没有特征键 '{node_feature_key}'。未提取节点特征。")

    return Graph(
        senders=senders,
        receivers=receivers,
        edge_weights=weights,
        n_nodes=n_nodes,
        n_edges=n_edges, # 使用修正后的 n_edges
        node_features=node_features
    )

def from_json(file_path: str) -> Graph:
    """从 JSON 文件加载图。"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    n_nodes = len(data['nodes'])
    
    # 提取边
    senders = jnp.array([link['source'] for link in data['links']], dtype=jnp.int32)
    receivers = jnp.array([link['target'] for link in data['links']], dtype=jnp.int32)
    n_edges = len(senders) # <--- 计算边的数量
    
    # 提取权重 (如果存在)
    if data['links'] and 'weight' in data['links'][0]:
        weights = jnp.array([link['weight'] for link in data['links']], dtype=jnp.float32)
    else:
        weights = jnp.ones(n_edges, dtype=jnp.float32)

    node_features = None
    if data['nodes'] and 'features' in data['nodes'][0]:
        try:
            features_list = [jnp.atleast_1d(node['features']) for node in data['nodes']]
            node_features = jnp.stack(features_list)
        except (KeyError, TypeError):
            warnings.warn("警告: 并非所有节点都有 'features' 字段或格式不正确。未提取节点特征。")

    return Graph(
        senders=senders,
        receivers=receivers,
        edge_weights=weights,
        n_nodes=n_nodes,
        n_edges=n_edges, # <--- 传入 n_edges
        node_features=node_features
    )

def from_csv(file_path: str) -> Graph:
    """从 CSV 文件加载图。"""
    try:
        # 尝试读取，允许不均匀的行
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            edges_list = [row for row in reader if row]
        
        if not edges_list:
            # 处理空文件的情况
            return Graph(
                senders=jnp.array([], dtype=jnp.int32),
                receivers=jnp.array([], dtype=jnp.int32),
                edge_weights=None,
                n_nodes=0,
                n_edges=0,
                node_features=None
            )

        edges = np.array(edges_list, dtype=float)

    except (IOError, ValueError) as e:
        print(f"读取 CSV 文件时出错: {e}")
        return None

    senders = jnp.array(edges[:, 0], dtype=jnp.int32)
    receivers = jnp.array(edges[:, 1], dtype=jnp.int32)
    n_edges = len(senders) # <--- 计算边的数量
    
    # 假定第三列是权重 (如果存在)
    if edges.shape[1] > 2:
        weights = jnp.array(edges[:, 2], dtype=jnp.float32)
    else:
        weights = jnp.ones(n_edges, dtype=jnp.float32)

    # 从边推断节点数 (处理图中可能存在的孤立节点)
    if n_edges > 0:
        n_nodes = int(jnp.maximum(jnp.max(senders), jnp.max(receivers))) + 1
    else:
        n_nodes = 0
    
    # CSV 格式通常不包含节点特征，所以设为 None
    node_features = None
    
    return Graph(
        senders=senders,
        receivers=receivers,
        edge_weights=weights,
        n_nodes=n_nodes,
        n_edges=n_edges, # <--- 传入 n_edges
        node_features=node_features
    )

def to_networkx(graph: Graph, node_feature_key: str = "features", create_using: Optional[Type[nx.Graph]] = None) -> nx.Graph:
    """将稀疏 JAX Graph 对象转换回 NetworkX Graph。"""
    g = create_using() if create_using else nx.Graph()
    g.add_nodes_from(range(graph.n_nodes))
    
    weights = graph.edge_weights if graph.edge_weights is not None else [1.0] * graph.n_edges
    edgelist = zip(graph.senders.tolist(), graph.receivers.tolist(), weights.tolist())
    g.add_weighted_edges_from(edgelist)

    if graph.node_features is not None:
        if graph.node_features.shape[1] == 1:
            features_dict = {i: val.item() for i, val in enumerate(graph.node_features)}
        else:
            features_dict = {i: val.tolist() for i, val in enumerate(graph.node_features)}
        nx.set_node_attributes(g, name=node_feature_key, values=features_dict)

    return g

def to_json(graph: Graph, file_path: str):
    """将稀疏 JAX Graph 对象保存为 JSON 文件 (node-link format)。"""
    nodes_list = []
    for i in range(graph.n_nodes):
        node_data = {"id": i}
        if graph.node_features is not None:
            node_data["features"] = graph.node_features[i].tolist()
        nodes_list.append(node_data)

    links_list = []
    weights = graph.edge_weights if graph.edge_weights is not None else np.ones(graph.n_edges)
    for i in range(graph.n_edges):
        link_data = {
            "source": int(graph.senders[i]),
            "target": int(graph.receivers[i]),
            "weight": float(weights[i])
        }
        links_list.append(link_data)

    with open(file_path, 'w') as f:
        json.dump({"nodes": nodes_list, "links": links_list}, f, indent=2)

def to_csv(graph: Graph, file_path: str):
    """将图的边列表保存到 CSV 文件。"""
    header = ['source', 'target', 'weight']
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # 修正: 先将 JAX 数组转换为 NumPy 数组
        senders_np = np.asarray(graph.senders)
        receivers_np = np.asarray(graph.receivers)

        # 检查是否有权重
        if graph.edge_weights is not None:
            weights_np = np.asarray(graph.edge_weights)
            rows = np.stack([senders_np, receivers_np, weights_np], axis=1)
        else:
            # 如果没有权重，只写入源和目标
            header = ['source', 'target']
            rows = np.stack([senders_np, receivers_np], axis=1)
        
        writer.writerows(rows)