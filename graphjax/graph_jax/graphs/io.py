import jax.numpy as jnp
import networkx as nx
import json
import pandas as pd
from typing import Dict, List, Optional, Type, Union
import warnings
import numpy as np
import csv
import re

from .graph import Graph

def from_networkx(g: nx.Graph, node_feature_key: Optional[str] = None) -> Graph:
    """Create Graph from NetworkX graph object."""
    if not isinstance(g, (nx.Graph, nx.DiGraph)):
        raise TypeError(f"Only nx.Graph and nx.DiGraph are supported, but got {type(g)}")

    # Optimization 1: Use more efficient node mapping creation
    all_nodes = list(g.nodes())
    n_nodes = len(all_nodes)
    
    # Optimization 2: Use dictionary comprehension to create mapping, avoid sorting
    node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
    index_to_node = all_nodes
    
    # Optimization 3: Pre-allocate list size, avoid dynamic expansion
    if g.is_directed():
        edges = list(g.edges(data=True))
        n_edges = len(edges)
        senders = [0] * n_edges
        receivers = [0] * n_edges
        weights_list = [1.0] * n_edges
        
        # Optimization 4: Use enumerate to avoid repeated lookups
        for i, (u, v, d) in enumerate(edges):
            senders[i] = node_to_index[u]
            receivers[i] = node_to_index[v]
            weights_list[i] = d.get('weight', 1.0)
    else:
        # Optimization 5: For undirected graphs, pre-allocate double size
        edges = list(g.edges(data=True))
        n_edges = len(edges) * 2
        senders = [0] * n_edges
        receivers = [0] * n_edges
        weights_list = [1.0] * n_edges
        
        for i, (u, v, data) in enumerate(edges):
            u_idx = node_to_index[u]
            v_idx = node_to_index[v]
            weight = data.get('weight', 1.0)
            
            # Add bidirectional edges
            senders[i*2] = u_idx
            receivers[i*2] = v_idx
            weights_list[i*2] = weight
            
            senders[i*2 + 1] = v_idx
            receivers[i*2 + 1] = u_idx
            weights_list[i*2 + 1] = weight

    # Optimization 6: Batch convert to JAX arrays
    senders = jnp.array(senders, dtype=jnp.int32)
    receivers = jnp.array(receivers, dtype=jnp.int32)

    # Optimization 7: More efficient weight processing
    weights = None
    if any(w != 1.0 for w in weights_list):
        weights = jnp.array(weights_list, dtype=jnp.float32)

    # Optimization 8: More efficient node feature extraction
    node_features = None
    if node_feature_key and g.nodes:
        try:
            # Optimization 9: Direct access to node data, avoid repeated lookups
            nodes_data = dict(g.nodes(data=True))
            if node_feature_key in next(iter(nodes_data.values())):
                node_features_list = [nodes_data[node].get(node_feature_key, 0) for node in all_nodes]
                
                # Optimization 10: More efficient string encoding
                if node_features_list and isinstance(node_features_list[0], str):
                    unique_labels = list(dict.fromkeys(node_features_list))  # Maintain order
                    label_map = {label: i for i, label in enumerate(unique_labels)}
                    encoded_features = [label_map[label] for label in node_features_list]
                    node_features = jnp.array(encoded_features)
                else:
                    node_features = jnp.array(node_features_list)

                if node_features.ndim == 1:
                    node_features = node_features[:, None]
        except (KeyError, TypeError) as e:
            warnings.warn(f"Warning: Error extracting or converting features: {e}. Node features not extracted.")

    return Graph(
        senders=senders,
        receivers=receivers,
        edge_weights=weights,
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_features=node_features,
        _node_to_index=node_to_index,
        _index_to_node=index_to_node
    )

def from_json(file_path: str) -> Graph:
    """Load graph from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    n_nodes = len(data['nodes'])
    
    # Extract edges
    senders = jnp.array([link['source'] for link in data['links']], dtype=jnp.int32)
    receivers = jnp.array([link['target'] for link in data['links']], dtype=jnp.int32)
    n_edges = len(senders) # <--- Calculate number of edges
    
    # Extract weights (if they exist)
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
            warnings.warn("Warning: Not all nodes have 'features' field or format is incorrect. Node features not extracted.")

    return Graph(
        senders=senders,
        receivers=receivers,
        edge_weights=weights,
        n_nodes=n_nodes,
        n_edges=n_edges, # <--- Pass n_edges
        node_features=node_features
    )

def from_csv(file_path: str) -> Graph:
    """Load graph from CSV file."""
    try:
        # Try to read, allowing uneven rows
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            edges_list = [row for row in reader if row]
        
        if not edges_list:
            # Handle empty file case
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
        print(f"Error reading CSV file: {e}")
        return None

    senders = jnp.array(edges[:, 0], dtype=jnp.int32)
    receivers = jnp.array(edges[:, 1], dtype=jnp.int32)
    n_edges = len(senders) # <--- Calculate number of edges
    
    # Assume third column is weight (if it exists)
    if edges.shape[1] > 2:
        weights = jnp.array(edges[:, 2], dtype=jnp.float32)
    else:
        weights = jnp.ones(n_edges, dtype=jnp.float32)

    # Infer number of nodes from edges (handle isolated nodes that may exist in the graph)
    if n_edges > 0:
        n_nodes = int(jnp.maximum(jnp.max(senders), jnp.max(receivers))) + 1
    else:
        n_nodes = 0
    
    # CSV format usually doesn't contain node features, so set to None
    node_features = None
    
    return Graph(
        senders=senders,
        receivers=receivers,
        edge_weights=weights,
        n_nodes=n_nodes,
        n_edges=n_edges, # <--- Pass n_edges
        node_features=node_features
    )

def to_networkx(graph: Graph, node_feature_key: str = "features", create_using: Optional[Type[nx.Graph]] = None) -> nx.Graph:
    """Convert sparse JAX Graph object back to NetworkX Graph."""
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
    """Save sparse JAX Graph object as JSON file (node-link format)."""
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
    """Save graph edge list to CSV file."""
    header = ['source', 'target', 'weight']
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Fix: Convert JAX arrays to NumPy arrays first
        senders_np = np.asarray(graph.senders)
        receivers_np = np.asarray(graph.receivers)

        # Check if weights exist
        if graph.edge_weights is not None:
            weights_np = np.asarray(graph.edge_weights)
            rows = np.stack([senders_np, receivers_np, weights_np], axis=1)
        else:
            # If no weights, only write source and target
            header = ['source', 'target']
            rows = np.stack([senders_np, receivers_np], axis=1)
        
        writer.writerows(rows)