#!/usr/bin/env python3
"""
Direct graph clustering test without path-based similarity matrices.

Uses adjacency matrix and node features directly for clustering.
"""

import time
import numpy as np
import jax.numpy as jnp
import networkx as nx
from typing import Dict, Tuple

from graph_jax.utils import set_backend
set_backend('cpu')

# Import sklearn for comparison
try:
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Error: scikit-learn is required for comparison")
    exit(1)

import graph_jax as gj
from graph_jax.algorithms.cluster import (
    spectral_clustering,
    kmeans_clustering
)

def create_test_network(n_nodes: int = 200, n_communities: int = 5, 
                       p_intra: float = 0.3, p_inter: float = 0.02,
                       random_seed: int = 42) -> Tuple[gj.graphs.Graph, np.ndarray]:
    """
    Create a test network with community structure.
    """
    print(f"Creating {n_nodes}-node network with {n_communities} communities...")
    
    np.random.seed(random_seed)
    
    # Assign nodes to communities
    community_sizes = np.random.multinomial(n_nodes - n_communities, 
                                           np.ones(n_communities) / n_communities) + 1
    true_labels = np.repeat(range(n_communities), community_sizes)
    np.random.shuffle(true_labels)
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    edges_added = 0
    
    # Add intra-community edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if true_labels[i] == true_labels[j]:
                if np.random.random() < p_intra:
                    G.add_edge(i, j, weight=np.random.uniform(0.8, 1.0))
                    edges_added += 1
            else:
                if np.random.random() < p_inter:
                    G.add_edge(i, j, weight=np.random.uniform(0.1, 0.3))
                    edges_added += 1
    
    print(f"Generated graph: {n_nodes} nodes, {edges_added} edges")
    print(f"Community sizes: {np.bincount(true_labels)}")
    
    # Convert to Graph-JAX
    graph = gj.graphs.from_networkx(G)
    
    return graph, true_labels

def extract_node_features(graph: gj.graphs.Graph) -> jnp.ndarray:
    """
    Extract node features from the graph for clustering.
    
    Features include:
    1. Degree centrality
    2. Clustering coefficient (local)
    3. Betweenness centrality (approximate)
    4. Eigenvector centrality (approximate)
    """
    n_nodes = graph.n_nodes
    
    # Convert to NetworkX for feature extraction
    nx_graph = nx.Graph()
    for i in range(graph.n_edges):
        sender = int(graph.senders[i])
        receiver = int(graph.receivers[i])
        weight = float(graph.edge_weights[i]) if graph.edge_weights is not None else 1.0
        nx_graph.add_edge(sender, receiver, weight=weight)
    
    # Extract features
    features = []
    
    # 1. Degree centrality
    degrees = dict(nx.degree(nx_graph))
    degree_features = [degrees.get(i, 0) for i in range(n_nodes)]
    features.append(degree_features)
    
    # 2. Clustering coefficient
    clustering = nx.clustering(nx_graph)
    clustering_features = [clustering.get(i, 0) for i in range(n_nodes)]
    features.append(clustering_features)
    
    # 3. Betweenness centrality (sample-based for efficiency)
    if n_nodes <= 100:
        betweenness = nx.betweenness_centrality(nx_graph)
        betweenness_features = [betweenness.get(i, 0) for i in range(n_nodes)]
    else:
        # Use degree as approximation for large graphs
        betweenness_features = degree_features
    features.append(betweenness_features)
    
    # 4. Eigenvector centrality (approximate)
    try:
        eigenvector = nx.eigenvector_centrality(nx_graph, max_iter=1000)
        eigenvector_features = [eigenvector.get(i, 0) for i in range(n_nodes)]
    except:
        # Fallback to degree centrality
        eigenvector_features = degree_features
    features.append(eigenvector_features)
    
    # Combine features
    feature_matrix = jnp.array(features).T  # Shape: (n_nodes, n_features)
    
    # Normalize features
    feature_matrix = (feature_matrix - jnp.mean(feature_matrix, axis=0)) / (jnp.std(feature_matrix, axis=0) + 1e-8)
    
    return feature_matrix

def create_adjacency_similarity(graph: gj.graphs.Graph) -> jnp.ndarray:
    """
    Create similarity matrix based on adjacency matrix.
    """
    # Get adjacency matrix
    adj_matrix = graph.to_adjacency_matrix()
    
    # Method 1: Direct adjacency matrix (binary)
    # This is what scikit-learn uses by default
    similarity = adj_matrix.astype(float)
    
    # Method 2: Add self-loops for numerical stability
    similarity = similarity + jnp.eye(similarity.shape[0])
    
    # Method 3: Normalize by degree (optional)
    # degrees = jnp.sum(adj_matrix, axis=1, keepdims=True)
    # similarity = similarity / (degrees + 1e-8)
    
    return similarity

def test_direct_clustering(graph: gj.graphs.Graph, true_labels: np.ndarray, 
                          n_clusters: int = 5) -> Dict:
    """
    Test clustering using direct graph features and adjacency.
    """
    print(f"\n{'='*60}")
    print(f"DIRECT GRAPH CLUSTERING TEST")
    print(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    print(f"Target clusters: {n_clusters}")
    print(f"{'='*60}")
    
    results = {}
    
    # Method 1: Node Features + K-means
    print("\n--- Method 1: Node Features + K-means ---")
    
    # Extract node features
    start_time = time.time()
    node_features = extract_node_features(graph)
    feature_time = time.time() - start_time
    
    print(f"Feature extraction: {feature_time:.2f}s")
    print(f"Feature matrix shape: {node_features.shape}")
    
    # Graph-JAX K-means on features
    start_time = time.time()
    gj_kmeans_labels = kmeans_clustering(node_features, n_clusters, random_state=42)
    gj_kmeans_time = time.time() - start_time
    
    # SciKit-Learn K-means on features
    start_time = time.time()
    sk_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    sk_kmeans_labels = sk_kmeans.fit_predict(np.array(node_features))
    sk_kmeans_time = time.time() - start_time
    
    # Compare results
    kmeans_ari = adjusted_rand_score(gj_kmeans_labels, sk_kmeans_labels)
    kmeans_true_ari = adjusted_rand_score(true_labels, gj_kmeans_labels)
    
    print(f"Graph-JAX time: {gj_kmeans_time:.2f}s")
    print(f"SciKit-Learn time: {sk_kmeans_time:.2f}s")
    print(f"ARI with SciKit-Learn: {kmeans_ari:.4f}")
    print(f"ARI with ground truth: {kmeans_true_ari:.4f}")
    
    if kmeans_ari > 0.9:
        print("✅ Excellent agreement with SciKit-Learn")
    elif kmeans_ari > 0.7:
        print("✅ Good agreement with SciKit-Learn")
    elif kmeans_ari > 0.5:
        print("⚠️  Moderate agreement with SciKit-Learn")
    else:
        print("❌ Poor agreement with SciKit-Learn")
    
    results['kmeans_features'] = {
        'gj_time': gj_kmeans_time,
        'sk_time': sk_kmeans_time,
        'ari_sklearn': kmeans_ari,
        'ari_true': kmeans_true_ari,
        'gj_labels': gj_kmeans_labels,
        'sk_labels': sk_kmeans_labels
    }
    
    # Method 2: Adjacency-based Spectral Clustering
    print("\n--- Method 2: Adjacency-based Spectral Clustering ---")
    
    # Create adjacency similarity matrix
    start_time = time.time()
    adj_similarity = create_adjacency_similarity(graph)
    similarity_time = time.time() - start_time
    
    print(f"Similarity computation: {similarity_time:.2f}s")
    print(f"Similarity matrix shape: {adj_similarity.shape}")
    
    # Graph-JAX Spectral Clustering
    start_time = time.time()
    gj_spectral_labels, eigenvalues = spectral_clustering(adj_similarity, n_clusters, random_state=42)
    gj_spectral_time = time.time() - start_time
    
    # SciKit-Learn Spectral Clustering
    start_time = time.time()
    sk_spectral = SpectralClustering(
        n_clusters=n_clusters,
        random_state=42,
        assign_labels='kmeans',
        affinity='precomputed'
    )
    sk_spectral_labels = sk_spectral.fit_predict(np.array(adj_similarity))
    sk_spectral_time = time.time() - start_time
    
    # Compare results
    spectral_ari = adjusted_rand_score(gj_spectral_labels, sk_spectral_labels)
    spectral_true_ari = adjusted_rand_score(true_labels, gj_spectral_labels)
    
    print(f"Graph-JAX time: {gj_spectral_time:.2f}s")
    print(f"SciKit-Learn time: {sk_spectral_time:.2f}s")
    print(f"ARI with SciKit-Learn: {spectral_ari:.4f}")
    print(f"ARI with ground truth: {spectral_true_ari:.4f}")
    
    if spectral_ari > 0.9:
        print("✅ Excellent agreement with SciKit-Learn")
    elif spectral_ari > 0.7:
        print("✅ Good agreement with SciKit-Learn")
    elif spectral_ari > 0.5:
        print("⚠️  Moderate agreement with SciKit-Learn")
    else:
        print("❌ Poor agreement with SciKit-Learn")
    
    results['spectral_adjacency'] = {
        'gj_time': gj_spectral_time,
        'sk_time': sk_spectral_time,
        'ari_sklearn': spectral_ari,
        'ari_true': spectral_true_ari,
        'gj_labels': gj_spectral_labels,
        'sk_labels': sk_spectral_labels,
        'eigenvalues': eigenvalues
    }
    
    # Clustering quality analysis
    print("\n--- Clustering Quality Analysis ---")
    
    def analyze_clustering_quality(labels, name):
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        min_size = min(counts)
        max_size = max(counts)
        balance_ratio = min_size / max_size if max_size > 0 else 0
        
        print(f"{name}:")
        print(f"  Clusters found: {len(unique_labels)}")
        print(f"  Cluster sizes: {cluster_sizes}")
        print(f"  Balance ratio: {balance_ratio:.3f}")
        
        if balance_ratio > 0.5:
            print(f"  ✅ Well-balanced clustering")
        elif balance_ratio > 0.2:
            print(f"  ⚠️  Moderately balanced clustering")
        else:
            print(f"  ❌ Unbalanced clustering")
    
    analyze_clustering_quality(gj_kmeans_labels, "K-means on Features")
    analyze_clustering_quality(gj_spectral_labels, "Spectral on Adjacency")
    analyze_clustering_quality(true_labels, "Ground Truth")
    
    return results

def main():
    """Main test function."""
    print("🔬 Direct Graph Clustering Test")
    print("Testing clustering without path-based similarity matrices")
    print("Using node features and adjacency matrix directly")
    
    # Create test network
    graph, true_labels = create_test_network(
        n_nodes=200,
        n_communities=5,
        p_intra=0.3,
        p_inter=0.02,
        random_seed=42
    )
    
    # Run clustering tests
    results = test_direct_clustering(graph, true_labels, n_clusters=5)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    kmeans_results = results['kmeans_features']
    spectral_results = results['spectral_adjacency']
    
    print(f"Graph size: {graph.n_nodes} nodes, {graph.n_edges} edges")
    print()
    
    print("K-means on Node Features:")
    print(f"  ARI with SciKit-Learn: {kmeans_results['ari_sklearn']:.4f}")
    print(f"  ARI with ground truth: {kmeans_results['ari_true']:.4f}")
    print(f"  Performance: {kmeans_results['gj_time']:.2f}s vs {kmeans_results['sk_time']:.2f}s")
    
    print("\nSpectral Clustering on Adjacency:")
    print(f"  ARI with SciKit-Learn: {spectral_results['ari_sklearn']:.4f}")
    print(f"  ARI with ground truth: {spectral_results['ari_true']:.4f}")
    print(f"  Performance: {spectral_results['gj_time']:.2f}s vs {spectral_results['sk_time']:.2f}s")
    
    # Overall assessment
    print("\n🎯 Overall Assessment:")
    
    if (kmeans_results['ari_sklearn'] > 0.8 and spectral_results['ari_sklearn'] > 0.8):
        print("✅ Both methods show excellent agreement with SciKit-Learn")
    elif (kmeans_results['ari_sklearn'] > 0.6 and spectral_results['ari_sklearn'] > 0.6):
        print("✅ Both methods show good agreement with SciKit-Learn")
    else:
        print("⚠️  Some methods need further optimization")
    
    # Performance comparison
    avg_gj_time = (kmeans_results['gj_time'] + spectral_results['gj_time']) / 2
    avg_sk_time = (kmeans_results['sk_time'] + spectral_results['sk_time']) / 2
    performance_ratio = avg_gj_time / avg_sk_time
    
    if performance_ratio < 2.0:
        print(f"🚀 Good performance: Graph-JAX is {performance_ratio:.1f}x slower than SciKit-Learn")
    elif performance_ratio < 5.0:
        print(f"⚡ Acceptable performance: Graph-JAX is {performance_ratio:.1f}x slower than SciKit-Learn")
    else:
        print(f"🐌 Performance needs optimization: Graph-JAX is {performance_ratio:.1f}x slower than SciKit-Learn")
    
    print("\n🎉 Test completed successfully!")

if __name__ == "__main__":
    main()
