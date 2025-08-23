"""
Test script for soft clustering with gradient optimization.

This script demonstrates the JAX-native soft clustering algorithms that leverage
automatic differentiation and PyTree capabilities.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import graph_jax as gj

# Force CPU backend for consistency
gj.utils.set_backend('cpu')

# Import our clustering algorithms
from graph_jax.algorithms.cluster import (
    soft_clustering,
    graph_aware_soft_clustering,
    compute_cluster_quality,
    kmeans_clustering
)


def create_test_network(n_nodes: int = 300, n_communities: int = 4, 
                       feature_dim: int = 6) -> Tuple[gj.graphs.Graph, np.ndarray, np.ndarray]:
    """
    Create a test network with node features suitable for K-means and soft clustering.
    
    Args:
        n_nodes: Number of nodes
        n_communities: Number of communities
        feature_dim: Number of node features
        
    Returns:
        Tuple of (graph, true_labels, node_features)
    """
    print(f"Creating {n_nodes}-node network with {n_communities} communities and {feature_dim} features...")
    
    # Create synthetic node features with clear cluster structure
    np.random.seed(42)
    
    # Generate cluster centers for features
    cluster_centers = np.random.randn(n_communities, feature_dim) * 2.0
    
    # Assign nodes to communities
    nodes_per_community = n_nodes // n_communities
    true_labels = np.zeros(n_nodes, dtype=int)
    node_features = np.zeros((n_nodes, feature_dim))
    
    for i in range(n_communities):
        start_idx = i * nodes_per_community
        end_idx = start_idx + nodes_per_community if i < n_communities - 1 else n_nodes
        true_labels[start_idx:end_idx] = i
        
        # Generate features around cluster center with some noise
        cluster_size = end_idx - start_idx
        noise = np.random.randn(cluster_size, feature_dim) * 0.5
        node_features[start_idx:end_idx] = cluster_centers[i] + noise
    
    # Create graph structure based on feature similarity
    # Connect nodes that are close in feature space
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Add edges based on feature similarity
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Compute feature similarity
            feature_dist = np.linalg.norm(node_features[i] - node_features[j])
            similarity = np.exp(-feature_dist / 2.0)  # Gaussian similarity
            
            # Add edge with probability based on similarity
            if np.random.random() < similarity * 0.3:  # Scale factor for sparsity
                G.add_edge(i, j, weight=similarity)
    
    # Ensure graph is connected
    if not nx.is_connected(G):
        # Add minimum spanning tree to ensure connectivity
        mst = nx.minimum_spanning_tree(G)
        for edge in mst.edges():
            G.add_edge(*edge, weight=1.0)
    
    # Convert to Graph-JAX format
    graph = gj.graphs.from_networkx(G)
    
    print(f"Generated graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    print(f"Community sizes: {[np.sum(true_labels == i) for i in range(n_communities)]}")
    print(f"Feature matrix shape: {node_features.shape}")
    
    return graph, true_labels, node_features





def test_soft_clustering_methods(graph: gj.graphs.Graph, true_labels: np.ndarray, 
                                node_features: np.ndarray, n_clusters: int = 4) -> Dict:
    """
    Test different soft clustering methods.
    
    Args:
        graph: Graph object
        true_labels: True community labels
        node_features: Node feature matrix
        n_clusters: Number of clusters
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"SOFT CLUSTERING WITH GRADIENT OPTIMIZATION")
    print(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    print(f"Target clusters: {n_clusters}")
    print(f"{'='*60}")
    
    results = {}
    
    # Convert features to JAX array and normalize
    node_features_jax = jnp.array(node_features)
    node_features_jax = (node_features_jax - jnp.mean(node_features_jax, axis=0)) / (jnp.std(node_features_jax, axis=0) + 1e-8)
    
    # Get adjacency matrix
    adjacency_matrix = graph.to_adjacency_matrix()
    
    print(f"\nFeature matrix shape: {node_features_jax.shape}")
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    
    # Method 1: Standard Soft Clustering
    print("\n--- Method 1: Standard Soft Clustering ---")
    
    start_time = time.time()
    soft_results = soft_clustering(
        data=node_features_jax,
        n_clusters=n_clusters,
        temperature=2.0,  # FCM fuzziness parameter
        max_iter=300,
        learning_rate=0.01,
        convergence_tol=1e-4,
        random_state=42
    )
    soft_time = time.time() - start_time
    
    soft_labels = soft_results['labels']
    soft_membership = soft_results['membership']
    soft_loss = soft_results['final_loss']
    soft_iterations = soft_results['iterations']
    
    # Compute quality metrics
    soft_quality = compute_cluster_quality(soft_membership, adjacency_matrix)
    soft_ari = adjusted_rand_score(true_labels, soft_labels)
    
    print(f"Time: {soft_time:.2f}s")
    print(f"Iterations: {soft_iterations}")
    print(f"Final loss: {soft_loss:.6f}")
    print(f"ARI with ground truth: {soft_ari:.4f}")
    print(f"Entropy: {soft_quality['entropy']:.4f}")
    print(f"Sparsity: {soft_quality['sparsity']:.4f}")
    print(f"Balance: {soft_quality['balance']:.4f}")
    if 'modularity' in soft_quality:
        print(f"Modularity: {soft_quality['modularity']:.4f}")
    
    results['soft_clustering'] = {
        'labels': soft_labels,
        'membership': soft_membership,
        'time': soft_time,
        'iterations': soft_iterations,
        'loss': soft_loss,
        'ari': soft_ari,
        'quality': soft_quality
    }
    
    # Method 2: Graph-Aware Soft Clustering
    print("\n--- Method 2: Graph-Aware Soft Clustering ---")
    
    start_time = time.time()
    graph_soft_results = graph_aware_soft_clustering(
        data=node_features_jax,
        adjacency=adjacency_matrix,
        n_clusters=n_clusters,
        temperature=2.0,  # FCM fuzziness parameter
        max_iter=300,
        learning_rate=0.01,
        convergence_tol=1e-4,
        random_state=42
    )
    graph_soft_time = time.time() - start_time
    
    graph_soft_labels = graph_soft_results['labels']
    graph_soft_membership = graph_soft_results['membership']
    graph_soft_loss = graph_soft_results['final_loss']
    graph_soft_iterations = graph_soft_results['iterations']
    
    # Compute quality metrics
    graph_soft_quality = compute_cluster_quality(graph_soft_membership, adjacency_matrix)
    graph_soft_ari = adjusted_rand_score(true_labels, graph_soft_labels)
    
    print(f"Time: {graph_soft_time:.2f}s")
    print(f"Iterations: {graph_soft_iterations}")
    print(f"Final loss: {graph_soft_loss:.6f}")
    print(f"ARI with ground truth: {graph_soft_ari:.4f}")
    print(f"Entropy: {graph_soft_quality['entropy']:.4f}")
    print(f"Sparsity: {graph_soft_quality['sparsity']:.4f}")
    print(f"Balance: {graph_soft_quality['balance']:.4f}")
    if 'modularity' in graph_soft_quality:
        print(f"Modularity: {graph_soft_quality['modularity']:.4f}")
    
    results['graph_aware_soft_clustering'] = {
        'labels': graph_soft_labels,
        'membership': graph_soft_membership,
        'time': graph_soft_time,
        'iterations': graph_soft_iterations,
        'loss': graph_soft_loss,
        'ari': graph_soft_ari,
        'quality': graph_soft_quality
    }
    
    # Method 3: Traditional K-means (for comparison)
    print("\n--- Method 3: Traditional K-means (Comparison) ---")
    
    start_time = time.time()
    kmeans_labels = kmeans_clustering(node_features_jax, n_clusters, random_state=42)
    kmeans_time = time.time() - start_time
    
    kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
    
    print(f"Time: {kmeans_time:.2f}s")
    print(f"ARI with ground truth: {kmeans_ari:.4f}")
    
    results['kmeans'] = {
        'labels': kmeans_labels,
        'time': kmeans_time,
        'ari': kmeans_ari
    }
    
    return results


def analyze_results(results: Dict) -> None:
    """
    Analyze and compare clustering results.
    
    Args:
        results: Dictionary with clustering results
    """
    print(f"\n{'='*60}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    # Create comparison table
    methods = ['soft_clustering', 'graph_aware_soft_clustering', 'kmeans']
    method_names = ['Soft Clustering', 'Graph-Aware Soft', 'K-means']
    
    print(f"\n{'Method':<25} {'ARI':<8} {'Time(s)':<8} {'Iterations':<12} {'Loss':<12}")
    print("-" * 70)
    
    for method, name in zip(methods, method_names):
        if method in results:
            result = results[method]
            ari = result['ari']
            time_val = result['time']
            iterations = result.get('iterations', 'N/A')
            loss = result.get('loss', 'N/A')
            
            # Handle different data types for formatting
            if isinstance(iterations, (int, float)):
                iterations_str = f"{iterations}"
            else:
                iterations_str = str(iterations)
            
            if isinstance(loss, (int, float)):
                loss_str = f"{loss:.6f}"
            else:
                loss_str = str(loss)
            
            print(f"{name:<25} {ari:<8.4f} {time_val:<8.2f} {iterations_str:<12} {loss_str:<12}")
    
    # Find best method
    best_method = max(methods, key=lambda m: results[m]['ari'] if m in results else 0)
    best_ari = results[best_method]['ari']
    
    print(f"\n🏆 Best Method: {method_names[methods.index(best_method)]}")
    print(f"🏆 Best ARI: {best_ari:.4f}")
    
    # Analyze soft clustering advantages
    if 'soft_clustering' in results and 'graph_aware_soft_clustering' in results:
        print(f"\n📊 Soft Clustering Analysis:")
        
        soft_entropy = results['soft_clustering']['quality']['entropy']
        graph_entropy = results['graph_aware_soft_clustering']['quality']['entropy']
        
        print(f"   Standard Soft Entropy: {soft_entropy:.4f}")
        print(f"   Graph-Aware Soft Entropy: {graph_entropy:.4f}")
        
        if 'modularity' in results['soft_clustering']['quality']:
            soft_modularity = results['soft_clustering']['quality']['modularity']
            graph_modularity = results['graph_aware_soft_clustering']['quality']['modularity']
            
            print(f"   Standard Soft Modularity: {soft_modularity:.4f}")
            print(f"   Graph-Aware Soft Modularity: {graph_modularity:.4f}")


def visualize_membership(membership: jnp.ndarray, title: str) -> None:
    """
    Visualize membership probabilities.
    
    Args:
        membership: Membership matrix
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(membership, cmap='viridis', aspect='auto')
    plt.colorbar(label='Membership Probability')
    plt.xlabel('Cluster')
    plt.ylabel('Node')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    """Main test function."""
    print("🧪 Soft Clustering with Gradient Optimization Test")
    print("Testing JAX-native soft clustering algorithms")
    
    # Create test network with node features
    graph, true_labels, node_features = create_test_network(n_nodes=300, n_communities=4, feature_dim=6)
    
    # Test soft clustering methods
    results = test_soft_clustering_methods(graph, true_labels, node_features, n_clusters=4)
    
    # Analyze results
    analyze_results(results)
    
    # Optional: Visualize membership probabilities
    if 'graph_aware_soft_clustering' in results:
        print(f"\n📈 Visualizing membership probabilities...")
        membership = results['graph_aware_soft_clustering']['membership']
        visualize_membership(membership, "Graph-Aware Soft Clustering Membership")
    
    print(f"\n🎉 Test completed successfully!")


if __name__ == "__main__":
    main()
