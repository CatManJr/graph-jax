#!/usr/bin/env python3
"""
Demonstration of path normalization and graph clustering algorithms.
This script shows practical usage examples of the implemented algorithms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Any, Tuple

# Import our graph-jax library
import graphjax as gj
from graphjax.algorithms import (
    shortest_paths,
    path_similarity_matrix,
    normalize_path_lengths,
    spectral_clustering,
    kmeans_clustering,
    modularity_optimization,
    hierarchical_clustering,
    compute_cluster_metrics,
    path_based_clustering
)

# Set JAX to use CPU for compatibility
from graphjax.utils.set_backend import set_backend
set_backend('cpu')

# Configure matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_community_graph():
    """Create a synthetic graph with clear community structure."""
    print("Creating synthetic community graph...")
    
    # Create a graph with 3 communities
    g = nx.Graph()
    
    # Community 1: nodes 0-9
    for i in range(10):
        for j in range(i+1, 10):
            g.add_edge(i, j, weight=1.0)
    
    # Community 2: nodes 10-19
    for i in range(10, 20):
        for j in range(i+1, 20):
            g.add_edge(i, j, weight=1.0)
    
    # Community 3: nodes 20-29
    for i in range(20, 30):
        for j in range(i+1, 30):
            g.add_edge(i, j, weight=1.0)
    
    # Add a few inter-community edges (weak connections)
    inter_edges = [(0, 10), (1, 11), (2, 12), (10, 20), (11, 21), (12, 22)]
    for u, v in inter_edges:
        g.add_edge(u, v, weight=0.1)
    
    return gj.graphs.from_networkx(g)

def demonstrate_path_normalization(graph: gj.graphs.Graph):
    """Demonstrate path normalization techniques."""
    print("\n" + "="*60)
    print("PATH NORMALIZATION DEMONSTRATION")
    print("="*60)
    
    # Step 1: Compute shortest paths
    print("\n1. Computing shortest paths...")
    distances = shortest_paths(graph, graph.n_nodes)
    print(f"   Distance matrix shape: {distances.shape}")
    print(f"   Distance range: [{jnp.min(distances):.2f}, {jnp.max(distances):.2f}]")
    
    # Step 2: Show different normalization methods
    print("\n2. Comparing normalization methods:")
    methods = ["min_max", "z_score", "robust"]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original distances
    im1 = axes[0].imshow(distances, cmap='viridis')
    axes[0].set_title('Original Distances')
    axes[0].set_xlabel('Node ID')
    axes[0].set_ylabel('Node ID')
    plt.colorbar(im1, ax=axes[0])
    
    for i, method in enumerate(methods, 1):
        normalized = normalize_path_lengths(distances, method)
        im = axes[i].imshow(normalized, cmap='viridis')
        axes[i].set_title(f'{method.replace("_", " ").title()}')
        axes[i].set_xlabel('Node ID')
        axes[i].set_ylabel('Node ID')
        plt.colorbar(im, ax=axes[i])
        
        print(f"   {method}: range [{jnp.min(normalized):.3f}, {jnp.max(normalized):.3f}]")
    
    plt.tight_layout()
    plt.savefig('path_normalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Visualization saved as 'path_normalization_comparison.png'")
    
    # Step 3: Show similarity matrices with different sigma values
    print("\n3. Computing similarity matrices with different σ values...")
    sigmas = [0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, sigma in enumerate(sigmas):
        similarities = path_similarity_matrix(distances, sigma)
        im = axes[i].imshow(similarities, cmap='plasma')
        axes[i].set_title(f'Similarity (σ={sigma})')
        axes[i].set_xlabel('Node ID')
        axes[i].set_ylabel('Node ID')
        plt.colorbar(im, ax=axes[i])
        
        print(f"   σ={sigma}: range [{jnp.min(similarities):.3f}, {jnp.max(similarities):.3f}]")
    
    plt.tight_layout()
    plt.savefig('similarity_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Visualization saved as 'similarity_matrices.png'")
    
    return distances, similarities

def demonstrate_clustering_algorithms(graph: gj.graphs.Graph, similarities: jnp.ndarray):
    """Demonstrate different clustering algorithms."""
    print("\n" + "="*60)
    print("CLUSTERING ALGORITHMS DEMONSTRATION")
    print("="*60)
    
    n_clusters = 3  # We know we have 3 communities
    
    # Create NetworkX graph for visualization
    nx_graph = nx.Graph()
    for i in range(graph.n_edges):
        nx_graph.add_edge(
            int(graph.senders[i]), 
            int(graph.receivers[i]),
            weight=float(graph.edge_weights[i]) if graph.edge_weights is not None else 1.0
        )
    
    pos = nx.spring_layout(nx_graph, seed=42)
    
    # Test different clustering methods
    clustering_methods = {
        'Spectral Clustering': lambda: spectral_clustering(similarities, n_clusters),
        'Modularity Optimization': lambda: (modularity_optimization(graph), None),
        'Hierarchical Clustering': lambda: (hierarchical_clustering(similarities, "ward"), None)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot original graph
    nx.draw(nx_graph, pos, ax=axes[0], node_color='lightblue', 
            node_size=200, with_labels=True, font_size=8)
    axes[0].set_title('Original Graph')
    
    # Plot clustering results
    for i, (method_name, clustering_func) in enumerate(clustering_methods.items(), 1):
        print(f"\n{method_name}:")
        
        try:
            cluster_labels, eigenvalues = clustering_func()
            if eigenvalues is None:
                cluster_labels = cluster_labels  # For methods that don't return eigenvalues
            
            # Compute metrics
            metrics = compute_cluster_metrics(similarities, cluster_labels)
            print(f"   Silhouette score: {metrics['silhouette']:.4f}")
            print(f"   Modularity: {metrics['modularity']:.4f}")
            print(f"   Number of clusters: {metrics['n_clusters']}")
            
            # Visualize
            unique_labels = jnp.unique(cluster_labels)
            colors = plt.cm.Set3(jnp.arange(len(unique_labels)) / len(unique_labels))
            node_colors = [colors[jnp.where(unique_labels == label)[0][0]] for label in cluster_labels]
            
            nx.draw(nx_graph, pos, ax=axes[i], node_color=node_colors, 
                    node_size=200, with_labels=True, font_size=8)
            
            title = f'{method_name}\nSilhouette: {metrics["silhouette"]:.3f}\nModularity: {metrics["modularity"]:.3f}'
            axes[i].set_title(title, fontsize=10)
            
        except Exception as e:
            print(f"   Error: {e}")
            axes[i].text(0.5, 0.5, f'Error in {method_name}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(method_name)
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n   Visualization saved as 'clustering_comparison.png'")

def demonstrate_path_based_clustering(graph: gj.graphs.Graph):
    """Demonstrate the complete path-based clustering pipeline."""
    print("\n" + "="*60)
    print("PATH-BASED CLUSTERING PIPELINE DEMONSTRATION")
    print("="*60)
    
    n_clusters = 3
    sigmas = [0.5, 1.0, 2.0, 5.0]
    
    # Create NetworkX graph for visualization
    nx_graph = nx.Graph()
    for i in range(graph.n_edges):
        nx_graph.add_edge(
            int(graph.senders[i]), 
            int(graph.receivers[i]),
            weight=float(graph.edge_weights[i]) if graph.edge_weights is not None else 1.0
        )
    
    pos = nx.spring_layout(nx_graph, seed=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot original graph
    nx.draw(nx_graph, pos, ax=axes[0], node_color='lightblue', 
            node_size=200, with_labels=True, font_size=8)
    axes[0].set_title('Original Graph')
    
    # Test path-based clustering with different sigma values
    for i, sigma in enumerate(sigmas, 1):
        print(f"\nPath-based clustering with σ={sigma}:")
        
        try:
            cluster_labels, metrics = path_based_clustering(
                graph, n_clusters, sigma=sigma, normalize_paths=True
            )
            
            print(f"   Silhouette score: {metrics['silhouette']:.4f}")
            print(f"   Modularity: {metrics['modularity']:.4f}")
            print(f"   Number of clusters: {metrics['n_clusters']}")
            
            # Visualize
            unique_labels = jnp.unique(cluster_labels)
            colors = plt.cm.Set3(jnp.arange(len(unique_labels)) / len(unique_labels))
            node_colors = [colors[jnp.where(unique_labels == label)[0][0]] for label in cluster_labels]
            
            nx.draw(nx_graph, pos, ax=axes[i], node_color=node_colors, 
                    node_size=200, with_labels=True, font_size=8)
            
            title = f'Path-Based Clustering (σ={sigma})\nSilhouette: {metrics["silhouette"]:.3f}\nModularity: {metrics["modularity"]:.3f}'
            axes[i].set_title(title, fontsize=10)
            
        except Exception as e:
            print(f"   Error: {e}")
            axes[i].text(0.5, 0.5, f'Error with σ={sigma}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Path-Based Clustering (σ={sigma})')
    
    plt.tight_layout()
    plt.savefig('path_based_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n   Visualization saved as 'path_based_clustering.png'")

def demonstrate_parameter_sensitivity(graph: gj.graphs.Graph):
    """Demonstrate how clustering results change with different parameters."""
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Test different numbers of clusters
    n_clusters_range = [2, 3, 4, 5, 6]
    sigma_range = [0.5, 1.0, 2.0, 5.0]
    
    # Compute base similarity matrix
    distances = shortest_paths(graph, graph.n_nodes)
    similarities = path_similarity_matrix(distances, sigma=1.0)
    
    # Test spectral clustering with different numbers of clusters
    print("\nSpectral clustering with different numbers of clusters:")
    silhouette_scores = []
    
    for n_clusters in n_clusters_range:
        try:
            cluster_labels, _ = spectral_clustering(similarities, n_clusters)
            metrics = compute_cluster_metrics(similarities, cluster_labels)
            silhouette_scores.append(metrics['silhouette'])
            print(f"   {n_clusters} clusters: Silhouette = {metrics['silhouette']:.4f}")
        except Exception as e:
            silhouette_scores.append(0.0)
            print(f"   {n_clusters} clusters: Error - {e}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Silhouette scores vs number of clusters
    ax1.plot(n_clusters_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Clustering Quality vs Number of Clusters')
    ax1.grid(True, alpha=0.3)
    
    # Test different sigma values
    print("\nPath-based clustering with different σ values:")
    sigma_scores = []
    
    for sigma in sigma_range:
        try:
            cluster_labels, metrics = path_based_clustering(graph, 3, sigma=sigma)
            sigma_scores.append(metrics['silhouette'])
            print(f"   σ={sigma}: Silhouette = {metrics['silhouette']:.4f}")
        except Exception as e:
            sigma_scores.append(0.0)
            print(f"   σ={sigma}: Error - {e}")
    
    # Silhouette scores vs sigma
    ax2.plot(sigma_range, sigma_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Sigma (σ)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Clustering Quality vs Sigma Parameter')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n   Visualization saved as 'parameter_sensitivity.png'")

def main():
    """Main demonstration function."""
    print("=== Path Normalization and Graph Clustering Demo ===")
    print("This demo shows practical usage of the implemented algorithms.")
    
    # Create a test graph with clear community structure
    graph = create_community_graph()
    print(f"Created graph with {graph.n_nodes} nodes and {graph.n_edges} edges")
    
    # Demonstrate path normalization
    distances, similarities = demonstrate_path_normalization(graph)
    
    # Demonstrate clustering algorithms
    demonstrate_clustering_algorithms(graph, similarities)
    
    # Demonstrate path-based clustering pipeline
    demonstrate_path_based_clustering(graph)
    
    # Demonstrate parameter sensitivity
    demonstrate_parameter_sensitivity(graph)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED")
    print("="*60)
    print("Generated visualizations:")
    print("  - path_normalization_comparison.png")
    print("  - similarity_matrices.png")
    print("  - clustering_comparison.png")
    print("  - path_based_clustering.png")
    print("  - parameter_sensitivity.png")
    print("\nThese files show the different aspects of the algorithms:")
    print("  1. How path normalization affects distance matrices")
    print("  2. How similarity matrices change with different σ values")
    print("  3. Comparison of different clustering algorithms")
    print("  4. Path-based clustering with different parameters")
    print("  5. Parameter sensitivity analysis")

if __name__ == "__main__":
    main()
