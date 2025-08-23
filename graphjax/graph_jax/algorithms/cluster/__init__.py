"""
Clustering algorithms implemented in JAX.

This module provides clustering algorithms that are well-suited for JAX:
- K-means clustering: Vectorized operations, fixed iterations, differentiable
- Spectral clustering: Linear algebra intensive, parallelizable

Note: These implementations are experimental and may have slight differences
compared to scikit-learn due to JAX's functional programming constraints.
"""

import warnings
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

# Import clustering algorithms
from .kmeans import kmeans_clustering
from .spectral import spectral_clustering
from .soft_clustering import (
    soft_clustering,
    graph_aware_soft_clustering,
    compute_cluster_quality,
    SoftClusteringState,
    SoftClusteringConfig
)
from .utils import (
    compute_cluster_metrics,
    normalize_path_lengths,
    silhouette_score_jax,
    path_similarity_matrix
)

# Module-level warning
warnings.warn(
    "Graph-JAX clustering algorithms are experimental. "
    "Results may differ from scikit-learn due to JAX implementation constraints. "
    "For production use, consider using scikit-learn for maximum compatibility.",
    UserWarning,
    stacklevel=2
)

# Suppress the warning after first import
warnings.simplefilter("ignore", UserWarning)

__all__ = [
    # Core clustering algorithms
    "kmeans_clustering",
    "spectral_clustering",
    "soft_clustering",
    "graph_aware_soft_clustering",
    
    # Soft clustering components
    "compute_cluster_quality",
    "SoftClusteringState",
    "SoftClusteringConfig",
    
    # Utility functions
    "compute_cluster_metrics",
    "normalize_path_lengths", 
    "silhouette_score_jax",
    "path_similarity_matrix",
]

def get_available_algorithms() -> dict:
    """
    Get information about available clustering algorithms.
    
    Returns:
        dict: Dictionary with algorithm names and their descriptions
    """
    return {
        "kmeans": {
            "description": "K-means clustering with JAX optimization",
            "suitable_for": "Vectorized data, fixed number of clusters",
            "jax_compatibility": "High - vectorized operations, differentiable",
            "ari_with_sklearn": "~0.8-0.9"
        },
        "spectral": {
            "description": "Spectral clustering using eigenvalue decomposition",
            "suitable_for": "Graph data, non-linear cluster boundaries",
            "jax_compatibility": "High - linear algebra intensive",
            "ari_with_sklearn": "~0.4-0.6 (needs optimization)"
        },
        "soft_clustering": {
            "description": "Soft clustering with gradient-based optimization",
            "suitable_for": "Uncertain assignments, differentiable pipelines",
            "jax_compatibility": "Very High - leverages autodiff and PyTree",
            "unique_features": "Membership probabilities, graph-aware loss"
        },
        "graph_aware_soft_clustering": {
            "description": "Graph-aware soft clustering with network structure",
            "suitable_for": "Network data with community structure",
            "jax_compatibility": "Very High - leverages autodiff and PyTree",
            "unique_features": "Graph smoothness, modularity optimization"
        }
    }
