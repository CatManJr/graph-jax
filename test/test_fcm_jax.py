"""
Test JAX FCM implementation against reference.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
# Force JAX to use CPU to avoid GPU issues
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from graphjax.graph_jax.algorithms.cluster.fcm_jax import (
    fuzzy_c_means_jax, compute_clustering_metrics
)
from reference_fcm import FuzzyCMeans


def test_jax_fcm_vs_reference():
    """Compare JAX FCM with reference implementation."""
    print("🧪 Testing JAX FCM vs Reference Implementation")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=200, centers=5, n_features=4, 
                          random_state=42, cluster_std=1.5)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_jax = jnp.array(X_scaled)
    
    print(f"Data shape: {X_scaled.shape}")
    print(f"True clusters: {len(np.unique(y_true))}")
    
    # Test parameters
    n_clusters = 5
    m = 2.0
    max_iter = 300
    tol = 1e-4
    random_state = 42
    
    print(f"\n--- Reference FCM Implementation ---")
    start_time = time.time()
    fcm_ref = FuzzyCMeans(n_clusters=n_clusters, m=m, max_iter=max_iter, 
                         tol=tol, random_state=random_state)
    labels_ref = fcm_ref.fit_predict(X_scaled)
    time_ref = time.time() - start_time
    
    ari_ref = adjusted_rand_score(y_true, labels_ref)
    
    print(f"Time: {time_ref:.3f}s")
    print(f"Iterations: {fcm_ref.n_iter_}")
    print(f"Final objective: {fcm_ref.loss_history_[-1]:.6f}")
    print(f"ARI with ground truth: {ari_ref:.4f}")
    
    # Analyze reference membership
    u_ref = fcm_ref.u_
    entropy_ref = -np.mean(np.sum(u_ref * np.log(u_ref + 1e-10), axis=1))
    max_membership_ref = np.mean(np.max(u_ref, axis=1))
    
    print(f"Membership entropy: {entropy_ref:.4f}")
    print(f"Average max membership: {max_membership_ref:.4f}")
    
    print(f"\n--- JAX FCM Implementation ---")
    start_time = time.time()
    result_jax = fuzzy_c_means_jax(
        X_jax, n_clusters=n_clusters, m=m, max_iter=max_iter, 
        tol=tol, init='k-means++', random_state=random_state
    )
    time_jax = time.time() - start_time
    
    labels_jax = np.array(result_jax['labels'])
    ari_jax = adjusted_rand_score(y_true, labels_jax)
    
    print(f"Time: {time_jax:.3f}s")
    print(f"Iterations: {result_jax['iterations']}")
    print(f"Final objective: {result_jax['final_loss']:.6f}")
    print(f"ARI with ground truth: {ari_jax:.4f}")
    print(f"Converged: {result_jax['converged']}")
    
    # Analyze JAX membership
    metrics_jax = compute_clustering_metrics(result_jax['membership'])
    print(f"Membership entropy: {metrics_jax['entropy']:.4f}")
    print(f"Average max membership: {metrics_jax['max_membership']:.4f}")
    print(f"Partition coefficient: {metrics_jax['partition_coefficient']:.4f}")
    print(f"Balance: {metrics_jax['balance']:.4f}")
    
    print(f"\n--- Comparison ---")
    print(f"ARI difference: {abs(ari_ref - ari_jax):.6f}")
    print(f"Objective difference: {abs(fcm_ref.loss_history_[-1] - result_jax['final_loss']):.6f}")
    print(f"Iteration difference: {abs(fcm_ref.n_iter_ - result_jax['iterations'])}")
    print(f"Speedup: {time_ref / time_jax:.2f}x")
    
    # Test different initialization methods
    print(f"\n--- Testing Random Initialization ---")
    result_random = fuzzy_c_means_jax(
        X_jax, n_clusters=n_clusters, m=m, max_iter=max_iter, 
        tol=tol, init='random', random_state=random_state
    )
    
    labels_random = np.array(result_random['labels'])
    ari_random = adjusted_rand_score(y_true, labels_random)
    
    print(f"Random init ARI: {ari_random:.4f}")
    print(f"K-means++ init ARI: {ari_jax:.4f}")
    print(f"Initialization improvement: {ari_jax - ari_random:.4f}")
    
    return result_jax, fcm_ref


if __name__ == "__main__":
    test_jax_fcm_vs_reference()
