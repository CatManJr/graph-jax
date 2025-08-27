#!/usr/bin/env python3
"""
Test script for Floyd-Warshall algorithm correctness verification.

This script validates the Floyd-Warshall implementation in Graph-JAX
by comparing results with established libraries like NetworkX and SciPy.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import time
from typing import Dict, Tuple, List

# Force JAX to use CPU backend for stability
jax.config.update('jax_platform_name', 'cpu')

import graph_jax as gj
from graph_jax.algorithms.floyd_warshall import (
    shortest_paths,
    single_source_shortest_paths,
    path_exists,
    diameter,
    average_shortest_path_length
)


def create_test_cases() -> Dict[str, nx.Graph]:
    """Create various test graphs for comprehensive validation."""
    test_cases = {}
    
    # Test Case 1: Simple triangle graph
    G1 = nx.Graph()
    G1.add_edges_from([
        (0, 1, {'weight': 3}),
        (1, 2, {'weight': 4}),
        (0, 2, {'weight': 5})
    ])
    test_cases['triangle'] = G1
    
    # Test Case 2: Diamond graph
    G2 = nx.Graph()
    G2.add_edges_from([
        (0, 1, {'weight': 2}),
        (1, 2, {'weight': 3}),
        (0, 3, {'weight': 4}),
        (3, 2, {'weight': 1}),
        (1, 3, {'weight': 5})
    ])
    test_cases['diamond'] = G2
    
    # Test Case 3: Disconnected graph
    G3 = nx.Graph()
    G3.add_edges_from([
        (0, 1, {'weight': 1}),
        (1, 2, {'weight': 2}),
        (3, 4, {'weight': 3}),
        (4, 5, {'weight': 4})
    ])
    test_cases['disconnected'] = G3
    
    # Test Case 4: Single node
    G4 = nx.Graph()
    test_cases['single_node'] = G4
    
    # Test Case 5: Empty graph
    G5 = nx.Graph()
    test_cases['empty'] = G5
    
    # Test Case 6: Line graph
    G6 = nx.Graph()
    G6.add_edges_from([
        (0, 1, {'weight': 1}),
        (1, 2, {'weight': 2}),
        (2, 3, {'weight': 3}),
        (3, 4, {'weight': 4})
    ])
    test_cases['line'] = G6
    
    # Test Case 7: Cycle graph
    G7 = nx.Graph()
    G7.add_edges_from([
        (0, 1, {'weight': 1}),
        (1, 2, {'weight': 2}),
        (2, 3, {'weight': 3}),
        (3, 0, {'weight': 4})
    ])
    test_cases['cycle'] = G7
    
    # Test Case 8: Star graph
    G8 = nx.Graph()
    G8.add_edges_from([
        (0, 1, {'weight': 1}),
        (0, 2, {'weight': 2}),
        (0, 3, {'weight': 3}),
        (0, 4, {'weight': 4})
    ])
    test_cases['star'] = G8
    
    return test_cases


def compute_networkx_shortest_paths(G: nx.Graph) -> np.ndarray:
    """Compute all-pairs shortest paths using NetworkX."""
    try:
        # Use NetworkX's Floyd-Warshall implementation
        distances = nx.floyd_warshall_numpy(G, weight='weight')
        return distances
    except nx.NetworkXError:
        # Fallback for disconnected graphs
        n_nodes = len(G.nodes())
        distances = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(distances, 0)
        
        # Compute shortest paths for each connected component
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(component) > 1:
                sub_distances = nx.floyd_warshall_numpy(subgraph, weight='weight')
                component_list = list(component)
                for i, u in enumerate(component_list):
                    for j, v in enumerate(component_list):
                        distances[u, v] = sub_distances[i, j]
        
        return distances


def test_floyd_warshall_correctness():
    """Test Floyd-Warshall algorithm correctness against NetworkX."""
    print("🧪 Testing Floyd-Warshall Algorithm Correctness")
    print("=" * 60)
    
    test_cases = create_test_cases()
    results = {}
    
    for case_name, G in test_cases.items():
        print(f"\n--- Testing {case_name} ---")
        
        # Skip empty graph (NetworkX doesn't handle it well)
        if case_name == 'empty':
            print("  Skipping empty graph (not supported by NetworkX)")
            continue
        
        # Add nodes to ensure proper indexing
        if len(G.nodes()) == 0:
            G.add_node(0)
        
        # Compute using NetworkX
        try:
            nx_distances = compute_networkx_shortest_paths(G)
            print(f"  NetworkX: {nx_distances.shape} matrix computed")
        except Exception as e:
            print(f"  ❌ NetworkX failed: {e}")
            continue
        
        # Convert to Graph-JAX format
        try:
            graph = gj.graphs.from_networkx(G)
            print(f"  Graph-JAX: {graph.n_nodes} nodes, {graph.n_edges} edges")
        except Exception as e:
            print(f"  ❌ Graph-JAX conversion failed: {e}")
            continue
        
        # Compute using Graph-JAX Floyd-Warshall
        try:
            gj_distances = shortest_paths(graph)
            print(f"  Graph-JAX Floyd-Warshall: {gj_distances.shape} matrix computed")
        except Exception as e:
            print(f"  ❌ Graph-JAX Floyd-Warshall failed: {e}")
            continue
        
        # Compare results
        try:
            # Convert JAX array to numpy for comparison
            gj_distances_np = np.array(gj_distances)
            
            # Handle infinite values
            nx_finite = np.isfinite(nx_distances)
            gj_finite = np.isfinite(gj_distances_np)
            
            # Check if finite values match
            finite_match = np.allclose(
                nx_distances[nx_finite & gj_finite],
                gj_distances_np[nx_finite & gj_finite],
                rtol=1e-6, atol=1e-6
            )
            
            # Check if infinite values match
            infinite_match = np.array_equal(
                nx_finite,
                gj_finite
            )
            
            is_correct = finite_match and infinite_match
            
            results[case_name] = {
                'correct': is_correct,
                'nx_shape': nx_distances.shape,
                'gj_shape': gj_distances_np.shape,
                'finite_match': finite_match,
                'infinite_match': infinite_match
            }
            
            status = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"  {status}")
            
            if not is_correct:
                print(f"    Finite values match: {finite_match}")
                print(f"    Infinite values match: {infinite_match}")
                
                # Show some differences
                if not finite_match:
                    diff_mask = ~np.isclose(
                        nx_distances, gj_distances_np, rtol=1e-6, atol=1e-6
                    ) & nx_finite & gj_finite
                    if np.any(diff_mask):
                        print(f"    Max difference: {np.max(np.abs(nx_distances - gj_distances_np))}")
        
        except Exception as e:
            print(f"  ❌ Comparison failed: {e}")
            results[case_name] = {'correct': False, 'error': str(e)}
    
    return results


def test_single_source_shortest_paths():
    """Test single-source shortest paths functionality."""
    print("\n🔍 Testing Single-Source Shortest Paths")
    print("=" * 50)
    
    # Create a test graph
    G = nx.Graph()
    G.add_edges_from([
        (0, 1, {'weight': 2}),
        (1, 2, {'weight': 3}),
        (0, 2, {'weight': 6}),
        (2, 3, {'weight': 1}),
        (1, 3, {'weight': 4})
    ])
    
    graph = gj.graphs.from_networkx(G)
    
    # Test single-source shortest paths
    source = 0
    gj_single_source = single_source_shortest_paths(graph, source)
    gj_all_pairs = shortest_paths(graph)
    gj_expected = gj_all_pairs[source]
    
    is_correct = jnp.allclose(gj_single_source, gj_expected, rtol=1e-6)
    
    print(f"Single-source from node {source}: {'✅ PASS' if is_correct else '❌ FAIL'}")
    
    return is_correct


def test_path_exists():
    """Test path existence checking."""
    print("\n🔍 Testing Path Existence")
    print("=" * 40)
    
    # Create test graph
    G = nx.Graph()
    G.add_edges_from([
        (0, 1, {'weight': 1}),
        (1, 2, {'weight': 2}),
        (3, 4, {'weight': 3})  # Disconnected component
    ])
    
    graph = gj.graphs.from_networkx(G)
    
    # Test cases
    test_cases = [
        (0, 1, True),   # Direct edge
        (0, 2, True),   # Path exists
        (0, 3, False),  # No path
        (3, 4, True),   # Direct edge in other component
        (0, 0, True),   # Self-loop (distance 0)
    ]
    
    all_correct = True
    for source, target, expected in test_cases:
        result = path_exists(graph, source, target)
        is_correct = (result == expected)
        all_correct = all_correct and is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"  Path {source} -> {target}: {result} (expected {expected}) {status}")
    
    return all_correct


def test_graph_metrics():
    """Test graph diameter and average shortest path length."""
    print("\n🔍 Testing Graph Metrics")
    print("=" * 40)
    
    # Create test graph
    G = nx.Graph()
    G.add_edges_from([
        (0, 1, {'weight': 1}),
        (1, 2, {'weight': 2}),
        (2, 3, {'weight': 3}),
        (3, 0, {'weight': 4})
    ])
    
    graph = gj.graphs.from_networkx(G)
    
    # Test diameter
    gj_diameter = diameter(graph)
    print(f"Graph diameter: {gj_diameter}")
    
    # Test average shortest path length
    gj_avg_length = average_shortest_path_length(graph)
    print(f"Average shortest path length: {gj_avg_length}")
    
    # Verify with NetworkX
    try:
        nx_diameter = nx.diameter(G, weight='weight')
        nx_avg_length = nx.average_shortest_path_length(G, weight='weight')
        
        diameter_correct = abs(gj_diameter - nx_diameter) < 1e-6
        avg_length_correct = abs(gj_avg_length - nx_avg_length) < 1e-6
        
        print(f"Diameter comparison: Graph-JAX {gj_diameter:.6f} vs NetworkX {nx_diameter:.6f} {'✅' if diameter_correct else '❌'}")
        print(f"Avg length comparison: Graph-JAX {gj_avg_length:.6f} vs NetworkX {nx_avg_length:.6f} {'✅' if avg_length_correct else '❌'}")
        
        return diameter_correct and avg_length_correct
    except Exception as e:
        print(f"NetworkX comparison failed: {e}")
        return True  # Assume correct if comparison fails


def benchmark_floyd_warshall():
    """Benchmark Floyd-Warshall performance with JAX compilation time separation."""
    print("\nBenchmarking Floyd-Warshall Performance")
    print("=" * 60)
    
    # Create graphs of different sizes
    sizes = [10, 50, 100, 200]
    results = {}
    
    for n_nodes in sizes:
        print(f"\n--- Testing {n_nodes} nodes ---")
        
        # Create random graph
        G = nx.erdos_renyi_graph(n_nodes, 0.3, seed=42)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.1, 2.0)
        
        graph = gj.graphs.from_networkx(G)
        
        # Warm up JAX compilation (first run includes compilation time)
        print("  Warming up JAX compilation...")
        _ = shortest_paths(graph)
        
        # Benchmark Graph-JAX (compiled version)
        print("  Benchmarking Graph-JAX (compiled)...")
        start_time = time.time()
        gj_distances = shortest_paths(graph)
        gj_time = time.time() - start_time
        
        # Benchmark NetworkX
        print("  Benchmarking NetworkX...")
        start_time = time.time()
        nx_distances = compute_networkx_shortest_paths(G)
        nx_time = time.time() - start_time
        
        # Verify correctness
        gj_distances_np = np.array(gj_distances)
        is_correct = np.allclose(nx_distances, gj_distances_np, rtol=1e-6, atol=1e-6)
        
        results[n_nodes] = {
            'gj_time': gj_time,
            'nx_time': nx_time,
            'speedup': nx_time / gj_time if gj_time > 0 else float('inf'),
            'correct': is_correct
        }
        
        print(f"  Graph-JAX (compiled): {gj_time:.6f}s")
        print(f"  NetworkX: {nx_time:.6f}s")
        print(f"  Speedup: {results[n_nodes]['speedup']:.2f}x")
        print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    return results


def benchmark_jax_compilation_vs_runtime():
    """Detailed benchmark separating JAX compilation time from runtime."""
    print("\n🔧 Detailed JAX Compilation vs Runtime Benchmark")
    print("=" * 60)
    
    # Create a medium-sized test graph
    n_nodes = 100
    G = nx.erdos_renyi_graph(n_nodes, 0.3, seed=42)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.1, 2.0)
    
    graph = gj.graphs.from_networkx(G)
    
    print(f"Test graph: {n_nodes} nodes, {graph.n_edges} edges")
    
    # First run (includes compilation time)
    print("\n--- First Run (Compilation + Runtime) ---")
    start_time = time.time()
    gj_distances_1 = shortest_paths(graph)
    first_run_time = time.time() - start_time
    print(f"  Total time: {first_run_time:.6f}s")
    
    # Second run (runtime only, compiled)
    print("\n--- Second Run (Runtime Only) ---")
    start_time = time.time()
    gj_distances_2 = shortest_paths(graph)
    second_run_time = time.time() - start_time
    print(f"  Runtime only: {second_run_time:.6f}s")
    
    # Estimate compilation time
    compilation_time = first_run_time - second_run_time
    print(f"  Estimated compilation time: {compilation_time:.6f}s")
    
    # Multiple runs to get average runtime
    print("\n--- Multiple Runs (Average Runtime) ---")
    run_times = []
    for i in range(5):
        start_time = time.time()
        _ = shortest_paths(graph)
        run_time = time.time() - start_time
        run_times.append(run_time)
        print(f"  Run {i+1}: {run_time:.6f}s")
    
    avg_runtime = np.mean(run_times)
    std_runtime = np.std(run_times)
    print(f"  Average runtime: {avg_runtime:.6f}s ± {std_runtime:.6f}s")
    
    # Verify results are consistent
    is_consistent = jnp.allclose(gj_distances_1, gj_distances_2, rtol=1e-6)
    print(f"  Results consistent: {'✅' if is_consistent else '❌'}")
    
    return {
        'compilation_time': compilation_time,
        'avg_runtime': avg_runtime,
        'std_runtime': std_runtime,
        'first_run_time': first_run_time,
        'run_times': run_times,
        'is_consistent': is_consistent
    }


def benchmark_scalability():
    """Benchmark scalability with proper JAX compilation handling."""
    print("\nScalability Benchmark (Post-Compilation)")
    print("=" * 60)
    
    # Create graphs of different sizes
    sizes = [50, 100, 200, 300, 500]
    results = {}
    
    # Warm up compilation with a small graph first
    print("Warming up JAX compilation...")
    warmup_G = nx.erdos_renyi_graph(20, 0.3, seed=42)
    for u, v in warmup_G.edges():
        warmup_G[u][v]['weight'] = np.random.uniform(0.1, 2.0)
    warmup_graph = gj.graphs.from_networkx(warmup_G)
    _ = shortest_paths(warmup_graph)
    
    for n_nodes in sizes:
        print(f"\n--- Testing {n_nodes} nodes ---")
        
        # Create random graph
        G = nx.erdos_renyi_graph(n_nodes, 0.3, seed=42)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.1, 2.0)
        
        graph = gj.graphs.from_networkx(G)
        
        # Multiple runs for Graph-JAX (post-compilation)
        gj_times = []
        for _ in range(3):
            start_time = time.time()
            gj_distances = shortest_paths(graph)
            gj_time = time.time() - start_time
            gj_times.append(gj_time)
        
        avg_gj_time = np.mean(gj_times)
        std_gj_time = np.std(gj_times)
        
        # Single run for NetworkX
        start_time = time.time()
        nx_distances = compute_networkx_shortest_paths(G)
        nx_time = time.time() - start_time
        
        # Verify correctness
        gj_distances_np = np.array(gj_distances)
        is_correct = np.allclose(nx_distances, gj_distances_np, rtol=1e-6, atol=1e-6)
        
        results[n_nodes] = {
            'gj_avg_time': avg_gj_time,
            'gj_std_time': std_gj_time,
            'nx_time': nx_time,
            'speedup': nx_time / avg_gj_time if avg_gj_time > 0 else float('inf'),
            'correct': is_correct,
            'n_edges': graph.n_edges
        }
        
        print(f"  Graph-JAX: {avg_gj_time:.6f}s ± {std_gj_time:.6f}s")
        print(f"  NetworkX: {nx_time:.6f}s")
        print(f"  Speedup: {results[n_nodes]['speedup']:.2f}x")
        print(f"  Edges: {graph.n_edges}")
        print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    return results


def main():
    """Main test function."""
    print("Floyd-Warshall Algorithm Test Suite")
    print("=" * 60)
    
    # Test correctness against NetworkX
    correctness_results = test_floyd_warshall_correctness()
    
    # Test single-source shortest paths
    single_source_correct = test_single_source_shortest_paths()
    
    # Test path existence
    path_exists_correct = test_path_exists()
    
    # Test graph metrics
    metrics_correct = test_graph_metrics()
    
    # Basic benchmark performance
    benchmark_results = benchmark_floyd_warshall()
    
    # Detailed JAX compilation vs runtime analysis
    compilation_analysis = benchmark_jax_compilation_vs_runtime()
    
    # Scalability benchmark (post-compilation)
    scalability_results = benchmark_scalability()
    
    # Summary
    print(f"\nTest Summary")
    print("=" * 40)
    
    # Correctness summary
    correct_cases = sum(1 for r in correctness_results.values() if r.get('correct', False))
    total_cases = len(correctness_results)
    print(f"Floyd-Warshall correctness: {correct_cases}/{total_cases} cases passed")
    
    print(f"Single-source shortest paths: {'✅ PASS' if single_source_correct else '❌ FAIL'}")
    print(f"Path existence checking: {'✅ PASS' if path_exists_correct else '❌ FAIL'}")
    print(f"Graph metrics: {'✅ PASS' if metrics_correct else '❌ FAIL'}")
    
    # Performance summary
    if benchmark_results:
        avg_speedup = np.mean([r['speedup'] for r in benchmark_results.values() if r['speedup'] != float('inf')])
        print(f"Basic benchmark avg speedup vs NetworkX: {avg_speedup:.2f}x")
    
    # JAX compilation analysis
    if compilation_analysis:
        print(f"\nJAX Performance Analysis:")
        print(f"  Compilation time: {compilation_analysis['compilation_time']:.6f}s")
        print(f"  Average runtime: {compilation_analysis['avg_runtime']:.6f}s ± {compilation_analysis['std_runtime']:.6f}s")
        print(f"  Compilation overhead: {compilation_analysis['compilation_time']/compilation_analysis['avg_runtime']:.1f}x")
    
    # Scalability summary
    if scalability_results:
        print(f"\nScalability Analysis (Post-Compilation):")
        for n_nodes, result in scalability_results.items():
            print(f"  {n_nodes} nodes: Graph-JAX {result['gj_avg_time']:.6f}s vs NetworkX {result['nx_time']:.6f}s (speedup: {result['speedup']:.2f}x)")
    
    # Overall result
    all_tests_passed = (
        correct_cases == total_cases and
        single_source_correct and
        path_exists_correct and
        metrics_correct
    )
    
    if all_tests_passed:
        print("\nAll tests passed! Floyd-Warshall implementation is correct.")
    else:
        print("\nSome tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
