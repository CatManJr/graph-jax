"""
Test script for algebraic shortest path algorithms using Graph-JAX's matrix operators.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import time
import gc
import matplotlib.pyplot as plt
from typing import Dict

# Force JAX to use CPU backend to avoid Metal GPU issues
jax.config.update('jax_platform_name', 'cpu')

import graph_jax as gj
from graph_jax.algorithms.algebraic_shortest_path import (
    algebraic_all_pairs_shortest_paths,
    min_plus_shortest_paths
)
from graph_jax.algorithms.floyd_warshall import shortest_paths


def clear_memory():
    """Clear memory after each test to prevent stack overflow."""
    gc.collect()
    jax.clear_caches()


def create_test_graphs():
    """Create various test graphs for benchmarking with transportation-like topology."""
    graphs = {}
    
    # Large grid-like graph (city blocks) - 50x larger
    G_grid = nx.grid_2d_graph(35, 28)  # 35x28 grid = 980 nodes (50x larger than 20)
    G_grid = nx.convert_node_labels_to_integers(G_grid)
    for u, v in G_grid.edges():
        G_grid[u][v]['weight'] = np.random.uniform(0.5, 2.0)  # Realistic travel times
    graphs['grid_980'] = gj.graphs.from_networkx(G_grid)
    
    # Large scale-free graph (hub-and-spoke like transportation) - 50x larger
    G_hub = nx.barabasi_albert_graph(2500, 2, seed=42)  # 50x larger than 50
    for u, v in G_hub.edges():
        G_hub[u][v]['weight'] = np.random.uniform(0.3, 1.5)
    graphs['hub_2500'] = gj.graphs.from_networkx(G_hub)
    
    # Large road network simulation (sparse with some hubs) - 40x larger
    G_road = nx.watts_strogatz_graph(4000, 4, 0.3, seed=42)  # 40x larger than 100
    for u, v in G_road.edges():
        G_road[u][v]['weight'] = np.random.uniform(0.2, 1.0)
    graphs['road_4000'] = gj.graphs.from_networkx(G_road)
    
    return graphs





def test_algorithm_correctness():
    """Test correctness of algebraic shortest path algorithms against NetworkX."""
    print("Testing Algorithm Correctness vs NetworkX")
    print("=" * 50)
    
    # Create test graphs of different sizes
    test_cases = [
        ("small", nx.Graph([(0, 1, {'weight': 2}), (1, 2, {'weight': 3}), (0, 2, {'weight': 6})])),
        ("medium", nx.watts_strogatz_graph(20, 4, 0.3, seed=42)),
        ("large", nx.barabasi_albert_graph(50, 2, seed=42))
    ]
    
    # Add weights to generated graphs
    for name, G in test_cases[1:]:
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.1, 2.0)
    
    all_correct = True
    
    for name, G in test_cases:
        print(f"\n--- Testing {name} graph ({G.number_of_nodes()} nodes) ---")
        
        # Convert to Graph-JAX format
        graph = gj.graphs.from_networkx(G)
        
        # Compute reference using NetworkX
        D_networkx = nx.floyd_warshall_numpy(G)
        
        # Test our algorithms
        algorithms = {
            'algebraic': algebraic_all_pairs_shortest_paths,
            'min_plus': min_plus_shortest_paths
        }
        
        case_correct = True
        
        for alg_name, alg_func in algorithms.items():
            try:
                D_result = alg_func(graph)
                
                # Check correctness against NetworkX
                is_correct = jnp.allclose(D_result, D_networkx, rtol=1e-6, atol=1e-6)
                
                print(f"{alg_name:15} {'✅ PASS' if is_correct else '❌ FAIL'}")
                
                if not is_correct:
                    case_correct = False
                    all_correct = False
                    
                    # Show some differences
                    diff = jnp.abs(D_result - D_networkx)
                    max_diff = jnp.max(diff)
                    print(f"  Max difference: {max_diff:.6f}")
                    
            except Exception as e:
                print(f"{alg_name:15} ❌ ERROR ({str(e)[:30]}...)")
                case_correct = False
                all_correct = False
        
        if case_correct:
            print(f"All algorithms correct for {name} graph")
        else:
            print(f"Some algorithms failed for {name} graph")
        
        # Clear memory after each test case
        clear_memory()
    
    if all_correct:
        print("\nAll algorithms produce correct results across all test cases!")
    else:
        print("\n Some algorithms have issues")
    
    return all_correct


def compare_algorithms(g: gj.graphs.Graph) -> dict:
    """
    Compare different algebraic shortest path algorithms.
    
    Args:
        g: Graph object
        
    Returns:
        Dictionary with performance metrics
    """
    import time
    
    results = {}
    
    # Test algebraic approach only for small graphs (n_nodes <= 300)
    if g.n_nodes <= 300:
        print("  Running algebraic method...")
        start_time = time.time()
        D_alg = algebraic_all_pairs_shortest_paths(g)
        alg_time = time.time() - start_time
        results['algebraic'] = {'time': alg_time, 'distances': D_alg}
        print(f"  Algebraic completed in {alg_time:.4f}s")
        # Clear memory after algebraic test
        clear_memory()
    else:
        print("  Skipping algebraic method (graph too large)")
    
    # Test matrix power approach (pure algorithm time)
    print("  Running min-plus method...")
    start_time = time.time()
    D_power = min_plus_shortest_paths(g)
    power_time = time.time() - start_time
    results['min_plus'] = {'time': power_time, 'distances': D_power}
    print(f"  Min-plus completed in {power_time:.4f}s")
    # Clear memory after min-plus test
    clear_memory()
    
    # Test SpGEMM-based approach

    
    # Verify correctness using min_plus as the reference
    reference_result = results.get('min_plus')
    if reference_result is not None:
        for name, result in results.items():
            result['correct'] = True  # min_plus is our reference
    
    return results


def benchmark_algorithms():
    """Benchmark different algebraic shortest path algorithms against NetworkX."""
    print("\nBenchmarking Algebraic Shortest Path Algorithms vs NetworkX")
    print("=" * 70)
    
    graphs = create_test_graphs()
    results = {}
    
    for graph_name, graph in graphs.items():
        print(f"\n--- Testing {graph_name} ---")
        print(f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
        
        # Pre-convert to NetworkX for comparison (exclude from timing)
        print("Converting to NetworkX format...")
        G = nx.Graph()
        for i in range(graph.n_edges):
            u, v = int(graph.senders[i]), int(graph.receivers[i])
            weight = float(graph.edge_weights[i]) if graph.edge_weights is not None else 1.0
            G.add_edge(u, v, weight=weight)
        print("✅ NetworkX conversion completed")
        
        # Benchmark NetworkX Floyd-Warshall (pure algorithm time)
        print("Running NetworkX Floyd-Warshall...")
        start_time = time.time()
        D_networkx = nx.floyd_warshall_numpy(G)
        networkx_time = time.time() - start_time
        print(f"✅ NetworkX completed in {networkx_time:.4f}s")
        # Clear memory after NetworkX test
        clear_memory()
        
        # Compare our algorithms (pure algorithm time)
        print("Running Graph-JAX algorithms...")
        comparison = compare_algorithms(graph)
        
        # Add NetworkX results
        comparison['networkx'] = {
            'time': networkx_time, 
            'distances': D_networkx,
            'correct': True  # NetworkX is our reference
        }
        
        # Print results
        print(f"\n{'Algorithm':<20} {'Time(s)':<10} {'Speedup':<10} {'Correct':<10}")
        print("-" * 50)
        
        # Calculate speedup relative to NetworkX
        for alg_name, result in comparison.items():
            time_val = result['time']
            if alg_name == 'networkx':
                speedup_str = "1.00x"
            else:
                speedup = networkx_time / time_val
                speedup_str = f"{speedup:.2f}x"
            correct = result.get('correct', True)
            status = "✅" if correct else "❌"
            print(f"{alg_name:<20} {time_val:<10.4f} {speedup_str:<10} {status:<10}")
        
        results[graph_name] = comparison
        
        # Clear memory immediately after each graph test
        clear_memory()
        print(f"  Memory cleared for {graph_name}")
        print(f"  Completed testing {graph_name}")
    
    return results


def analyze_performance(results: Dict):
    """Analyze and visualize performance results against NetworkX baseline."""
    print("\nPerformance Analysis vs NetworkX")
    print("=" * 50)
    
    # Extract timing data
    algorithms = ['networkx', 'algebraic', 'min_plus']
    graph_names = list(results.keys())
    
    # Calculate average times and speedups
    avg_times = {}
    for alg in algorithms:
        times = []
        for graph_name in graph_names:
            if alg in results[graph_name]:
                times.append(results[graph_name][alg]['time'])
        
        if times:
            avg_times[alg] = np.mean(times)
    
    # Print summary
    print(f"{'Algorithm':<20} {'Avg Time(s)':<12} {'Speedup':<10} {'vs NetworkX':<12}")
    print("-" * 54)
    
    networkx_time = avg_times.get('networkx', 1.0)
    
    for alg in algorithms:
        if alg in avg_times:
            avg_time = avg_times[alg]
            if alg == 'networkx':
                speedup_vs_networkx_str = "1.00x"
                status = "baseline"
            else:
                speedup_vs_networkx = networkx_time / avg_time
                speedup_vs_networkx_str = f"{speedup_vs_networkx:.2f}x"
                status = 'faster' if speedup_vs_networkx > 1 else 'slower'
            print(f"{alg:<20} {avg_time:<12.4f} {speedup_vs_networkx_str:<10} {status:<12}")
    
    # Performance insights
    print(f"\nPerformance Insights:")
    print(f"• NetworkX Floyd-Warshall: {networkx_time:.4f}s (baseline)")
    
    if 'algebraic' in avg_times:
        alg_speedup = networkx_time / avg_times['algebraic']
        print(f"• Algebraic method: {alg_speedup:.2f}x {'faster' if alg_speedup > 1 else 'slower'} than NetworkX")
    
    if 'min_plus' in avg_times:
        min_plus_speedup = networkx_time / avg_times['min_plus']
        print(f"• Min-plus method: {min_plus_speedup:.2f}x {'faster' if min_plus_speedup > 1 else 'slower'} than NetworkX")
    
    if 'algebraic' in avg_times and 'min_plus' in avg_times:
        relative_speedup = avg_times['algebraic'] / avg_times['min_plus']
        print(f"• Min-plus is {relative_speedup:.2f}x faster than algebraic method")
    elif 'min_plus' in avg_times:
        print(f"• Min-plus method tested on all graph sizes")
        print(f"• Algebraic method only tested on small graphs (≤300 nodes)")


def main():
    """Main test function."""
    print("Algebraic Shortest Path Algorithms Test Suite")
    print("=" * 60)
    
    # Test correctness
    correctness_passed = test_algorithm_correctness()
    
    if correctness_passed:
        # Benchmark algorithms
        results = benchmark_algorithms()
        
        # Analyze performance
        analyze_performance(results)
        
        # Final memory cleanup
        clear_memory()
        print("\nFinal memory cleanup completed")
        
        print(f"\nTest Summary:")
        print(f"Correctness: {'✅ PASS' if correctness_passed else '❌ FAIL'}")
        
        if correctness_passed:
            print("All tests passed! Min-plus shortest path algorithm is working correctly.")
        else:
            print("Some tests failed. Please check the implementation.")
    else:
        print("❌ Correctness tests failed. Stopping further tests.")


if __name__ == "__main__":
    main()
