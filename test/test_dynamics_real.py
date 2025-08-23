#!/usr/bin/env python3
"""
Large-scale testing using the complete California road network dataset roadNet-CA.txt
Optimized version: supports multi-core parallel computing, fully compliant with paper algorithm
"""

# Configure multi-core computing
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import graph_jax as gj
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import random
from collections import defaultdict
import psutil
import os

from graph_jax.utils import set_backend

# Set CPU backend to avoid GPU memory issues
set_backend('cpu')

# Configure matplotlib to not display plots, only save
plt.ioff()
import matplotlib
matplotlib.use('Agg')

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def get_peak_memory():
    """Get peak memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    # Use rss as fallback since peak_wset might not be available on all platforms
    return getattr(memory_info, 'peak_wset', memory_info.rss) / 1024 / 1024  # MB

def load_roadnet_ca(max_nodes=None, subsample_factor=1):
    """
    Load California road network data
    
    Args:
        max_nodes: Maximum node limit (for testing)
        subsample_factor: Subsampling factor, 1 means use all data
    
    Returns:
        NetworkX graph object
    """
    print(f"Loading road network...")
    
    G = nx.Graph()
    node_degrees = defaultdict(int)
    
    # Read and parse file
    start_time = time.time()
    with open('roadNet-CA.txt', 'r') as f:
        lines = f.readlines()
    
    edge_count = 0
    node_set = set()
    
    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue
        if i % subsample_factor != 0:
            continue
            
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                u, v = int(parts[0]), int(parts[1])
                if max_nodes and (u >= max_nodes or v >= max_nodes):
                    continue
                if u != v:
                    G.add_edge(u, v)
                    node_degrees[u] += 1
                    node_degrees[v] += 1
                    node_set.add(u)
                    node_set.add(v)
                    edge_count += 1
            except ValueError:
                continue
    
    load_time = time.time() - start_time
    
    # Set edge capacities
    capacity_start = time.time()
    for u, v in G.edges():
        capacity = node_degrees[u] + node_degrees[v]
        G[u][v]['capacity'] = float(capacity)
    
    capacity_time = time.time() - capacity_start
    total_time = time.time() - start_time
    
    print(f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Load time: {total_time:.2f}s (parse: {load_time:.2f}s, capacity: {capacity_time:.2f}s)")
    
    return G

def create_road_network_partitions(G):
    """
    Create three-layer partitions for road network:
    - Residential (low-degree nodes)
    - Arterial (medium-degree nodes)  
    - Highway (high-degree nodes)
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    # Calculate node degrees
    degrees = np.array([G.degree(node) for node in nodes])
    
    # Use more balanced partitioning strategy
    # Ensure each layer has at least a certain number of nodes
    min_nodes_per_layer = max(1, n_nodes // 10)  # At least 10% of nodes
    
    if n_nodes < 3:
        # If too few nodes, simple even distribution
        ref_mask = np.zeros(n_nodes, dtype=bool)
        term_mask = np.zeros(n_nodes, dtype=bool)
        gas_mask = np.zeros(n_nodes, dtype=bool)
        
        ref_mask[:n_nodes//3] = True
        term_mask[n_nodes//3:2*n_nodes//3] = True
        gas_mask[2*n_nodes//3:] = True
    else:
        # Balanced partitioning based on degree and spatial distribution
        unique_degrees = np.unique(degrees)
        
        if len(unique_degrees) <= 2:
            # Limited degree variation, use node ID partitioning
            print("Limited degree variation, using spatial partitioning...")
            node_indices = np.arange(n_nodes)
            p33_idx = n_nodes // 3
            p66_idx = 2 * n_nodes // 3
            
            ref_mask = node_indices < p33_idx
            term_mask = (node_indices >= p33_idx) & (node_indices < p66_idx)
            gas_mask = node_indices >= p66_idx
        else:
            # Use degree percentiles but ensure balance
            p25 = np.percentile(degrees, 25)
            p75 = np.percentile(degrees, 75)
            
            ref_mask = degrees <= p25
            gas_mask = degrees >= p75
            term_mask = ~ref_mask & ~gas_mask
            
            # If any layer is too small, rebalance
            if np.sum(ref_mask) < min_nodes_per_layer or np.sum(term_mask) < min_nodes_per_layer or np.sum(gas_mask) < min_nodes_per_layer:
                print("Rebalancing partitions to ensure minimum size...")
                node_indices = np.arange(n_nodes)
                np.random.seed(42)  # Fixed seed for reproducible results
                np.random.shuffle(node_indices)  # Random shuffle to ensure fair distribution
                
                n_per_layer = n_nodes // 3
                ref_mask = np.zeros(n_nodes, dtype=bool)
                term_mask = np.zeros(n_nodes, dtype=bool)
                gas_mask = np.zeros(n_nodes, dtype=bool)
                
                ref_mask[node_indices[:n_per_layer]] = True
                term_mask[node_indices[n_per_layer:2*n_per_layer]] = True
                gas_mask[node_indices[2*n_per_layer:]] = True
    
    print(f"Network partitions:")
    print(f"  Residential (ref_mask): {np.sum(ref_mask):,} nodes ({np.sum(ref_mask)/n_nodes*100:.1f}%)")
    print(f"  Arterial (term_mask): {np.sum(term_mask):,} nodes ({np.sum(term_mask)/n_nodes*100:.1f}%)")
    print(f"  Highway (gas_mask): {np.sum(gas_mask):,} nodes ({np.sum(gas_mask)/n_nodes*100:.1f}%)")
    
    # Ensure no empty layers
    if np.sum(ref_mask) == 0 or np.sum(term_mask) == 0 or np.sum(gas_mask) == 0:
        raise ValueError("One or more network layers is empty. Please check data or adjust partitioning.")
    
    return ref_mask, term_mask, gas_mask

def compute_capacity_params_networkx(G, ref_mask, term_mask, gas_mask):
    """
    Compute capacity parameters using NetworkX - compliant with paper algorithm
    Implements minimum cut algorithm from paper, including direct connections and indirect paths
    """
    print("Computing capacity parameters using NetworkX (paper algorithm)...")
    
    # Node counts
    N1 = np.sum(ref_mask)
    N2 = np.sum(term_mask)
    N3 = np.sum(gas_mask)
    
    print(f"Node counts: N1={N1:,}, N2={N2:,}, N3={N3:,}")
    
    # Precompute node-to-index mapping
    print("Precomputing node-to-index mapping...")
    start_time = time.time()
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    mapping_time = time.time() - start_time
    print(f"Node mapping created in {mapping_time:.2f} seconds")
    
    # Precompute layer node sets
    print("Precomputing layer node sets...")
    start_time = time.time()
    ref_nodes = set(np.where(ref_mask)[0])
    term_nodes = set(np.where(term_mask)[0])
    gas_nodes = set(np.where(gas_mask)[0])
    layer_time = time.time() - start_time
    print(f"Layer sets created in {layer_time:.2f} seconds")
    
    # Compute inter-layer minimum cut capacity - compliant with paper algorithm
    def compute_min_cut_between_layers_paper(source_nodes, sink_nodes):
        if len(source_nodes) == 0 or len(sink_nodes) == 0:
            return 0.0
        
        # Find intermediate nodes (nodes that are neither source nor sink)
        all_nodes = set(range(len(nodes)))
        intermediate_nodes = all_nodes - source_nodes - sink_nodes
        
        print(f"  Processing {G.number_of_edges():,} edges...")
        start_time = time.time()
        
        # Calculate direct connection capacity
        direct_capacity = 0.0
        direct_edges = 0
        
        for u, v, data in G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # Check if this is a direct source-to-sink connection
            if (u_idx in source_nodes and v_idx in sink_nodes) or (u_idx in sink_nodes and v_idx in source_nodes):
                direct_capacity += data['capacity']
                direct_edges += 1
        
        # Calculate path capacity through intermediate nodes
        source_to_intermediate = 0.0
        intermediate_to_sink = 0.0
        
        for u, v, data in G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # Source to intermediate node edges
            if (u_idx in source_nodes and v_idx in intermediate_nodes) or (u_idx in intermediate_nodes and v_idx in source_nodes):
                source_to_intermediate += data['capacity']
            
            # Intermediate node to sink edges
            if (u_idx in intermediate_nodes and v_idx in sink_nodes) or (u_idx in sink_nodes and v_idx in intermediate_nodes):
                intermediate_to_sink += data['capacity']
        
        # According to max-flow min-cut theorem, take the smaller value as bottleneck
        indirect_capacity = min(source_to_intermediate, intermediate_to_sink)
        
        # Total capacity = direct capacity + indirect capacity (compliant with paper algorithm)
        total_capacity = direct_capacity + indirect_capacity
        
        process_time = time.time() - start_time
        print(f"  Found {direct_edges:,} direct edges, {len(intermediate_nodes):,} intermediate nodes")
        print(f"  Direct capacity: {direct_capacity:.2f}, Indirect capacity: {indirect_capacity:.2f}")
        print(f"  Total capacity: {total_capacity:.2f} in {process_time:.2f} seconds")
        print(f"  Processing rate: {G.number_of_edges()/process_time:.0f} edges/sec")
        
        return total_capacity
    
    print("\nComputing W12 (residential→arterial)...")
    start_time = time.time()
    W12 = compute_min_cut_between_layers_paper(ref_nodes, term_nodes)
    w12_time = time.time() - start_time
    print(f"W12 computed in {w12_time:.2f}s: {W12:.2f}")
    
    print("\nComputing W23 (arterial→highway)...")
    start_time = time.time()
    W23 = compute_min_cut_between_layers_paper(term_nodes, gas_nodes)
    w23_time = time.time() - start_time
    print(f"W23 computed in {w23_time:.2f}s: {W23:.2f}")
    
    print("\nComputing W13 (residential→highway)...")
    start_time = time.time()
    W13 = compute_min_cut_between_layers_paper(ref_nodes, gas_nodes)
    w13_time = time.time() - start_time
    print(f"W13 computed in {w13_time:.2f}s: {W13:.2f}")
    
    # 道路网络特定的容量常数
    C1, C2, C3 = 50.0, 100.0, 200.0  # 调整为道路网络合适的值
    
    # 计算参数
    s12 = W12 / (N1 * C1 + 1e-8)
    s23 = W23 / (N2 * C2 + 1e-8)
    s13 = W13 / (N1 * C1 + 1e-8)
    
    alpha12 = (N2 * C2) / (N1 * C1 + 1e-8)
    alpha23 = (N3 * C3) / (N2 * C2 + 1e-8)
    
    return {
        'N1': N1, 'N2': N2, 'N3': N3,
        'W12': W12, 'W23': W23, 'W13': W13,
        's12': s12, 's23': s23, 's13': s13,
        'alpha12': alpha12, 'alpha23': alpha23,
        'p': 0.1,   # 道路网络生产率（较低）
        'd': 0.05   # 道路网络消耗率（较低）
    }

def compute_steady_state_scipy(params):
    """
    Use SciPy to calculate steady state
    Note: Both Graph-JAX and NetworkX now use SciPy for ODE solving consistency
    """
    from scipy.integrate import odeint
    
    def rhs(y, t, params):
        y1, y2, y3 = y
        p, d = params["p"], params["d"]
        s12, s23, s13 = params["s12"], params["s23"], params["s13"]
        a12, a23 = params["alpha12"], params["alpha23"]
        
        dy1 = p - s12 * y1 * y2 - s13 * y1 * y3
        dy2 = s12 / a12 * y1 * y2 - s23 * y2 * y3
        dy3 = -d * y3 + s13 / (a12 * a23) * y1 * y3 + s23 / a23 * y2 * y3
        return [dy1, dy2, dy3]
    
    y0 = [1.0, 1.0, 1.0]
    t = np.linspace(0, 100.0, 1000)  # Longer time to ensure convergence
    solution = odeint(rhs, y0, t, args=(params,))
    return solution[-1]

def main():
    print("=" * 80)
    print("FULL California Road Network Analysis - Step by Step")
    print("Dataset: Complete roadNet-CA.txt (1.9M nodes, 5.5M edges)")
    print("=" * 80)
    
    # Parse command line arguments
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Graph-JAX vs NetworkX on California road network')
    parser.add_argument('--test-mode', choices=['jax', 'networkx', 'both'], default='both',
                       help='Test mode: jax, networkx, or both')
    parser.add_argument('--max-nodes', type=int, default=None,
                       help='Maximum number of nodes to load (for testing)')
    parser.add_argument('--subsample-factor', type=int, default=1,
                       help='Subsampling factor for edges')
    
    args = parser.parse_args()
    test_mode = args.test_mode
    MAX_NODES = args.max_nodes
    SUBSAMPLE_FACTOR = args.subsample_factor
    
    print(f"Test mode: {test_mode}")
    print(f"Max nodes: {MAX_NODES}")
    print(f"Subsample factor: {SUBSAMPLE_FACTOR}")
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # 1. Load data
    print("\n1. Loading Road Network Data")
    print("-" * 40)
    
    try:
        G = load_roadnet_ca(max_nodes=MAX_NODES, subsample_factor=SUBSAMPLE_FACTOR)
        
        if G.number_of_nodes() == 0:
            print("Error: No nodes loaded. Check data file format.")
            return
        
        # Sample edge capacities
        sample_edges = random.sample(list(G.edges(data=True)), min(10000, G.number_of_edges()))
        sample_capacities = [data['capacity'] for _, _, data in sample_edges]
        avg_capacity = np.mean(sample_capacities)
        
    except Exception as e:
        print(f"Error loading network: {e}")
        return
    
    # 2. Create network partitions
    print("\n2. Creating Network Partitions")
    print("-" * 40)
    
    try:
        ref_mask, term_mask, gas_mask = create_road_network_partitions(G)
        print(f"Partitions: {np.sum(ref_mask)}, {np.sum(term_mask)}, {np.sum(gas_mask)} nodes")
    except Exception as e:
        print(f"Error creating partitions: {e}")
        return
    
    # Decide which steps to execute based on test mode
    if test_mode in ["jax", "both"]:
        # 3. Graph-JAX computation
        print("\n3. Computing Graph-JAX Solution")
        print("-" * 40)
        
        try:
            # Convert NetworkX to Graph-JAX (for fair comparison)
            start_time = time.time()
            g = gj.from_networkx(G)
            g.node_mask = jnp.ones(g.n_nodes, dtype=bool)
            conversion_time = time.time() - start_time
            
            # Capacity parameters
            start_time = time.time()
            params_jax = gj.algorithms.capacity_params(
                g, ref_mask, term_mask, gas_mask, 
                edge_cap=avg_capacity,
                C1=50.0, C2=100.0, C3=200.0,
                use_parallel=True
            )
            jax_capacity_time = time.time() - start_time
            
            # Steady state
            start_time = time.time()
            params_jax.update({'p': 0.1, 'd': 0.05})
            y_steady_jax = gj.algorithms.steady_state(params_jax, t_max=100.0, n_steps=1000)
            jax_steady_time = time.time() - start_time
            
            # Failure time
            start_time = time.time()
            τ_jax, QD_jax = gj.algorithms.failure_time(params_jax, ΔT=7.0)
            jax_failure_time = time.time() - start_time
            
            print(f"Graph-JAX Results:")
            print(f"  Conversion: {conversion_time:.3f}s")
            print(f"  Capacity: {jax_capacity_time:.3f}s")
            print(f"  Steady state: {jax_steady_time:.3f}s")
            print(f"  Failure time: {jax_failure_time:.3f}s")
            print(f"  Total: {conversion_time + jax_capacity_time + jax_steady_time + jax_failure_time:.3f}s")
            print(f"  Graph: {g}")  # 使用新的字符串表示
            print(f"  s12={params_jax['s12']:.4f}, s23={params_jax['s23']:.4f}, s13={params_jax['s13']:.4f}")
            print(f"  Steady: {y_steady_jax}")
            print(f"  Failure: τ={τ_jax:.2f}d, QD={QD_jax:.3f}")
            
        except Exception as e:
            print(f"Error computing Graph-JAX solution: {e}")
            return
        
        # Clean up Graph-JAX related memory
        del g
        gc.collect()
        jax_peak_memory = get_peak_memory()
        print(f"JAX peak memory: {jax_peak_memory:.1f} MB")
    
    if test_mode in ["networkx", "both"]:
        # 4. NetworkX computation
        print("\n4. Computing NetworkX Solution")
        print("-" * 40)
        
        try:
            # Capacity parameters
            start_time = time.time()
            params_correct = compute_capacity_params_networkx(G, ref_mask, term_mask, gas_mask)
            nx_capacity_time = time.time() - start_time
            
            # Steady state
            start_time = time.time()
            y_steady_correct = compute_steady_state_scipy(params_correct)
            nx_steady_time = time.time() - start_time
            
            print(f"NetworkX Results:")
            print(f"  Capacity: {nx_capacity_time:.3f}s")
            print(f"  Steady state: {nx_steady_time:.3f}s")
            print(f"  Total: {nx_capacity_time + nx_steady_time:.3f}s")
            print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"  s12={params_correct['s12']:.4f}, s23={params_correct['s23']:.4f}, s13={params_correct['s13']:.4f}")
            print(f"  Steady: {y_steady_correct}")
            
        except Exception as e:
            print(f"Error computing NetworkX solution: {e}")
            return
        
        nx_peak_memory = get_peak_memory()
        print(f"NetworkX peak memory: {nx_peak_memory:.1f} MB")
    
    # Results comparison (only when both methods are run)
    if test_mode == "both":
        print("\n5. Results Comparison")
        print("-" * 40)
        
        try:
            # Ensure variables exist
            if 'params_jax' not in locals() or 'params_correct' not in locals():
                print("Error: Required variables not found for comparison")
                return
            
            # Calculate errors
            print("Capacity parameter errors:")
            for key in ['s12', 's23', 's13']:
                if key in params_correct and key in params_jax:
                    error = abs(params_jax[key] - params_correct[key])
                    rel_error = error / (abs(params_correct[key]) + 1e-8) * 100
                    print(f"  {key}: {error:.6f} ({rel_error:.2f}%)")
            
            # Steady state errors
            if 'y_steady_jax' in locals() and 'y_steady_correct' in locals():
                steady_error = np.array(y_steady_jax) - np.array(y_steady_correct)
                steady_rel_error = steady_error / (np.array(y_steady_correct) + 1e-8) * 100
                print(f"Steady state errors: {steady_error}")
                print(f"Relative errors: {steady_rel_error}%")
            
            # Performance comparison
            jax_total = jax_capacity_time + jax_steady_time + jax_failure_time
            nx_total = nx_capacity_time + nx_steady_time
            speedup = nx_total / jax_total
            print(f"Performance: JAX=Convertion time {conversion_time:.3f}s, Computation time {jax_total:.3f}s, NX={nx_total:.3f}s, Speedup={speedup:.2f}x")
            
            # Memory comparison
            if 'jax_peak_memory' in locals() and 'nx_peak_memory' in locals():
                print(f"Memory: JAX={jax_peak_memory:.1f}MB, NX={nx_peak_memory:.1f}MB")
            
        except Exception as e:
            print(f"Error in comparison: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Analysis Completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
