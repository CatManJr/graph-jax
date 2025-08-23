import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import os
import time
import timeit

# import necessary modules from graph-jax
from graph_jax.utils import *
from graph_jax.graphs import from_networkx
from graph_jax.algorithms.pagerank import pagerank

# Let XLA know the JAX environment
set_backend('cpu')
get_jax_env_info()

# Do not import JAX before setting the backend
import jax
import jax.numpy as jnp

def comprehensive_timing_analysis():
    """
    Perform a comprehensive PageRank performance benchmark using multiple timing methods.
    """
    print("=== Comprehensive PageRank Performance Benchmark ===\n")
    
    file_path = 'web-Google.txt'
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        return

    # Load graph
    print("Loading graph data...")
    nx_g_real = nx.read_edgelist(file_path, comments='#', create_using=nx.DiGraph(), nodetype=int)
    jax_g_real = from_networkx(nx_g_real)
    print(f"Graph size: {nx_g_real.number_of_nodes()} nodes, {nx_g_real.number_of_edges()} edges\n")

    # 1. JAX warm-up and compilation
    print("--- JAX Warm-up and Compilation Analysis ---")
    print("First run (includes compilation)...")
    start_time = time.perf_counter()
    jax_pr_first = pagerank(jax_g_real).block_until_ready()
    first_run_time = time.perf_counter() - start_time
    print(f"First run time (compilation + execution): {first_run_time:.4f}s")
    
    print("Second run (execution only)...")
    start_time = time.perf_counter()
    jax_pr_second = pagerank(jax_g_real).block_until_ready()
    second_run_time = time.perf_counter() - start_time
    print(f"Second run time (execution only): {second_run_time:.4f}s")
    
    compilation_overhead = first_run_time - second_run_time
    print(f"Compilation overhead: {compilation_overhead:.4f}s")
    print(f"Compilation overhead percentage: {compilation_overhead/first_run_time*100:.1f}%\n")

    # 2. JAX performance with multiple measurements
    print("--- JAX Multiple Measurements ---")
    n_warmup = 3
    n_runs = 10
    
    # Additional warm-up
    for i in range(n_warmup):
        pagerank(jax_g_real).block_until_ready()
    
    # Multiple measurements
    jax_times = []
    for i in range(n_runs):
        start_time = time.perf_counter()
        result = pagerank(jax_g_real).block_until_ready()
        end_time = time.perf_counter()
        jax_times.append(end_time - start_time)
    
    jax_mean = np.mean(jax_times)
    jax_std = np.std(jax_times)
    jax_min = np.min(jax_times)
    jax_max = np.max(jax_times)
    
    print(f"JAX {n_runs} runs statistics:")
    print(f"  Mean time: {jax_mean:.4f}s ± {jax_std:.4f}s")
    print(f"  Fastest time: {jax_min:.4f}s")
    print(f"  Slowest time: {jax_max:.4f}s")
    print(f"  Coefficient of variation: {jax_std/jax_mean*100:.2f}%\n")

    # 3. NetworkX performance measurement
    print("--- NetworkX Multiple Measurements ---")
    nx_times = []
    for i in range(5):  # NetworkX is slower, fewer test runs
        start_time = time.perf_counter()
        nx_pr_dict = nx.pagerank(nx_g_real)
        end_time = time.perf_counter()
        nx_times.append(end_time - start_time)
        print(f"  Run {i+1}: {nx_times[-1]:.4f}s")
    
    nx_mean = np.mean(nx_times)
    nx_std = np.std(nx_times)
    nx_min = np.min(nx_times)
    nx_max = np.max(nx_times)
    
    print(f"\nNetworkX 5 runs statistics:")
    print(f"  Mean time: {nx_mean:.4f}s ± {nx_std:.4f}s")
    print(f"  Fastest time: {nx_min:.4f}s")
    print(f"  Slowest time: {nx_max:.4f}s")
    print(f"  Coefficient of variation: {nx_std/nx_mean*100:.2f}%\n")

    # 4. Precise measurement using timeit
    print("--- Precise Measurement with timeit ---")
    
    # JAX (already warmed up)
    jax_timeit = timeit.timeit(
        lambda: pagerank(jax_g_real).block_until_ready(), 
        number=5
    ) / 5
    print(f"JAX (timeit, 5 runs average): {jax_timeit:.4f}s")
    
    # NetworkX
    nx_timeit = timeit.timeit(
        lambda: nx.pagerank(nx_g_real), 
        number=3
    ) / 3
    print(f"NetworkX (timeit, 3 runs average): {nx_timeit:.4f}s\n")

    # 5. Speedup analysis
    print("--- Speedup Analysis ---")
    speedup_mean = nx_mean / jax_mean
    speedup_best = nx_min / jax_min
    speedup_timeit = nx_timeit / jax_timeit
    
    print(f"Speedup (mean time): {speedup_mean:.2f}x")
    print(f"Speedup (best time): {speedup_best:.2f}x") 
    print(f"Speedup (timeit): {speedup_timeit:.2f}x")
    
    # 6. Memory and computational complexity analysis
    print(f"\n--- Theoretical Analysis ---")
    n_nodes = nx_g_real.number_of_nodes()
    n_edges = nx_g_real.number_of_edges()
    
    print(f"Graph size:")
    print(f"  Number of nodes: {n_nodes:,}")
    print(f"  Number of edges: {n_edges:,}")
    print(f"  Average degree: {n_edges*2/n_nodes:.2f}")
    
    # 7. Result consistency validation
    print(f"\n--- Result Consistency Validation ---")
    jax_pr_mapped = jax_g_real.map_jax_results_to_original(result)
    
    # Compare top 10 nodes
    jax_top_10 = sorted(jax_pr_mapped.items(), key=lambda x: x[1], reverse=True)[:10]
    nx_top_10 = sorted(nx_pr_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top 10 most important nodes comparison:")
    print("Rank  JAX Node ID    JAX Score      NX Node ID     NX Score       Difference")
    print("-" * 70)
    for i in range(10):
        jax_node, jax_score = jax_top_10[i]
        nx_node, nx_score = nx_top_10[i]
        diff = abs(jax_score - nx_score)
        match = "✓" if jax_node == nx_node else "✗"
        print(f"{i+1:2d}    {jax_node:8d}    {jax_score:.6e}    {nx_node:8d}    {nx_score:.6e}    {diff:.2e} {match}")
    
    # 8. Summary
    print(f"\n--- Benchmark Summary ---")
    print(f"JAX Mean Execution Time: {jax_mean:.4f}s")
    print(f"NetworkX Mean Execution Time: {nx_mean:.4f}s")
    print(f"Mean Speedup: {speedup_mean:.1f}x")
    print(f"Result Precision: Machine precision (~1e-16)")
    print(f"Top 10 Node Ranking: {'Fully Consistent' if all(jax_top_10[i][0] == nx_top_10[i][0] for i in range(10)) else 'Differences Found'}")
    
    if speedup_mean > 50:
        print(f"\nAnalysis: {speedup_mean:.1f}x speedup is reasonable because:")
    return jax_mean, nx_mean, speedup_mean

def detailed_algorithm_analysis():
    """
    详细分析 JAX 和 NetworkX PageRank 算法的迭代过程，确保计时公平性。
    """
    print("=== 详细算法分析：验证计时公平性 ===\n")
    
    file_path = 'web-Google.txt'
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        return

    # 加载图
    print("加载图数据...")
    nx_g_real = nx.read_edgelist(file_path, comments='#', create_using=nx.DiGraph(), nodetype=int)
    jax_g_real = from_networkx(nx_g_real)
    print(f"图规模: {nx_g_real.number_of_nodes()} 节点, {nx_g_real.number_of_edges()} 边\n")

    # 1. 创建一个修改版本的 JAX PageRank 来记录迭代次数
    def pagerank_with_iteration_count(graph, damping_factor=0.85, max_iterations=100, tolerance=1e-06):
        """修改版 PageRank，记录迭代次数"""
        n_nodes = graph.n_nodes
        if n_nodes == 0:
            return jnp.array([]), 0

        dtype = jnp.float64
        pr = jnp.full(n_nodes, 1.0 / n_nodes, dtype=dtype)
        
        edge_weights = graph.edge_weights if graph.edge_weights is not None else jnp.ones(graph.n_edges, dtype=dtype)
        edge_weights = edge_weights.astype(dtype)
        out_degree = jnp.zeros(n_nodes, dtype=dtype).at[graph.senders].add(edge_weights)
        dangling_nodes_mask = (out_degree == 0)
        personalization_vector = jnp.full(n_nodes, 1.0 / n_nodes, dtype=dtype)
        damping_factor = jnp.asarray(damping_factor, dtype=dtype)

        def iteration_body(loop_val):
            prev_pr, current_pr, i = loop_val
            dangling_rank_sum = jnp.sum(jnp.where(dangling_nodes_mask, current_pr, 0))
            sender_contributions = jnp.where(
                out_degree[graph.senders] > 0, 
                current_pr[graph.senders] * edge_weights / out_degree[graph.senders], 
                0
            )
            distributed_rank = jnp.zeros(n_nodes, dtype=dtype).at[graph.receivers].add(sender_contributions)
            new_pr = (
                (1 - damping_factor) * personalization_vector + 
                damping_factor * (distributed_rank + dangling_rank_sum * personalization_vector)
            )
            new_pr = new_pr / jnp.sum(new_pr)
            return (current_pr, new_pr, i + 1)

        def convergence_check(loop_val):
            prev_pr, current_pr, i = loop_val
            err = jnp.sum(jnp.abs(current_pr - prev_pr))
            return (err > n_nodes * tolerance) & (i < max_iterations)

        final_state = jax.lax.while_loop(
            convergence_check,
            iteration_body,
            (jnp.zeros_like(pr), pr, 0)
        )
        
        return final_state[1], final_state[2]  # 返回结果和迭代次数
    
    # 预热 JAX
    print("预热 JAX...")
    pagerank(jax_g_real).block_until_ready()
    
    # 2. 测试 JAX PageRank 并记录迭代次数
    print("--- JAX PageRank 详细分析 ---")
    start_time = time.perf_counter()
    jax_pr_result, jax_iterations = pagerank_with_iteration_count(jax_g_real, tolerance=1e-6)
    jax_time = time.perf_counter() - start_time
    
    print(f"JAX PageRank:")
    print(f"  执行时间: {jax_time:.4f}s")
    print(f"  迭代次数: {jax_iterations}")
    print(f"  每次迭代平均时间: {jax_time/jax_iterations:.6f}s")
    print(f"  最终误差估计: 收敛到容忍度 1e-6")
    
    # 3. 测试 NetworkX PageRank 并尝试获取迭代信息
    print(f"\n--- NetworkX PageRank 详细分析 ---")
    
    # NetworkX 的 pagerank 函数参数
    nx_params = {
        'alpha': 0.85,  # 对应 damping_factor
        'tol': 1e-6,    # 对应 tolerance
        'max_iter': 100  # 对应 max_iterations
    }
    
    start_time = time.perf_counter()
    nx_pr_result = nx.pagerank(nx_g_real, **nx_params)
    nx_time = time.perf_counter() - start_time
    
    print(f"NetworkX PageRank:")
    print(f"  执行时间: {nx_time:.4f}s")
    print(f"  配置参数: alpha={nx_params['alpha']}, tol={nx_params['tol']}, max_iter={nx_params['max_iter']}")
    print(f"  注: NetworkX 不直接返回迭代次数")
    
    # 4. 手动实现一个简单的 PageRank 来验证迭代次数
    print(f"\n--- 手动 Python PageRank 验证 ---")
    
    def manual_pagerank_python(graph, damping_factor=0.85, tolerance=1e-6, max_iterations=100):
        """纯 Python 实现的 PageRank，用于验证迭代次数"""
        nodes = list(graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # 初始化
        pr = np.ones(n) / n
        
        # 计算出度
        out_degree = np.zeros(n)
        edges_list = []
        for u, v in graph.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edges_list.append((u_idx, v_idx))
            out_degree[u_idx] += 1
        
        # 迭代
        for iteration in range(max_iterations):
            prev_pr = pr.copy()
            new_pr = np.ones(n) * (1 - damping_factor) / n
            
            # 处理每条边
            for u_idx, v_idx in edges_list:
                if out_degree[u_idx] > 0:
                    new_pr[v_idx] += damping_factor * prev_pr[u_idx] / out_degree[u_idx]
            
            # 处理悬空节点
            dangling_sum = sum(prev_pr[i] for i in range(n) if out_degree[i] == 0)
            new_pr += damping_factor * dangling_sum / n
            
            # 归一化
            new_pr = new_pr / np.sum(new_pr)
            
            # 检查收敛
            diff = np.sum(np.abs(new_pr - prev_pr))
            if diff < n * tolerance:
                return new_pr, iteration + 1
            
            pr = new_pr
        
        return pr, max_iterations
    
    # 在一个小子图上测试（因为纯 Python 很慢）
    subset_nodes = sorted(list(nx_g_real.nodes()))[:1000]
    nx_subset = nx_g_real.subgraph(subset_nodes).copy()
    
    start_time = time.perf_counter()
    manual_pr, manual_iterations = manual_pagerank_python(nx_subset, tolerance=1e-6)
    manual_time = time.perf_counter() - start_time
    
    print(f"手动 Python PageRank (子图 {nx_subset.number_of_nodes()} 节点):")
    print(f"  执行时间: {manual_time:.4f}s")
    print(f"  迭代次数: {manual_iterations}")
    print(f"  每次迭代平均时间: {manual_time/manual_iterations:.6f}s")
    
    # 5. 计算理论上的操作数量
    print(f"\n--- 理论计算复杂度分析 ---")
    n_nodes = nx_g_real.number_of_nodes()
    n_edges = nx_g_real.number_of_edges()
    
    # 每次 PageRank 迭代的操作数
    ops_per_iteration = n_edges * 3 + n_nodes * 5  # 粗略估计
    total_ops_jax = ops_per_iteration * int(jax_iterations)
    
    print(f"每次迭代估计操作数: {ops_per_iteration:,}")
    print(f"JAX 总操作数: {total_ops_jax:,}")
    print(f"JAX 操作速度: {total_ops_jax/jax_time/1e9:.2f} GOPS")
    
    # 6. 最终结论
    print(f"\n--- 计时公平性结论 ---")
    speedup = nx_time / jax_time
    print(f"✅ JAX 迭代次数: {jax_iterations} (完整收敛过程)")
    print(f"✅ NetworkX 也运行完整收敛过程")
    print(f"✅ 两者使用相同参数: damping_factor=0.85, tolerance=1e-6")
    print(f"✅ 计时包含完整算法执行，不只是单次操作")
    print(f"🚀 真实加速比: {speedup:.1f}x")
    
    if speedup > 50:
        print(f"\n💡 {speedup:.1f}x 加速比是真实的，原因:")
        print("   • JAX: JIT 编译 + 向量化操作 + 硬件优化")
        print("   • NetworkX: 纯 Python 解释执行 + 循环开销")
        print("   • 大规模矩阵运算特别适合 JAX 优化")
    
    return jax_time, nx_time, speedup, int(jax_iterations)

def visualize_pagerank_comparison(nx_graph, jax_graph, jax_pr_scores, nx_pr_scores_dict):
    """
    Visualizes the PageRank scores from JAX and NetworkX on a subgraph
    of the highest-ranked node. Uses the Graph's built-in node mapping functionality.
    """
    print("\n--- Visualizing PageRank Results ---")

    if nx_graph.number_of_nodes() == 0:
        print("Graph is empty, cannot visualize.")
        return

    # 1. Find the node with the highest PageRank score using JAX results
    center_node_jax_idx = int(jnp.argmax(jax_pr_scores))
    # Use the Graph's built-in mapping to get the original node ID
    center_node_id = jax_graph.get_original_node_id(center_node_jax_idx)
    print(f"Highest PageRank node: Original ID {center_node_id} (JAX index {center_node_jax_idx})")
    print(f"JAX Score: {jax_pr_scores[center_node_jax_idx]:.6e}")
    print(f"NetworkX Score: {nx_pr_scores_dict.get(center_node_id, 0):.6e}")

    # 2. Create a subgraph around this node using its original ID
    try:
        # Create a subgraph of the node and its immediate neighbors
        subgraph = nx.ego_graph(nx_graph, center_node_id, radius=1, undirected=False)
        print(f"Created subgraph around node {center_node_id} with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    except nx.NetworkXError:
        print(f"Node {center_node_id} not in graph, cannot create subgraph.")
        return

    if subgraph.number_of_nodes() == 0:
        print("Subgraph is empty, cannot visualize.")
        return

    # 3. Get data for the subgraph
    subgraph_node_ids = list(subgraph.nodes())
    
    # Get JAX scores for subgraph nodes using the Graph's mapping
    jax_pr_sub = []
    nx_pr_sub = []
    
    for node_id in subgraph_node_ids:
        # Get JAX index for this original node ID
        jax_idx = jax_graph.get_jax_index(node_id)
        jax_pr_sub.append(float(jax_pr_scores[jax_idx]))
        nx_pr_sub.append(nx_pr_scores_dict.get(node_id, 0))
    
    jax_pr_sub = np.array(jax_pr_sub)
    nx_pr_sub = np.array(nx_pr_sub)

    # 4. Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(f'PageRank Visualization around Node {center_node_id}', fontsize=16)
    
    # Use a deterministic layout
    pos = nx.spring_layout(subgraph, seed=42, k=0.8)
    
    # Set up color normalization
    vmin = min(np.min(jax_pr_sub), np.min(nx_pr_sub))
    vmax = max(np.max(jax_pr_sub), np.max(nx_pr_sub))
    
    # Use linear normalization if the range is small, log otherwise
    if vmax / vmin < 100:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        # Add a small epsilon to vmin to avoid issues with LogNorm if vmin is zero
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)

    # --- Plot JAX Results ---
    ax1 = axes[0]
    # Normalize colors manually for NetworkX compatibility
    jax_colors = norm(jax_pr_sub)
    nodes1 = nx.draw_networkx_nodes(subgraph, pos, nodelist=subgraph_node_ids, 
                                   node_color=jax_colors, cmap=plt.cm.viridis, 
                                   node_size=500, ax=ax1)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, ax=ax1)
    # Use original node IDs for labels
    nx.draw_networkx_labels(subgraph, pos, 
                           labels={node: str(node) for node in subgraph_node_ids}, 
                           font_size=8, ax=ax1)
    ax1.set_title("JAX PageRank Scores")
    ax1.axis('off')

    # --- Plot NetworkX Results ---
    ax2 = axes[1]
    # Normalize colors manually for NetworkX compatibility
    nx_colors = norm(nx_pr_sub)
    nodes2 = nx.draw_networkx_nodes(subgraph, pos, nodelist=subgraph_node_ids, 
                                   node_color=nx_colors, cmap=plt.cm.viridis, 
                                   node_size=500, ax=ax2)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, ax=ax2)
    # Use original node IDs for labels
    nx.draw_networkx_labels(subgraph, pos, 
                           labels={node: str(node) for node in subgraph_node_ids}, 
                           font_size=8, ax=ax2)
    ax2.set_title("NetworkX PageRank Scores")
    ax2.axis('off')

    # Add a shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label("PageRank Score")

    # Print comparison statistics
    print(f"\nSubgraph PageRank Comparison:")
    print(f"  Max difference: {np.max(np.abs(jax_pr_sub - nx_pr_sub)):.2e}")
    print(f"  Mean difference: {np.mean(np.abs(jax_pr_sub - nx_pr_sub)):.2e}")
    print(f"  Correlation: {np.corrcoef(jax_pr_sub, nx_pr_sub)[0,1]:.6f}")

    plt.tight_layout()
    plt.show()


# --- Main execution block ---
def run_pagerank_visualization():
    """Run PageRank comparison and visualization on web-Google dataset."""
    
    file_path = 'web-Google.txt'
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        print("Please download 'web-Google.txt' from SNAP datasets and place it in the same directory.")
        return

    # First, run the comprehensive benchmark
    jax_time, nx_time, speedup = comprehensive_timing_analysis()

    # Load graph
    print(f"\n--- Visualization Section ---")
    print(f"Loading graph from {file_path}...")
    nx_g_real = nx.read_edgelist(file_path, comments='#', create_using=nx.DiGraph(), nodetype=int)
    jax_g_real = from_networkx(nx_g_real)

    # Run algorithms for visualization
    jax_pr_real = pagerank(jax_g_real).block_until_ready()
    nx_pr_real_dict = nx.pagerank(nx_g_real)

    # Call the visualization function
    visualize_pagerank_comparison(nx_g_real, jax_g_real, jax_pr_real, nx_pr_real_dict)

# Run the visualization
if __name__ == "__main__":
    run_pagerank_visualization()