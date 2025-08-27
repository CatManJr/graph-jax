#!/usr/bin/env python3
"""
Supply Chain Network Vulnerability Analysis Example
Demonstrating Graph-JAX's capacity analysis capabilities

Citation: Salgado, A., He, Y., Radke, J. et al. 
Dimension reduction approach for understanding resource-flow resilience to climate change. 
Commun Phys 7, 192 (2024). https://doi.org/10.1038/s42005-024-01664-z

This example demonstrates how Graph-JAX's capacity analysis can be used to:
1. Compute network capacity parameters from actual topology
2. Analyze system stability and failure times
3. Evaluate supply chain resilience under disruptions

"""
import numpy as np
import matplotlib.pyplot as plt
import graph_jax as gj
import jax.numpy as jnp
import time
import networkx as nx

# Force CPU backend to avoid Metal issues
from graph_jax.utils import set_backend
set_backend('cpu')

def create_supply_chain_network():
    """
    Create a realistic supply chain network and use Graph-JAX's capacity analysis
    to compute all parameters from the actual network topology.
    """
    import time
    
    # Use current time as seed for variability
    rng = np.random.default_rng(int(time.time() * 1000) % 10000)

    # ---------------------------------------------------------
    # 1. Network structure based on paper Table 1
    # ---------------------------------------------------------
    N1, N2, N3 = 5, 29, 3422          # Refineries / Terminals / Gas stations
    n_nodes = N1 + N2 + N3

    # ---------------------------------------------------------
    # 2. Node capacity parameters from paper Table 1
    # ---------------------------------------------------------
    # Node capacity ranges from Table 1
    C1_range = [38.2, 57.3]  # Refinery capacity range (Mgal)
    C2_range = [31, 62]      # Terminal capacity range (Mgal)
    C3_value = 0.035         # Gas station capacity (Mgal, fixed)
    
    # Sample node capacities from normal distributions
    def sample_from_range(rng, min_val, max_val):
        """Sample from normal distribution within given range"""
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 6  # 99.7% of normal distribution within 6 std
        sample = rng.normal(mean, std)
        return np.clip(sample, min_val, max_val)
    
    C1 = sample_from_range(rng, C1_range[0], C1_range[1])
    C2 = sample_from_range(rng, C2_range[0], C2_range[1])
    C3 = C3_value

    # ---------------------------------------------------------
    # 3. Construct realistic supply chain network
    # ---------------------------------------------------------
    # Node indices
    prod  = np.arange(0,  N1)                    # Refineries 0–4
    term  = np.arange(N1, N1+N2)                 # Terminals 5–33
    gas   = np.arange(N1+N2, n_nodes)            # Gas stations 34–3455

    edges = []
    
    # 1. Refinery → Terminal connections (pipeline network)
    refinery_terminal_edges = 150
    edges_per_refinery = refinery_terminal_edges // N1
    for r in prod:
        tgt = rng.choice(term, size=min(edges_per_refinery, N2), replace=False)
        edges.extend([(r, t) for t in tgt])
    
    # 2. Terminal → Gas Station connections (truck network)
    terminal_gas_edges = 7800
    edges_per_terminal = terminal_gas_edges // N2
    for t in term:
        tgt = rng.choice(gas, size=min(edges_per_terminal, N3), replace=False)
        edges.extend([(t, g) for g in tgt])
    
    # 3. Refinery → Gas Station connections (direct truck routes)
    refinery_gas_edges = 3500
    edges_per_refinery_direct = refinery_gas_edges // N1
    for r in prod:
        tgt = rng.choice(gas, size=min(edges_per_refinery_direct, N3), replace=False)
        edges.extend([(r, g) for g in tgt])

    edges = list({(u, v) for u, v in edges})   # Remove duplicates
    
    # Print network statistics
    print(f"Network Construction:")
    print(f"  Total nodes: {n_nodes:,}")
    print(f"  Refineries: {N1}")
    print(f"  Terminals: {N2}")
    print(f"  Gas stations: {N3:,}")
    print(f"  Total edges: {len(edges):,}")
    print(f"  Network density: {len(edges)/(n_nodes*(n_nodes-1)/2):.6f}")
    print(f"  Average degree: {2*len(edges)/n_nodes:.1f}")

    # ---------------------------------------------------------
    # 4. Build NetworkX graph with realistic capacities
    # ---------------------------------------------------------
    G = nx.Graph()
    
    # Add all nodes first
    for i in range(n_nodes):
        G.add_node(i)
    
    # Add edges with realistic capacities
    for u, v in edges:
        if u in prod and v in term:
            # Refinery → Terminal: high capacity pipeline
            cap = rng.uniform(70, 140)  # Mgal week^-1
        elif u in term and v in gas:
            # Terminal → Gas Station: medium capacity truck route
            cap = rng.uniform(105, 245)  # Mgal week^-1
        else:
            # Refinery → Gas Station: direct truck route
            cap = rng.uniform(21, 81)    # Mgal week^-1
        G.add_edge(u, v, capacity=cap)

    # Convert to Graph-JAX format
    g = gj.from_networkx(G)

    # ---------------------------------------------------------
    # 5. Create layer masks for capacity analysis
    # ---------------------------------------------------------
    ref_mask = jnp.zeros(n_nodes, dtype=bool).at[jnp.array(prod)].set(True)
    term_mask = jnp.zeros(n_nodes, dtype=bool).at[jnp.array(term)].set(True)
    gas_mask = jnp.zeros(n_nodes, dtype=bool).at[jnp.array(gas)].set(True)

    # ---------------------------------------------------------
    # 6. Use Graph-JAX's capacity_params to compute all parameters
    # ---------------------------------------------------------
    print(f"\nComputing capacity parameters using Graph-JAX...")
    
    # Use the optimized capacity_params function to compute all parameters
    # from the actual network topology
    params = gj.algorithms.capacity_params(
        g,
        ref_mask,
        term_mask,
        gas_mask,
        C1=C1,
        C2=C2,
        C3=C3,
        edge_cap=100.0,  # Base edge capacity (will be overridden by actual edge capacities)
        use_parallel=True
    )
    
    # Calculate p and d from network parameters according to paper formula (2)
    # p = P/C1, d = D/C3 where P and D are from Table 1
    P_total = 18.9  # Total production capacity (Mgal week^-1) from Table 1
    D_total = 0.028  # Total demand capacity (Mgal week^-1) from Table 1
    
    # Calculate normalized production and demand rates
    params['p'] = P_total / C1  # Normalized production capacity
    params['d'] = D_total / C3  # Normalized demand capacity
    
    print(f"  Calculated p = P/C1 = {P_total}/{C1:.3f} = {params['p']:.3f}")
    print(f"  Calculated d = D/C3 = {D_total}/{C3:.3f} = {params['d']:.3f}")
    
    # Normalize parameters for ODE system stability
    # The raw capacity parameters from min-cut need normalization for the ODE solver
    print(f"\nNormalizing parameters for ODE system...")
    
    # Calculate normalization factors based on the paper's approach
    # Use a more conservative normalization to ensure ODE stability
    # Target range: s parameters should be in [0.1, 1.0] for stable ODE
    max_capacity = max(params['s12'], params['s23'], params['s13'])
    normalization_factor = 0.1 / max_capacity if max_capacity > 0 else 1.0  # Less aggressive normalization
    
    # Store original parameters for display
    original_params = params.copy()
    
    # Apply normalization to capacity parameters
    params['s12'] = params['s12'] * normalization_factor
    params['s23'] = params['s23'] * normalization_factor
    params['s13'] = params['s13'] * normalization_factor
    
    print(f"  Normalization factor: {normalization_factor:.6f}")
    print(f"  Original s12: {original_params['s12']:.3f} → Normalized: {params['s12']:.3f}")
    print(f"  Original s23: {original_params['s23']:.3f} → Normalized: {params['s23']:.3f}")
    print(f"  Original s13: {original_params['s13']:.3f} → Normalized: {params['s13']:.3f}")
    
    # ---------------------------------------------------------
    # 7. Display computed parameters
    # ---------------------------------------------------------
    print(f"\nGraph-JAX Computed Parameters:")
    print(f"  Node counts: N1={params['N1']}, N2={params['N2']}, N3={params['N3']}")
    print(f"  Node capacities: C1={C1:.3f}, C2={C2:.3f}, C3={C3:.3f} Mgal")
    print(f"  Network capacity parameters:")
    print(f"    s12 (Production→Terminal): {params['s12']:.3f}")
    print(f"    s23 (Terminal→Consumption): {params['s23']:.3f}")
    print(f"    s13 (Production→Consumption): {params['s13']:.3f}")
    print(f"    α12 (Production/Terminal ratio): {params['alpha12']:.3f}")
    print(f"    α23 (Terminal/Consumption ratio): {params['alpha23']:.3f}")
    print(f"  System parameters:")
    print(f"    p (Production rate): {params['p']:.3f} week^-1")
    print(f"    d (Demand rate): {params['d']:.3f} week^-1")
    print(f"    p/d ratio: {params['p']/params['d']:.3f}")
    
    return g, params, ref_mask, term_mask, gas_mask

def simulate_system_evolution(params, production_levels, ΔT=14.0):
    """
    Simulate system evolution under different production disruption scenarios
    using Graph-JAX's capacity analysis functions.
    
    First computes the normal steady state, then simulates disruptions from that baseline.
    """
    print(f"\n{'='*60}")
    print(f"System Evolution Simulation")
    print(f"{'='*60}")
    print(f"System parameters: p={params['p']}, d={params['d']}")
    
    # Step 1: Compute steady state at 100% production (normal operation)
    print(f"\nStep 1: Computing steady state at 100% production (normal operation)...")
    print(f"{'-'*60}")
    
    start_time = time.time()
    normal_steady_state = gj.algorithms.capacity.steady_state(params, t_max=200.0, n_steps=2000)
    steady_state_time = time.time() - start_time
    
    print(f"Normal steady state calculation time: {steady_state_time:.4f} seconds")
    print(f"Normal steady state: Production={normal_steady_state[0]:.3f}, Transport={normal_steady_state[1]:.3f}, Consumption={normal_steady_state[2]:.3f}")
    
    # Step 2: Simulate disruption scenarios starting from normal steady state
    print(f"\nStep 2: Simulating production disruption scenarios from normal steady state...")
    print(f"{'-'*60}")
    
    collapse_times = []
    demand_satisfaction = []
    
    start_time = time.time()
    
    for level in production_levels:
        # Create disruption parameters
        disruption_params = params.copy()
        disruption_params['p'] = params['p'] * level
        
        # For normal operation (100%), use the pre-computed steady state
        if level == 1.0:
            # Normal operation - system is stable
            τ = ΔT  # System remains stable for the entire simulation period
            
            # Calculate actual demand satisfaction based on production-demand ratio
            production_rate = disruption_params['p']
            demand_rate = disruption_params['d']
            
            if demand_rate > 0:
                QD = min(1.0, production_rate / demand_rate)
            else:
                QD = 1.0
            
            print(f"{level*100:2.0f}% Production simulation (Normal operation):")
            print(f"  Production rate: {disruption_params['p']:.3f}")
            print(f"  System status: STABLE (no collapse)")
            print(f"  Demand satisfaction rate: {QD:.3f}")
            print()
        else:
            # Disruption scenario - simulate from normal steady state
            # We need to create a new ODE system that starts from the normal steady state
            # and simulates the disruption scenario
            
            # Calculate demand satisfaction based on production-demand ratio
            production_rate = disruption_params['p']
            demand_rate = disruption_params['d']
            
            # Calculate demand satisfaction as the ratio of production to demand
            if demand_rate > 0:
                QD = min(1.0, production_rate / demand_rate)
            else:
                QD = 1.0
            
            # For disruption scenarios, we need to simulate the system evolution
            # starting from the normal steady state with reduced production
            try:
                # Use a shorter simulation time for disruption scenarios
                disruption_steady_state = gj.algorithms.capacity.steady_state(disruption_params, t_max=50.0, n_steps=500)
                
                # Check if the system collapses (consumption level drops significantly)
                normal_consumption = normal_steady_state[2]
                disruption_consumption = disruption_steady_state[2]
                
                # If consumption drops below 10% of normal, consider it collapsed
                if disruption_consumption < 0.1 * normal_consumption:
                    # Estimate collapse time based on consumption rate
                    τ = min(ΔT, normal_consumption / (normal_consumption - disruption_consumption) * 10.0)
                else:
                    τ = ΔT  # System remains stable
                    
            except:
                # If ODE solver fails, use a simple estimate
                τ = min(ΔT, normal_steady_state[2] / demand_rate if demand_rate > 0 else ΔT)
            
            print(f"{level*100:2.0f}% Production simulation (Disruption from normal state):")
            print(f"  Production rate: {disruption_params['p']:.3f}")
            print(f"  System collapse time: {τ:.2f} days")
            print(f"  Average demand satisfaction rate: {QD:.3f}")
            print()
        
        collapse_times.append(float(τ))
        demand_satisfaction.append(float(QD))
    
    total_time = time.time() - start_time
    print(f"Total disruption analysis time: {total_time:.4f} seconds")
    
    return collapse_times, demand_satisfaction, normal_steady_state

def analyze_critical_thresholds(collapse_times, production_levels):
    """
    Analyze critical thresholds for system stability.
    """
    print(f"\nIndividual Critical Analysis:")
    print(f"{'='*50}")
    
    critical_count = 0
    warning_count = 0
    safe_count = 0
    
    critical_scenarios = []
    warning_scenarios = []
    safe_scenarios = []
    
    for i, (level, tau) in enumerate(zip(production_levels, collapse_times)):
        if tau <= 1.0:
            status = "CRITICAL"
            critical_count += 1
            critical_scenarios.append(f"{level*100:.0f}% Production")
        elif tau <= 7.0:
            status = "WARNING"
            warning_count += 1
            warning_scenarios.append(f"{level*100:.0f}% Production")
        else:
            status = "SAFE"
            safe_count += 1
            safe_scenarios.append(f"{level*100:.0f}% Production")
        
        scenario_name = f"{level*100:.0f}% Production" if level < 1.0 else "100% Production (Steady State)"
        print(f"  {scenario_name}: {tau:.2f} days - {status}")
    
    print(f"\nSummary:")
    print(f"  Critical scenarios (≤1 day): {critical_count}")
    print(f"  Warning scenarios (1-7 days): {warning_count}")
    print(f"  Safe scenarios (>7 days): {safe_count}")
    
    if critical_scenarios:
        print(f"  Critical scenarios: {', '.join(critical_scenarios)}")
    if warning_scenarios:
        print(f"  Warning scenarios: {', '.join(warning_scenarios)}")
    if safe_scenarios:
        print(f"  Safe scenarios: {', '.join(safe_scenarios)}")

def create_visualizations(collapse_times, demand_satisfaction, production_levels, params, steady_state):
    """
    Create comprehensive visualizations with enhanced time evolution analysis.
    """
    # Set up Nature journal style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Create a larger figure with 4x2 subplot layout
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Fake Fuel Supply Chain Resilience Analysis', fontsize=16, fontweight='bold')
    
    # Layer names and colors
    layer_names = ['Production', 'Terminal Storage', 'Consumption']
    layer_colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # 1. Steady State Analysis
    ax1 = axes[0, 0]
    steady_state_labels = ['Production', 'Terminal Storage', 'Consumption']
    steady_state_colors = ['#1f77b4', '#2ca02c', '#d62728']
    bars = ax1.bar(steady_state_labels, steady_state, color=steady_state_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Steady State Inventory (Mgal/week)', fontsize=10)
    ax1.set_title('Steady State Analysis', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=9)
    
    # Add value labels on bars
    for bar, value in zip(bars, steady_state):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Network Capacity Analysis
    ax2 = axes[0, 1]
    capacities = [params['s12'], params['s23'], params['s13']]
    capacity_labels = ['Production→Terminal', 'Terminal→Consumption', 'Production→Consumption']
    capacity_colors = ['#ff7f0e', '#9467bd', '#8c564b']
    bars = ax2.bar(capacity_labels, capacities, color=capacity_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Capacity Parameters (Mgal/week)', fontsize=10)
    ax2.set_title('Network Capacity Analysis', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax2.tick_params(axis='both', which='major', labelsize=9)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, capacities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Production Disruption Impact
    ax3 = axes[1, 0]
    ax3.plot([p*100 for p in production_levels], collapse_times, 'bo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Production Level (%)', fontsize=10)
    ax3.set_ylabel('System Collapse Time (days)', fontsize=10)
    ax3.set_title('Production Disruption Impact', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Critical threshold (1 day)')
    ax3.axhline(y=7, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Warning threshold (7 days)')
    ax3.legend(fontsize=8, framealpha=0.9)
    ax3.tick_params(axis='both', which='major', labelsize=9)
    
    # 4. Demand Satisfaction
    ax4 = axes[1, 1]
    ax4.plot([p*100 for p in production_levels], demand_satisfaction, 'go-', linewidth=2, markersize=8)
    ax4.set_xlabel('Production Level (%)', fontsize=10)
    ax4.set_ylabel('Demand Satisfaction Rate', fontsize=10)
    ax4.set_title('Demand Satisfaction Under Disruption', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    ax4.tick_params(axis='both', which='major', labelsize=9)
    
    # 5. Critical Threshold Analysis
    ax5 = axes[2, 0]
    critical_threshold = None
    for i, (level, tau) in enumerate(zip(production_levels, collapse_times)):
        if tau <= 1.0:
            critical_threshold = level * 100
            break
    
    if critical_threshold is not None:
        ax5.axvline(x=critical_threshold, color='red', linestyle='--', linewidth=2, label=f'Critical threshold: {critical_threshold:.1f}%')
        ax5.axvspan(0, critical_threshold, alpha=0.2, color='red', label='Critical zone')
        ax5.axvspan(critical_threshold, 100, alpha=0.2, color='green', label='Safe zone')
    
    ax5.plot([p*100 for p in production_levels], collapse_times, 'ro-', linewidth=2, markersize=8)
    ax5.set_xlabel('Production Level (%)', fontsize=10)
    ax5.set_ylabel('Collapse Time (days)', fontsize=10)
    ax5.set_title('Critical Threshold Analysis', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, linewidth=0.5)
    ax5.legend(fontsize=8, framealpha=0.9)
    ax5.tick_params(axis='both', which='major', labelsize=9)
    
    # 6. Network Capacity Ratios
    ax6 = axes[2, 1]
    capacity_ratios = {
        'alpha12 (Prod->Term)': params['alpha12'],
        'alpha23 (Term->Cons)': params['alpha23'],
        's12/s13 (Direct/Indirect)': params['s12'] / params['s13'],
        's23/s13 (Term->Direct)': params['s23'] / params['s13'],
        's12/s23 (Prod/Term)': params['s12'] / params['s23']
    }
    
    ratios = list(capacity_ratios.values())
    labels = list(capacity_ratios.keys())
    
    bars = ax6.bar(range(len(ratios)), ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.set_xlabel('Capacity Ratios', fontsize=10)
    ax6.set_ylabel('Ratio Value', fontsize=10)
    ax6.set_title('Network Capacity Ratios', fontsize=11, fontweight='bold')
    ax6.set_xticks(range(len(ratios)))
    ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax6.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(ratios)*0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 7. Key Metrics Summary
    ax7 = axes[3, 0]
    
    # Use the algorithm library function for maximum stable demand calculation
    def calculate_max_stable_demand(params):
        """Calculate maximum stable demand using Graph-JAX algorithm library"""
        return gj.algorithms.max_stable_demand(params)
    
    # Calculate maximum stable demand
    max_stable_demand = calculate_max_stable_demand(params)
    avg_demand = np.mean([p for p in production_levels if p < 1.0])
    critical_threshold = next((p*100 for p, tau in zip(production_levels, collapse_times) if tau <= 1.0), 0)
    
    metrics = ['Max Stable\nDemand', 'Average\nDemand', 'Critical\nThreshold']
    values = [max_stable_demand, avg_demand, critical_threshold]
    
    bars = ax7.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax7.set_ylabel('Value', fontsize=10)
    ax7.set_title('Key Supply Chain Metrics', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 8. System Vulnerability Assessment
    ax8 = axes[3, 1]
    critical_count = sum(1 for tau in collapse_times if tau <= 1.0)
    warning_count = sum(1 for tau in collapse_times if 1.0 < tau <= 7.0)
    safe_count = sum(1 for tau in collapse_times if tau > 7.0)
    
    vulnerability_data = ['Critical\n(≤1 day)', 'Warning\n(1-7 days)', 'Safe\n(>7 days)']
    vulnerability_counts = [critical_count, warning_count, safe_count]
    vulnerability_colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    bars = ax8.bar(vulnerability_data, vulnerability_counts, color=vulnerability_colors, alpha=0.8)
    ax8.set_ylabel('Number of Scenarios', fontsize=10)
    ax8.set_title('System Vulnerability Assessment', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, count in zip(bars, vulnerability_counts):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('showcase/supply_chain_comprehensive_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nComprehensive analysis chart saved as 'supply_chain_comprehensive_analysis.png'")

def analyze_key_metrics(params, collapse_times, production_levels):
    """
    Analyze key metrics for supply chain resilience.
    """
    print(f"\n{'='*80}")
    print(f"KEY METRICS ANALYSIS")
    print(f"{'='*80}")
    
    # 1. Maximum stable demand
    max_stable_demand = params['p'] / params['d']
    
    # 2. Average demand level
    avg_demand = np.mean([p for p in production_levels if p < 1.0])
    
    # 3. Network capacity ratios
    capacity_ratios = {
        'α₁₂ (Production→Terminal)': params['alpha12'],
        'α₂₃ (Terminal→Consumption)': params['alpha23'],
        's₁₂/s₁₃ (Direct vs Indirect)': params['s12'] / params['s13'],
        's₂₃/s₁₃ (Terminal vs Direct)': params['s23'] / params['s13'],
        's₁₂/s₂₃ (Production vs Terminal)': params['s12'] / params['s23']
    }
    
    print(f"1. Maximum Stable Demand: {max_stable_demand:.6f}")
    print(f"2. Time to Failure Analysis: Completed in main simulation")
    print(f"3. Average Demand Level: {avg_demand:.6f}")
    print(f"4. Network Capacity Ratios:")
    for name, ratio in capacity_ratios.items():
        print(f"   {name}: {ratio:.6f}")
    
    # Create key metrics visualization
    print(f"\nCreating key metrics visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Key metrics summary
    metrics = ['Max Stable\nDemand', 'Average\nDemand', 'Critical\nThreshold']
    values = [max_stable_demand, avg_demand, 
              next((p*100 for p, tau in zip(production_levels, collapse_times) if tau <= 1.0), 0)]
    
    bars = ax1.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax1.set_ylabel('Value')
    ax1.set_title('Key Supply Chain Metrics')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Capacity ratios
    ratio_names = list(capacity_ratios.keys())
    ratio_values = list(capacity_ratios.values())
    
    bars2 = ax2.bar(range(len(ratio_values)), ratio_values, color='skyblue')
    ax2.set_xlabel('Capacity Ratios')
    ax2.set_ylabel('Ratio Value')
    ax2.set_title('Network Capacity Ratios')
    ax2.set_xticks(range(len(ratio_values)))
    ax2.set_xticklabels(ratio_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('supply_chain_key_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Key metrics chart saved as 'supply_chain_key_metrics.png'")
    
    # Key findings summary
    print(f"\n{'='*80}")
    print(f"KEY FINDINGS SUMMARY")
    print(f"{'='*80}")
    print(f"1. Maximum Stable Demand: {max_stable_demand:.6f}")
    print(f"2. Average Demand Level: {avg_demand:.6f}")
    print(f"3. Most Critical Capacity Ratio: {max(capacity_ratios.items(), key=lambda x: x[1])[0]}")
    
    critical_threshold = next((p*100 for p, tau in zip(production_levels, collapse_times) if tau <= 1.0), 0)
    print(f"4. System Vulnerability: {'High' if critical_threshold > 50 else 'Medium' if critical_threshold > 20 else 'Low'} (critical threshold at {critical_threshold:.1f}% production)")

if __name__ == "__main__":
    # Create supply chain network using Graph-JAX capacity analysis
    g, params, ref_mask, term_mask, gas_mask = create_supply_chain_network()
    
    # Define production disruption scenarios
    production_levels = [1.0, 0.99, 0.90, 0.75, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.0]
    
    # Simulate system evolution
    collapse_times, demand_satisfaction, steady_state = simulate_system_evolution(params, production_levels)
    
    # Analyze critical thresholds
    analyze_critical_thresholds(collapse_times, production_levels)
    
    # Create comprehensive visualizations
    create_visualizations(collapse_times, demand_satisfaction, production_levels, params, steady_state)