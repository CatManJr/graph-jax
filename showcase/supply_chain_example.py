#!/usr/bin/env python3
"""
Supply Chain Network Vulnerability Analysis Example
Demonstrating the practical problems solved by Max-Flow Min-Cut + ODE solving
"""

import numpy as np
import matplotlib.pyplot as plt
import graph_jax as gj
import jax.numpy as jnp
import time

# Force CPU backend to avoid Metal issues
from graph_jax.utils import set_backend
set_backend('cpu')

def create_supply_chain_example():
    """
    Create a simplified supply chain network example using Graph-JAX
    """
    print("=" * 60)
    print("Supply Chain Network Vulnerability Analysis Example")
    print("=" * 60)
    
    # 1. Network Structure
    print("\n1. Network Structure")
    print("-" * 30)
    print("Three-layer supply chain network:")
    print("  Production Layer (y1): Oil rigs, refineries")
    print("  Transportation Layer (y2): Pipelines, tankers, trucks")
    print("  Consumption Layer (y3): Gas stations, factories, households")
    
    # 2. Create a simple network for demonstration
    print("\n2. Creating Network with Graph-JAX")
    print("-" * 30)
    
    # Create a more realistic network with different capacity values
    n_nodes = 15
    edges = [
        # Production to Transportation (limited capacity)
        (0, 5), (0, 6), (1, 5), (1, 7), (2, 6), (2, 7), 
        (3, 8), (3, 9), (4, 8), (4, 9),
        # Transportation to Consumption (moderate capacity)
        (5, 10), (5, 11), (6, 10), (6, 12), (7, 11), (7, 12),
        (8, 13), (8, 14), (9, 13), (9, 14),
        # Some direct connections (limited)
        (0, 10), (2, 12), (4, 14)
    ]
    
    # Create NetworkX graph with realistic capacities
    import networkx as nx
    G = nx.Graph()
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Add edges with realistic capacity values
    for u, v in edges:
        if u < 5 and v >= 10:  # Direct production to consumption
            capacity = np.random.uniform(0.5, 1.5)  # Limited direct capacity
        elif u < 5 and 5 <= v < 10:  # Production to transportation
            capacity = np.random.uniform(2.0, 4.0)  # Good production capacity
        elif 5 <= u < 10 and v >= 10:  # Transportation to consumption
            capacity = np.random.uniform(1.5, 2.5)  # Moderate transport capacity
        else:
            capacity = 1.0
        G.add_edge(u, v, capacity=capacity)
    
    # Convert to Graph-JAX
    g = gj.from_networkx(G)
    print(f"Created network: {g}")
    
    # 3. Network Capacity Parameters
    print("\n3. Network Capacity Parameters")
    print("-" * 30)
    
    # Create layer masks for the new network structure
    production_mask = jnp.array([True] * 5 + [False] * 10)  # Nodes 0-4: Production
    transport_mask = jnp.array([False] * 5 + [True] * 5 + [False] * 5)  # Nodes 5-9: Transportation
    consumption_mask = jnp.array([False] * 10 + [True] * 5)  # Nodes 10-14: Consumption
    
    print("Layer masks created:")
    print(f"  Production nodes: {jnp.sum(production_mask)}")
    print(f"  Transportation nodes: {jnp.sum(transport_mask)}")
    print(f"  Consumption nodes: {jnp.sum(consumption_mask)}")
    
    # 4. Calculate capacity parameters using Graph-JAX
    print("\n4. Calculating Capacity Parameters with Graph-JAX")
    print("-" * 30)
    
    start_time = time.time()
    params = gj.algorithms.capacity_params(
        g, production_mask, transport_mask, consumption_mask,
        edge_cap=2.5,  # Average edge capacity
        C1=50.0, C2=30.0, C3=20.0  # Realistic node capacities
    )
    jax_time = time.time() - start_time
    
    print(f"Graph-JAX calculation time: {jax_time:.4f} seconds")
    print(f"Capacity parameters:")
    print(f"  s12 (production→transport): {params['s12']:.6f}")
    print(f"  s23 (transport→consumption): {params['s23']:.6f}")
    print(f"  s13 (production→consumption): {params['s13']:.6f}")
    print(f"  alpha12: {params['alpha12']:.6f}")
    print(f"  alpha23: {params['alpha23']:.6f}")
    
    return params, g, production_mask, transport_mask, consumption_mask

def simulate_system_evolution(params):
    """
    Simulate system evolution using Graph-JAX ODE solver
    """
    print("\n" + "=" * 60)
    print("System Evolution Simulation")
    print("=" * 60)
    
    # Set more realistic system parameters
    system_params = params.copy()
    system_params.update({'p': 0.42, 'd': 0.79})  # More realistic production and consumption rates
    
    print(f"System parameters: p={system_params['p']}, d={system_params['d']}")
    
    # Initial state
    y0 = [1.0, 1.0, 1.0]  # Normal operation state
    print(f"Initial state: Production={y0[0]:.2f}, Transport={y0[1]:.2f}, Consumption={y0[2]:.2f}")
    
    # Calculate steady state using Graph-JAX
    print("\nCalculating steady state with Graph-JAX...")
    start_time = time.time()
    steady_state = gj.algorithms.steady_state(system_params, t_max=100.0, n_steps=1000)
    steady_time = time.time() - start_time
    
    print(f"Steady state calculation time: {steady_time:.4f} seconds")
    print(f"Steady state: Production={steady_state[0]:.3f}, Transport={steady_state[1]:.3f}, Consumption={steady_state[2]:.3f}")
    
    # Calculate failure time with realistic disruption period
    print(f"\nInterruption simulation (ΔT=14 days):")
    start_time = time.time()
    τ, QD = gj.algorithms.failure_time(system_params, ΔT=14.0)
    failure_time = time.time() - start_time
    
    print(f"Failure time calculation time: {failure_time:.4f} seconds")
    print(f"  System collapse time: {τ:.2f} days")
    print(f"  Average demand satisfaction rate: {QD:.3f}")
    
    # Simulate time evolution using SciPy for comparison
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
    
    t = np.linspace(0, 50, 1000)
    solution = odeint(rhs, y0, t, args=(system_params,))
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, solution[:, 0], 'b-', label='Production (y1)', linewidth=2)
    plt.plot(t, solution[:, 1], 'g-', label='Transport (y2)', linewidth=2)
    plt.plot(t, solution[:, 2], 'r-', label='Consumption (y3)', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Inventory Level')
    plt.title('System Evolution Process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.bar(['Production', 'Transport', 'Consumption'], steady_state, 
            color=['blue', 'green', 'red'], alpha=0.7)
    plt.ylabel('Steady State Inventory')
    plt.title('Steady State Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    capacities = [system_params['s12'], system_params['s23'], system_params['s13']]
    labels = ['Production→Transport', 'Transport→Consumption', 'Production→Consumption']
    plt.bar(labels, capacities, color=['orange', 'purple', 'brown'], alpha=0.7)
    plt.ylabel('Capacity Parameters')
    plt.title('Network Capacity Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    remaining_time = max(0, 14-τ)
    plt.pie([τ, remaining_time], labels=[f'Collapse Time\n{τ:.1f} days', f'Remaining Time\n{remaining_time:.1f} days'], 
            colors=['red', 'lightgray'], autopct='%1.1f%%')
    plt.title('Interruption Impact Analysis (14-day period)')
    
    plt.tight_layout()
    plt.savefig('supply_chain_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nChart saved as 'supply_chain_analysis.png'")
    
    return steady_state, τ, QD

def demonstrate_graph_jax_capabilities():
    """
    Demonstrate Graph-JAX capabilities for large-scale networks
    """
    print("\n" + "=" * 60)
    print("Graph-JAX Capabilities Demonstration")
    print("=" * 60)
    
    # Create a larger network for demonstration
    print("Creating larger network for performance demonstration...")
    
    # Generate a more reasonable sized network for demonstration
    n_nodes = 300
    n_edges = 800
    
    import networkx as nx
    G_large = nx.gnm_random_graph(n_nodes, n_edges)
    
    # Add capacities
    for u, v in G_large.edges():
        G_large[u][v]['capacity'] = np.random.uniform(0.1, 2.0)
    
    # Convert to Graph-JAX
    g_large = gj.from_networkx(G_large)
    
    print(f"Large network created: {g_large}")
    
    # Create random layer masks
    np.random.seed(42)
    production_mask = np.random.choice([True, False], size=n_nodes, p=[0.3, 0.7])
    transport_mask = np.random.choice([True, False], size=n_nodes, p=[0.3, 0.7])
    consumption_mask = np.random.choice([True, False], size=n_nodes, p=[0.4, 0.6])
    
    # Ensure no overlap
    transport_mask = transport_mask & ~production_mask
    consumption_mask = consumption_mask & ~production_mask & ~transport_mask
    
    production_mask = jnp.array(production_mask)
    transport_mask = jnp.array(transport_mask)
    consumption_mask = jnp.array(consumption_mask)
    
    print(f"Layer distribution:")
    print(f"  Production: {jnp.sum(production_mask)} nodes")
    print(f"  Transport: {jnp.sum(transport_mask)} nodes")
    print(f"  Consumption: {jnp.sum(consumption_mask)} nodes")
    
    # Performance test
    print("\nPerformance test with large network...")
    start_time = time.time()
    params_large = gj.algorithms.capacity_params(
        g_large, production_mask, transport_mask, consumption_mask,
        edge_cap=1.5, C1=50.0, C2=30.0, C3=20.0
    )
    large_network_time = time.time() - start_time
    
    print(f"Large network calculation time: {large_network_time:.4f} seconds")
    print(f"Capacity parameters for large network:")
    print(f"  s12: {params_large['s12']:.6f}")
    print(f"  s23: {params_large['s23']:.6f}")
    print(f"  s13: {params_large['s13']:.6f}")

if __name__ == "__main__":
    # Create example with Graph-JAX
    params, g, prod_mask, trans_mask, cons_mask = create_supply_chain_example()
    
    # Simulate system evolution
    steady_state, τ, QD = simulate_system_evolution(params)
    
    # Demonstrate Graph-JAX capabilities
    demonstrate_graph_jax_capabilities()
