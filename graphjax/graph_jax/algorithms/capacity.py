import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from functools import partial
from ..graphs import Graph
from ..kernels.min_cut import min_cut_matrix_optimized

@partial(jax.jit, static_argnames=('use_parallel',))
def capacity_params(
    g: Graph,
    ref_mask: jnp.ndarray,
    term_mask: jnp.ndarray,
    gas_mask: jnp.ndarray,
    *,
    C1: float = 38.2,
    C2: float = 31.0,
    C3: float = 0.035,
    edge_cap: float = 4.0 * 7 * 1e6 / 1e6,
    use_parallel: bool = False
) -> dict:
    """
    Compute capacity parameters using XLA-optimized routines.
    Inter-layer connections are computed using the min-cut algorithm, fully consistent with the paper.
    Parallel computation is supported to improve performance on large-scale graphs.
    """
    alive = g.node_mask if g.node_mask is not None else jnp.ones(g.n_nodes, dtype=bool)

    N1 = jnp.sum(alive * ref_mask)
    N2 = jnp.sum(alive * term_mask)
    N3 = jnp.sum(alive * gas_mask)

    # Create sparse adjacency matrix
    capacities = jnp.full(g.n_edges, edge_cap, dtype=jnp.float32)
    coo = BCOO((capacities, jnp.stack([g.senders, g.receivers], axis=1)),
               shape=(g.n_nodes, g.n_nodes))
    
    # Compute inter-layer connections using the optimized min-cut algorithm, fully consistent with the paper
    # Automatically use the parallel version when the graph size exceeds the threshold
    # Compute min-cut capacity from ref to term
    W12 = min_cut_matrix_optimized(coo, ref_mask, term_mask, use_parallel)
    
    # Compute min-cut capacity from term to gas
    W23 = min_cut_matrix_optimized(coo, term_mask, gas_mask, use_parallel)
    
    # Compute min-cut capacity from ref to gas
    W13 = min_cut_matrix_optimized(coo, ref_mask, gas_mask, use_parallel)

    s12 = W12 / (N1 * C1 + 1e-8)
    s23 = W23 / (N2 * C2 + 1e-8)
    s13 = W13 / (N1 * C1 + 1e-8)

    alpha12 = (N2 * C2) / (N1 * C1 + 1e-8)
    alpha23 = (N3 * C3) / (N2 * C2 + 1e-8)

    return dict(N1=N1, N2=N2, N3=N3,
                s12=s12, s23=s23, s13=s13,
                alpha12=alpha12, alpha23=alpha23,
                p=0.42, d=0.79)

# JIT-compiled capacity algorithm
capacity_params_jit = jax.jit(capacity_params)
batch_capacity_params = jax.vmap(capacity_params_jit, in_axes=(None, 0, 0, 0))

def steady_state(
    params: dict,
    *,
    t_max: float = 50.0,
    n_steps: int = 200
) -> jnp.ndarray:
    """
    Compute the system steady state using the SciPy ODE solver.
    
    TODO: Replace with fast JAX-native ODE solver when stable and mature.
    Currently using SciPy for consistency and stability.
    """
    import numpy as np
    from scipy.integrate import odeint

    def rhs(y, t, params):
        y1, y2, y3 = y
        p, d, s12, s23, s13 = params["p"], params["d"], params["s12"], params["s23"], params["s13"]
        a12, a23 = params["alpha12"], params["alpha23"]

            # ODE system according to paper formula (13)
    # Assume Π(y₁) = y₁, Δ(y₃) = y₃, Ψ(y_q, y_r) = y_q * y_r
        dy1 = p * y1 - s12 * y1 * y2 - s13 * y1 * y3
        dy2 = (s12 / a12) * y1 * y2 - s23 * y2 * y3
        dy3 = -d * y3 + (s13 / (a12 * a23)) * y1 * y3 + (s23 / a23) * y2 * y3
        return [dy1, dy2, dy3]

    y0 = [1.0, 1.0, 1.0]
    t = np.linspace(0, t_max, n_steps)
    solution = odeint(rhs, y0, t, args=(params,))
    
    # Convert back to JAX array for consistency
    return jnp.array(solution[-1])

def failure_time(params: dict, ΔT: float) -> tuple:
    """
    Compute the system failure time and average demand level after production interruption.
    
    TODO: Make this JIT-compilable when ODE solver is replaced with JAX-native version.
    """
    y_steady = steady_state(params)
    y3_0 = float(y_steady[2])  # Convert to Python float for compatibility
    p = float(params["p"])     # Production rate
    d = float(params["d"])     # Demand rate
    
    import numpy as np
    
    # Check if system is stable (production >= demand)
    if p >= d:
        # System is stable, no failure
        τ = ΔT
        QD = 1.0
    else:
        # System will fail, calculate failure time
        τ = min(ΔT, y3_0 / d)
        QD = 1.0 if τ >= ΔT else (y3_0 - 0.5 * d * τ) / y3_0
    
    # Convert back to JAX arrays for consistency
    return jnp.array(τ), jnp.array(QD)

# Batch computation functions
def batch_steady_state(params_array: list, *, t_max: float = 50.0, n_steps: int = 200) -> jnp.ndarray:
    """
    Batch steady state computation.
    
    TODO: Optimize with JAX vmap when ODE solver is replaced with JAX-native version.
    Currently using Python loop for compatibility with SciPy.
    """
    results = []
    for params in params_array:
        result = steady_state(params, t_max=t_max, n_steps=n_steps)
        results.append(result)
    return jnp.stack(results)

def batch_failure_time(params_array: list, ΔT: float) -> tuple:
    """
    Batch failure time computation.
    
    TODO: Optimize with JAX vmap when functions are JIT-compilable.
    """
    tau_results = []
    qd_results = []
    for params in params_array:
        τ, QD = failure_time(params, ΔT)
        tau_results.append(τ)
        qd_results.append(QD)
    return jnp.stack(tau_results), jnp.stack(qd_results)
