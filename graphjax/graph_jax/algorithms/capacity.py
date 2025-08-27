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

@jax.jit
def steady_state(
    params: dict,
    *,
    t_max: float = 50.0,
    n_steps: int = 200
) -> jnp.ndarray:
    """
    Compute the system steady state using Diffrax ODE solver.
    
    Uses diffrax for JIT-compilable ODE solving.
    """
    import diffrax

    def vector_field(t, y, args):
        """ODE vector field according to paper formula (13)"""
        y1, y2, y3 = y
        p, d, s12, s23, s13 = args["p"], args["d"], args["s12"], args["s23"], args["s13"]
        a12, a23 = args["alpha12"], args["alpha23"]

        # ODE system according to paper formula (13)
        # Assume Π(y₁) = y₁, Δ(y₃) = y₃, Ψ(y_q, y_r) = y_q * y_r
        dy1 = p * y1 - s12 * y1 * y2 - s13 * y1 * y3
        dy2 = (s12 / a12) * y1 * y2 - s23 * y2 * y3
        dy3 = -d * y3 + (s13 / (a12 * a23)) * y1 * y3 + (s23 / a23) * y2 * y3
        return jnp.array([dy1, dy2, dy3])

    y0 = jnp.array([1.0, 1.0, 1.0])
    t0, t1 = 0.0, t_max
    
    # Use Diffrax solver
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),  # 5th order Runge-Kutta method
        t0, t1, dt0=0.1,
        y0=y0,
        args=params,
        saveat=diffrax.SaveAt(ts=jnp.array([t1])),  # Only save final state
    )
    
    return solution.ys[0]  # Return final state

def max_stable_demand(params: dict) -> float:
    """
    Calculate maximum stable demand using paper's max-flow condition.
    
    According to paper equation (19): α12α23d ≤ s13 + min(s12, α12s23)
    For maximum stable demand, we set equality: α12α23d = s13 + min(s12, α12s23)
    
    Args:
        params: Dictionary containing network parameters (s12, s23, s13, alpha12, alpha23)
    
    Returns:
        Maximum stable demand level
    """
    s12 = float(params["s12"])
    s23 = float(params["s23"])
    s13 = float(params["s13"])
    alpha12 = float(params["alpha12"])
    alpha23 = float(params["alpha23"])
    
    # Calculate the right side of the equation
    direct_path_capacity = s13  # Direct path from refineries to gas stations
    long_path_capacity = min(s12, alpha12 * s23)  # Long path through terminals
    
    total_capacity = direct_path_capacity + long_path_capacity
    
    # Calculate maximum stable demand
    max_stable_demand = total_capacity / (alpha12 * alpha23)
    
    return max_stable_demand

@jax.jit
def failure_time(params: dict, ΔT: float) -> tuple:
    """
    Compute the system failure time and average demand level after production interruption.
    
    Uses JAX operations for JIT compilation.
    """
    y_steady = steady_state(params)
    y3_0 = y_steady[2]  # Keep as JAX array
    p = params["p"]     # Keep as JAX array
    d = params["d"]     # Keep as JAX array
    
    # Check if system is stable (production >= demand)
    def stable_case():
        return ΔT, 1.0
    
    def unstable_case():
        τ = jnp.minimum(ΔT, y3_0 / d)
        QD = jnp.where(τ >= ΔT, 1.0, (y3_0 - 0.5 * d * τ) / y3_0)
        return τ, QD
    
    τ, QD = jax.lax.cond(p >= d, stable_case, unstable_case)
    return τ, QD

# Batch computation functions using JAX vmap
batch_steady_state = jax.vmap(steady_state, in_axes=(0, None, None))

def batch_failure_time(params_array: jnp.ndarray, ΔT: float) -> tuple:
    """
    Batch failure time computation using JAX vmap.
    """
    return jax.vmap(lambda p: failure_time(p, ΔT))(params_array)
