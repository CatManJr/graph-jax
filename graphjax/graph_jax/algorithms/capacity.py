import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from functools import partial
from ..graphs import Graph
from ..kernels.min_cut import min_cut_matrix, min_cut_matrix_optimized

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
    使用XLA优化的capacity参数计算。
    使用最小割算法计算层间连接，完全符合论文描述。
    支持并行计算以提高大规模图的性能。
    """
    alive = g.node_mask if g.node_mask is not None else jnp.ones(g.n_nodes, dtype=bool)

    N1 = jnp.sum(alive * ref_mask)
    N2 = jnp.sum(alive * term_mask)
    N3 = jnp.sum(alive * gas_mask)

    # 创建稀疏邻接矩阵
    capacities = jnp.full(g.n_edges, edge_cap, dtype=jnp.float32)
    coo = BCOO((capacities, jnp.stack([g.senders, g.receivers], axis=1)),
               shape=(g.n_nodes, g.n_nodes))
    
    # 使用优化的最小割算法计算层间连接，完全符合论文描述
    # 当图规模超过阈值时自动使用并行版本
    # 计算从ref到term的最小割容量
    W12 = min_cut_matrix_optimized(coo, ref_mask, term_mask, use_parallel)
    
    # 计算从term到gas的最小割容量
    W23 = min_cut_matrix_optimized(coo, term_mask, gas_mask, use_parallel)
    
    # 计算从ref到gas的最小割容量
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

# JIT编译的capacity算法
capacity_params_jit = jax.jit(capacity_params)
batch_capacity_params = jax.vmap(capacity_params_jit, in_axes=(None, 0, 0, 0))

def steady_state(
    params: dict,
    *,
    t_max: float = 50.0,
    n_steps: int = 200
) -> jnp.ndarray:
    """
    使用SciPy ODE求解器计算系统的稳态。
    
    TODO: Replace with fast JAX-native ODE solver when stable and mature.
    Currently using SciPy for consistency and stability.
    """
    import numpy as np
    from scipy.integrate import odeint

    def rhs(y, t, params):
        y1, y2, y3 = y
        p, d, s12, s23, s13 = params["p"], params["d"], params["s12"], params["s23"], params["s13"]
        a12, a23 = params["alpha12"], params["alpha23"]

        dy1 = p - s12 * y1 * y2 - s13 * y1 * y3
        dy2 = s12 / a12 * y1 * y2 - s23 * y2 * y3
        dy3 = -d * y3 + s13 / (a12 * a23) * y1 * y3 + s23 / a23 * y2 * y3
        return [dy1, dy2, dy3]

    y0 = [1.0, 1.0, 1.0]
    t = np.linspace(0, t_max, n_steps)
    solution = odeint(rhs, y0, t, args=(params,))
    
    # Convert back to JAX array for consistency
    return jnp.array(solution[-1])

def failure_time(params: dict, ΔT: float) -> tuple:
    """
    计算生产中断后的系统失败时间和平均需求水平。
    
    TODO: Make this JIT-compilable when ODE solver is replaced with JAX-native version.
    """
    y_steady = steady_state(params)
    y3_0 = float(y_steady[2])  # Convert to Python float for compatibility
    d = float(params["d"])
    
    import numpy as np
    τ = min(ΔT, y3_0 / d)
    QD = 1.0 if τ >= ΔT else (y3_0 - 0.5 * d * τ) / y3_0
    
    # Convert back to JAX arrays for consistency
    return jnp.array(τ), jnp.array(QD)

# Batch computation functions
def batch_steady_state(params_array: list, *, t_max: float = 50.0, n_steps: int = 200) -> jnp.ndarray:
    """
    批量稳态计算。
    
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
    批量失败时间计算。
    
    TODO: Optimize with JAX vmap when functions are JIT-compilable.
    """
    tau_results = []
    qd_results = []
    for params in params_array:
        τ, QD = failure_time(params, ΔT)
        tau_results.append(τ)
        qd_results.append(QD)
    return jnp.stack(tau_results), jnp.stack(qd_results)

