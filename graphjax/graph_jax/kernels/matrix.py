import jax
import jax.numpy as jnp
from ..graphs import Graph
from functools import partial
from typing import Optional
from jax.experimental import ode

# --- Internal JIT-compiled pure functions ---

# Modified: Add n_edges to static_argnames as well
@partial(jax.jit, static_argnames=('n_nodes', 'n_edges', 'as_diagonal'))
def _degree_matrix_pure(
    receivers: jnp.ndarray, 
    edge_weights: Optional[jnp.ndarray], # Parameter kept but not used in function body
    n_nodes: int, 
    n_edges: int,
    as_diagonal: bool
) -> jnp.ndarray:
    """
    Pure function version of degree matrix computation.
    Modified: This version computes unweighted degrees to match networkx's default behavior.
    """
    # Ignore edge_weights, add 1 for each edge to compute unweighted degrees
    ones = jnp.ones(n_edges, dtype=jnp.float32)
    degrees = jnp.zeros(n_nodes, dtype=jnp.float32).at[receivers].add(ones)
    return jnp.diag(degrees) if as_diagonal else degrees

@partial(jax.jit, static_argnames=('n_nodes',))
def _laplacian_matrix_pure(
    adj: jnp.ndarray,
    n_nodes: int
) -> jnp.ndarray:
    """Pure function version of Laplacian matrix computation."""
    degrees = jnp.sum(adj, axis=1)
    deg_matrix = jnp.diag(degrees)
    return deg_matrix - adj

@partial(jax.jit, static_argnames=('n_nodes', 'add_self_loops'))
def _normalized_laplacian_sym_pure(adj: jnp.ndarray, n_nodes: int, add_self_loops: bool) -> jnp.ndarray:
    """
    Pure function: Compute symmetric normalized Laplacian matrix.
    Follows NetworkX's standard definition.
    """
    # Fix: Always use original adjacency matrix to compute degrees, this is the standard definition
    degrees = jnp.sum(adj, axis=1)
    
    # If needed, add self-loops to the final adjacency matrix
    if add_self_loops:
        adj_processed = adj + jnp.eye(n_nodes)
    else:
        adj_processed = adj

    # Prevent division by zero
    degrees_inv_sqrt = jnp.where(degrees > 0, 1.0 / jnp.sqrt(degrees), 0)
    D_inv_sqrt = jnp.diag(degrees_inv_sqrt)
    
    # L_sym = I - D^(-1/2) A' D^(-1/2)
    # where A' is the adjacency matrix that may have self-loops added
    l_sym = jnp.eye(n_nodes) - D_inv_sqrt @ adj_processed @ D_inv_sqrt
    return l_sym

@partial(jax.jit, static_argnames=('n_nodes', 'add_self_loops'))
def _random_walk_normalized_laplacian_pure(
    adj: jnp.ndarray,
    n_nodes: int,
    add_self_loops: bool
) -> jnp.ndarray:
    """Pure function version of random walk normalized Laplacian matrix computation (L_rw = I - D⁻¹A)."""
    if add_self_loops:
        adj += jnp.eye(n_nodes)
    
    degrees = jnp.sum(adj, axis=1)
    # Prevent division by zero
    inv_deg = jnp.where(degrees > 0, 1.0 / degrees, 0)
    d_inv = jnp.diag(inv_deg)

    return jnp.eye(n_nodes) - d_inv @ adj

@partial(jax.jit, static_argnames=('k', 'force_float64'))
def _laplacian_eigensystem_pure(l_sym: jnp.ndarray, k: int, force_float64: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure function: Compute eigendecomposition of symmetric matrix."""
    # Optimization: Decide whether to convert to float64 based on parameter
    if force_float64:
        l_sym = l_sym.astype(jnp.float64)
    
    # Use jax.scipy.linalg.eigh for efficient computation
    # eigh assumes the matrix is Hermitian (symmetric for real matrices)
    # It returns eigenvalues in sorted order
    vals, vecs = jnp.linalg.eigh(l_sym)
    return vals[:k], vecs[:, :k]


# --- External call wrapper functions ---

def degree_matrix(graph: Graph, as_diagonal: bool = True) -> jnp.ndarray:
    """Compute graph degrees."""
    return _degree_matrix_pure(
        receivers=graph.receivers,
        edge_weights=graph.edge_weights,
        n_nodes=graph.n_nodes,
        n_edges=graph.n_edges,
        as_diagonal=as_diagonal
    )

def laplacian_matrix(graph: Graph) -> jnp.ndarray:
    """Compute the combinatorial Laplacian matrix of the graph (L = D - A)."""
    adj = graph.to_adjacency_matrix()
    return _laplacian_matrix_pure(adj, n_nodes=graph.n_nodes)

def normalized_laplacian_sym(graph: Graph, add_self_loops: bool = True, use_weights: bool = False) -> jnp.ndarray:
    """
    Compute symmetric normalized Laplacian matrix.

    Args:
        graph (Graph): Input graph.
        add_self_loops (bool): Whether to add self-loops before computation.
        use_weights (bool): Whether to use edge weights. Default is False.

    Returns:
        jnp.ndarray: Symmetric normalized Laplacian matrix.
    """
    # Fix: Choose correct adjacency matrix based on use_weights parameter
    if use_weights:
        adj = graph.to_adjacency_matrix()
    else:
        adj = graph.to_unweighted_adjacency_matrix()
    return _normalized_laplacian_sym_pure(adj, graph.n_nodes, add_self_loops)

def random_walk_normalized_laplacian(graph: Graph, add_self_loops: bool = True) -> jnp.ndarray:
    """
    Compute random walk normalized Laplacian matrix (L_rw = I - D⁻¹A).

    Args:
        graph (Graph): Input graph.
        add_self_loops (bool): Whether to add self-loops before computing degrees.

    Returns:
        jnp.ndarray: Random walk normalized Laplacian matrix.
    """
    adj = graph.to_adjacency_matrix()
    return _random_walk_normalized_laplacian_pure(adj, n_nodes=graph.n_nodes, add_self_loops=add_self_loops)

def laplacian_eigensystem(graph: Graph, k: int, use_weights: bool = False, force_float64: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the first k eigenvalues and eigenvectors of the symmetric normalized Laplacian matrix.

    This is crucial for tasks like spectral clustering, graph Fourier transform, etc.

    Args:
        graph (Graph): Input graph.
        k (int): Number of smallest eigenvalues/vectors to compute.
        use_weights (bool): Whether to use edge weights in computation. Default is False.
        force_float64 (bool): Whether to force float64 computation to ensure numerical precision.
                              Default is True to match SciPy/NumPy results.
                              Can be set to False in performance-sensitive applications.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: (eigenvalues, eigenvectors)
    """
    # Spectral analysis is usually performed on symmetric normalized Laplacian without self-loops
    l_sym = normalized_laplacian_sym(graph, add_self_loops=False, use_weights=use_weights)
    return _laplacian_eigensystem_pure(l_sym, k=k, force_float64=force_float64)

# steady state solver for the ODE system
def steady_state(params, *, t_max=50.0, n_steps=200):
    """
    Use SciPy ODE solver to compute system steady state.
    
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

def steady_state_batch(params_array, *, t_max=50.0, n_steps=200):
    """
    Batch steady state computation.
    
    TODO: Optimize with JAX vmap when ODE solver is replaced with JAX-native version.
    """
    results = []
    for params in params_array:
        result = steady_state(params, t_max=t_max, n_steps=n_steps)
        results.append(result)
    return jnp.stack(results)