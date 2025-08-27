import jax
import jax.numpy as jnp
from ..graphs import Graph
from typing import Optional, Sequence, Union
from functools import partial
from .matrix import steady_state

# Optimized graph operation functions using JAX native operations
@jax.jit
def compute_node_degrees(graph: Graph) -> jnp.ndarray:
    """
    Compute node degrees - using JAX native operations
    """
    n_nodes = graph.n_nodes
    degrees = jnp.zeros(n_nodes, dtype=jnp.int32)
    
    # Use scatter_add to compute degrees
    degrees = degrees.at[graph.senders].add(1)
    degrees = degrees.at[graph.receivers].add(1)
    
    return degrees

@jax.jit
def compute_edge_capacities(graph: Graph, capacity_func=None) -> jnp.ndarray:
    """
    Compute edge capacities - using JAX native operations
    """
    if capacity_func is None:
        # Default capacity function: based on node degrees
        degrees = compute_node_degrees(graph)
        sender_degrees = degrees[graph.senders]
        receiver_degrees = degrees[graph.receivers]
        return sender_degrees + receiver_degrees
    else:
        return capacity_func(graph)

@jax.jit
def create_layer_masks(n_nodes: int, ref_ratio: float = 0.33, term_ratio: float = 0.33) -> tuple:
    """
    Create network layer masks - using JAX native operations
    """
    n_ref = int(n_nodes * ref_ratio)
    n_term = int(n_nodes * term_ratio)
    
    ref_mask = jnp.arange(n_nodes) < n_ref
    term_mask = (jnp.arange(n_nodes) >= n_ref) & (jnp.arange(n_nodes) < n_ref + n_term)
    gas_mask = jnp.arange(n_nodes) >= n_ref + n_term
    
    return ref_mask, term_mask, gas_mask

@jax.jit
def compute_layer_statistics(graph: Graph, ref_mask: jnp.ndarray, term_mask: jnp.ndarray, gas_mask: jnp.ndarray) -> dict:
    """
    Compute network layer statistics - using JAX native operations
    """
    degrees = compute_node_degrees(graph)
    
    ref_degrees = degrees * ref_mask
    term_degrees = degrees * term_mask
    gas_degrees = degrees * gas_mask
    
    ref_avg_degree = jnp.sum(ref_degrees) / (jnp.sum(ref_mask) + 1e-8)
    term_avg_degree = jnp.sum(term_degrees) / (jnp.sum(term_mask) + 1e-8)
    gas_avg_degree = jnp.sum(gas_degrees) / (jnp.sum(gas_mask) + 1e-8)
    
    return {
        'ref_avg_degree': ref_avg_degree,
        'term_avg_degree': term_avg_degree,
        'gas_avg_degree': gas_avg_degree,
        'ref_nodes': jnp.sum(ref_mask),
        'term_nodes': jnp.sum(term_mask),
        'gas_nodes': jnp.sum(gas_mask)
    }

@jax.jit
def gemm(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    General Matrix Multiplication (GEMM).
    This is a simple wrapper around jnp.matmul to maintain API consistency.
    """
    return a @ b

@partial(jax.jit, static_argnames=('strides', 'padding'))
def conv(
    inputs: jnp.ndarray, 
    kernel: jnp.ndarray, 
    strides: Union[int, Sequence[int]], 
    padding: str
) -> jnp.ndarray:
    """
    Generic N-D convolution operator.
    This is a wrapper around jax.lax.conv_general_dilated to simplify common 2D convolution.

    Args:
        inputs (jnp.ndarray): Input tensor, typically shape (N, H, W, C_in).
        kernel (jnp.ndarray): Convolution kernel, typically shape (H_k, W_k, C_in, C_out).
        strides (Union[int, Sequence[int]]): Stride.
        padding (str): Padding mode, 'SAME' or 'VALID'.

    Returns:
        jnp.ndarray: Convolved output tensor.
    """
    if isinstance(strides, int):
        strides = (strides, strides)
        
    # JAX's low-level API requires more detailed dimension information
    dn = jax.lax.conv_dimension_numbers(
        inputs.shape,     # Input shape
        kernel.shape,     # Kernel shape
        ('NHWC', 'HWIO', 'NHWC')  # (Input format, kernel format, output format)
    )

    return jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        dimension_numbers=dn
    )

@partial(jax.jit, static_argnames=('strides', 'padding'))
def conv_1x1(
    inputs: jnp.ndarray, 
    kernel: jnp.ndarray, 
    strides: Union[int, Sequence[int]] = 1, 
    padding: str = 'SAME'
) -> jnp.ndarray:
    """
    1x1 convolution. Usually used for cross-channel mixing information or adjusting channel count.
    
    Args:
        inputs (jnp.ndarray): Input tensor (N, H, W, C_in).
        kernel (jnp.ndarray): Convolution kernel, must be (1, 1, C_in, C_out).
        strides (Union[int, Sequence[int]]): Stride.
        padding (str): Padding mode.

    Returns:
        jnp.ndarray: Convolved output tensor.
    """
    assert kernel.shape[:2] == (1, 1), "Kernel for conv_1x1 must have shape (1, 1, ...)"
    return conv(inputs, kernel, strides, padding)

@partial(jax.jit, static_argnames=('strides', 'padding'))
def conv_3x3(
    inputs: jnp.ndarray, 
    kernel: jnp.ndarray, 
    strides: Union[int, Sequence[int]] = 1, 
    padding: str = 'SAME'
) -> jnp.ndarray:
    """
    3x3 convolution. The most commonly used feature extractor in CNNs.

    Args:
        inputs (jnp.ndarray): Input tensor (N, H, W, C_in).
        kernel (jnp.ndarray): Convolution kernel, must be (3, 3, C_in, C_out).
        strides (Union[int, Sequence[int]]): Stride.
        padding (str): Padding mode.

    Returns:
        jnp.ndarray: Convolved output tensor.
    """
    assert kernel.shape[:2] == (3, 3), "Kernel for conv_3x3 must have shape (3, 3, ...)"
    return conv(inputs, kernel, strides, padding)

@jax.jit
def failure_time(params, ΔT):
    y_steady = steady_state(params)
    y3_0 = y_steady[2]
    d = params["d"]
    τ = jnp.minimum(ΔT, y3_0 / d)
    QD = jnp.where(τ >= ΔT, 1.0, (y3_0 - 0.5 * d * τ) / y3_0)
    return τ, QD