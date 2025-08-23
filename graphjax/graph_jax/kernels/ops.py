import jax
import jax.numpy as jnp
from ..graphs import Graph
from typing import Optional, Sequence, Union
from functools import partial
from .matrix import steady_state

# 优化的图操作函数，使用 JAX 原生操作
@jax.jit
def compute_node_degrees(graph: Graph) -> jnp.ndarray:
    """
    计算节点度数 - 使用 JAX 原生操作
    """
    n_nodes = graph.n_nodes
    degrees = jnp.zeros(n_nodes, dtype=jnp.int32)
    
    # 使用 scatter_add 计算度数
    degrees = degrees.at[graph.senders].add(1)
    degrees = degrees.at[graph.receivers].add(1)
    
    return degrees

@jax.jit
def compute_edge_capacities(graph: Graph, capacity_func=None) -> jnp.ndarray:
    """
    计算边容量 - 使用 JAX 原生操作
    """
    if capacity_func is None:
        # 默认容量函数：基于节点度数
        degrees = compute_node_degrees(graph)
        sender_degrees = degrees[graph.senders]
        receiver_degrees = degrees[graph.receivers]
        return sender_degrees + receiver_degrees
    else:
        return capacity_func(graph)

@jax.jit
def create_layer_masks(n_nodes: int, ref_ratio: float = 0.33, term_ratio: float = 0.33) -> tuple:
    """
    创建网络层掩码 - 使用 JAX 原生操作
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
    计算网络层统计信息 - 使用 JAX 原生操作
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
    通用矩阵乘法 (General Matrix Multiplication, GEMM)。
    这是一个对 jnp.matmul 的简单封装，以保持 API 的一致性。
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
    通用的 N-D 卷积算子。
    这是一个对 jax.lax.conv_general_dilated 的封装，以简化常用的 2D 卷积。

    Args:
        inputs (jnp.ndarray): 输入张量，通常形状为 (N, H, W, C_in)。
        kernel (jnp.ndarray): 卷积核，通常形状为 (H_k, W_k, C_in, C_out)。
        strides (Union[int, Sequence[int]]): 步长。
        padding (str): 填充模式，'SAME' 或 'VALID'。

    Returns:
        jnp.ndarray: 卷积后的输出张量。
    """
    if isinstance(strides, int):
        strides = (strides, strides)
        
    # JAX 的底层 API 需要更详细的维度信息
    dn = jax.lax.conv_dimension_numbers(
        inputs.shape,     # 输入形状
        kernel.shape,     # 核形状
        ('NHWC', 'HWIO', 'NHWC')  # (输入格式, 核格式, 输出格式)
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
    1x1 卷积。通常用于跨通道混合信息或调整通道数。
    
    Args:
        inputs (jnp.ndarray): 输入张量 (N, H, W, C_in)。
        kernel (jnp.ndarray): 卷积核，必须为 (1, 1, C_in, C_out)。
        strides (Union[int, Sequence[int]]): 步长。
        padding (str): 填充模式。

    Returns:
        jnp.ndarray: 卷积后的输出张量。
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
    3x3 卷积。CNN 中最常用的特征提取器。

    Args:
        inputs (jnp.ndarray): 输入张量 (N, H, W, C_in)。
        kernel (jnp.ndarray): 卷积核，必须为 (3, 3, C_in, C_out)。
        strides (Union[int, Sequence[int]]): 步长。
        padding (str): 填充模式。

    Returns:
        jnp.ndarray: 卷积后的输出张量。
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