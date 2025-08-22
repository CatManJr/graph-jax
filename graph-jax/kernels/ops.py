import jax
import jax.numpy as jnp
from graphs import Graph
from typing import Optional, Sequence, Union
from functools import partial

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