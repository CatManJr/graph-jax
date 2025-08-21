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

@jax.jit
def spgemm(graph: Graph, node_features: jnp.ndarray) -> jnp.ndarray:
    """
    稀疏-稠密矩阵乘法 (Sparse-GEMM)，也称为消息传递或图传播算子。
    它高效地计算 A_sparse @ X_dense，其中 A_sparse 是图的邻接矩阵。

    Args:
        graph (Graph): 包含稀疏连接信息的图对象。
        node_features (jnp.ndarray): 节点特征矩阵 (X_dense)。

    Returns:
        jnp.ndarray: 传播后的新节点特征矩阵。
    """
    # 1. 从发送方节点收集特征 (Gather)
    sender_features = node_features[graph.senders]

    # 2. (可选) 按边权重缩放消息
    if graph.edge_weights is not None:
        # 使用广播将权重应用到每个特征维度
        messages = sender_features * graph.edge_weights[:, None]
    else:
        messages = sender_features

    # 3. 将消息聚合到接收方节点 (Sum Aggregation)
    # 创建一个全零矩阵，然后将消息累加到对应的接收方索引上
    aggregated_features = jnp.zeros_like(node_features).at[graph.receivers].add(messages)
    
    return aggregated_features

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