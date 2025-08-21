import jax
import jax.numpy as jnp
from jax.nn import log_softmax

@jax.jit
def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    计算多分类交叉熵损失。

    Args:
        logits (jnp.ndarray): 模型的原始输出 (未经过 softmax)，形状为 [..., n_classes]。
        labels (jnp.ndarray): 真实的标签，可以是 one-hot 编码或整数索引。

    Returns:
        jnp.ndarray: 一个标量，表示平均损失。
    """
    lsm = log_softmax(logits)
    if labels.shape == logits.shape:
        loss = -jnp.sum(labels * lsm, axis=-1)
    else:
        loss = -jnp.take_along_axis(lsm, labels[..., None], axis=-1).squeeze(-1)
    return jnp.mean(loss)

@jax.jit
def binary_cross_entropy_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    计算二元交叉熵损失，输入为 logits 以保证数值稳定性。
    这是 jax.nn.sigmoid_binary_cross_entropy 的手动实现。

    Args:
        logits (jnp.ndarray): 模型的原始输出 (未经过 sigmoid)。
        labels (jnp.ndarray): 真实的标签 (0 或 1)。

    Returns:
        jnp.ndarray: 一个标量，表示平均损失。
    """
    # 使用 log-sum-exp 技巧实现数值稳定的版本
    # 公式: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # JAX 的 softplus(x) = log(1 + exp(x)) 是稳定的
    # softplus(-x) = log(1 + exp(-x))
    loss = jax.nn.softplus(-logits) + logits - labels * logits
    return jnp.mean(loss)

@jax.jit
def mean_squared_error(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    计算均方误差 (MSE)。

    Args:
        y_pred (jnp.ndarray): 模型的预测值。
        y_true (jnp.ndarray): 真实的标签值。

    Returns:
        jnp.ndarray: 一个标量，表示平均损失。
    """
    return jnp.mean((y_pred - y_true) ** 2)

@jax.jit
def mean_absolute_error(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    计算平均绝对误差 (MAE)。

    Args:
        y_pred (jnp.ndarray): 模型的预测值。
        y_true (jnp.ndarray): 真实的标签值。

    Returns:
        jnp.ndarray: 一个标量，表示平均损失。
    """
    return jnp.mean(jnp.abs(y_pred - y_true))