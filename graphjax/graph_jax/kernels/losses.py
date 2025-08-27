import jax
import jax.numpy as jnp
from jax.nn import log_softmax

@jax.jit
def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    Compute multi-class cross-entropy loss.

    Args:
        logits (jnp.ndarray): Raw model output (before softmax), shape [..., n_classes].
        labels (jnp.ndarray): True labels, can be one-hot encoded or integer indices.

    Returns:
        jnp.ndarray: A scalar representing the average loss.
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
    Compute binary cross-entropy loss with logits input for numerical stability.
    This is a manual implementation of jax.nn.sigmoid_binary_cross_entropy.

    Args:
        logits (jnp.ndarray): Raw model output (before sigmoid).
        labels (jnp.ndarray): True labels (0 or 1).

    Returns:
        jnp.ndarray: A scalar representing the average loss.
    """
    # Use log-sum-exp trick to implement numerically stable version
    # Formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # JAX's softplus(x) = log(1 + exp(x)) is stable
    # softplus(-x) = log(1 + exp(-x))
    loss = jax.nn.softplus(-logits) + logits - labels * logits
    return jnp.mean(loss)

@jax.jit
def mean_squared_error(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    Compute mean squared error (MSE).

    Args:
        y_pred (jnp.ndarray): Model predictions.
        y_true (jnp.ndarray): True label values.

    Returns:
        jnp.ndarray: A scalar representing the average loss.
    """
    return jnp.mean((y_pred - y_true) ** 2)

@jax.jit
def mean_absolute_error(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    Compute mean absolute error (MAE).

    Args:
        y_pred (jnp.ndarray): Model predictions.
        y_true (jnp.ndarray): True label values.

    Returns:
        jnp.ndarray: A scalar representing the average loss.
    """
    return jnp.mean(jnp.abs(y_pred - y_true))