import os
import jax

def set_backend(platform: str = "cpu", *, force: bool = False):
    """
    Must be called once before importing other graph_jax modules.
    platform: cpu / gpu / tpu
    force   : Whether to override user-set environment variables
    """
    key = "JAX_PLATFORMS"
    if force or key not in os.environ:
        os.environ[key] = platform
    # Reload jax backend (only effective the first time)
    jax.config.update("jax_platform_name", platform)