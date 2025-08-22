import os
import jax

def set_backend(platform: str = "cpu", *, force: bool = False):
    """
    必须在 import graph_jax 其它模块之前调用一次。
    platform: cpu / gpu / tpu
    force   : 是否覆盖用户已设置的环境变量
    """
    key = "JAX_PLATFORMS"
    if force or key not in os.environ:
        os.environ[key] = platform
    # 重新加载 jax 后端（仅第一次有效）
    jax.config.update("jax_platform_name", platform)