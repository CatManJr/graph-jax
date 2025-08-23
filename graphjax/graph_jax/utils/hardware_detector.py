def get_jax_env_info() -> dict:
    """
    收集当前机器的 JAX 运行环境信息并返回字典。
    不修改任何环境变量，不强制初始化任何后端。
    """
    import platform, psutil, sys, os, subprocess

    info = {
        "system": {
            "os": platform.platform(),
            "arch": platform.machine(),
            "cpu_brand": platform.processor(),
            "cpu_count": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "python": sys.version,
        },
        "jax": {},
        "devices": [],
        "apple_metal": {},
        "nvidia_gpu": [],
    }

    # --- JAX 版本与依赖 ---
    try:
        import jax, jaxlib, numpy as np, ml_dtypes
        info["jax"]["available"] = True
        info["jax"]["versions"] = {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "ml_dtypes": ml_dtypes.__version__,
        }
    except ImportError as e:
        info["jax"]["available"] = False
        info["jax"]["error"] = str(e)
        return info  # 没有 JAX，后续检测无意义

    # --- 当前可见设备 ---
    try:
        info["devices"] = [
            {"id": i, "description": str(d), "platform": d.platform, "device_kind": d.device_kind}
            for i, d in enumerate(jax.devices())
        ]
        info["jax"]["default_backend"] = jax.default_backend()
    except Exception as e:
        info["jax"]["device_error"] = str(e)

    # --- Apple Metal 提示 ---
    try:
        import jax_metal
        info["apple_metal"]["installed"] = True
        info["apple_metal"]["version"] = getattr(jax_metal, "__version__", "unknown")
    except ImportError:
        info["apple_metal"]["installed"] = False

    # --- NVIDIA GPU ---
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                name, driver, mem = line.split(", ")
                info["nvidia_gpu"].append({
                    "name": name,
                    "driver": driver,
                    "memory_mb": int(mem)
                })
    except Exception:
        pass  # nvidia-smi 不存在或无 GPU

    return info