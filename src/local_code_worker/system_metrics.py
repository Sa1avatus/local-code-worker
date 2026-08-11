import psutil


def read_system_metrics() -> dict[str, object]:
    """Read non-sensitive host metrics, degrading gracefully without NVIDIA access."""
    memory = psutil.virtual_memory()
    result: dict[str, object] = {
        "cpu": {"usage_percent": round(psutil.cpu_percent(interval=0.1), 1)},
        "memory": {
            "usage_percent": round(memory.percent, 1),
            "used_bytes": memory.used,
            "total_bytes": memory.total,
        },
        "gpus": [],
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            vram = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result["gpus"].append(
                {
                    "name": pynvml.nvmlDeviceGetName(handle),
                    "usage_percent": utilization.gpu,
                    "memory_used_bytes": vram.used,
                    "memory_total_bytes": vram.total,
                    "temperature_celsius": pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "power_watts": round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1_000, 1),
                    "power_limit_watts": round(
                        pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1_000, 1
                    ),
                    "clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS),
                }
            )
    except Exception:
        # NVIDIA metrics are optional; do not make model management unavailable.
        pass
    return result
