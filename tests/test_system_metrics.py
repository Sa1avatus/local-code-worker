from local_code_worker import system_metrics


def test_read_system_metrics_has_cpu_memory_and_gpu_list(monkeypatch) -> None:
    class Memory:
        percent = 25.0
        used = 8
        total = 32

    monkeypatch.setattr(system_metrics.psutil, "cpu_percent", lambda interval: 12.5)
    monkeypatch.setattr(system_metrics.psutil, "virtual_memory", lambda: Memory())
    result = system_metrics.read_system_metrics()
    assert result["cpu"] == {"usage_percent": 12.5}
    assert result["memory"] == {"usage_percent": 25.0, "used_bytes": 8, "total_bytes": 32}
    assert isinstance(result["gpus"], list)
