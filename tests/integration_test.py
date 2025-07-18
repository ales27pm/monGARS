import sys
import asyncio

import pytest


def test_system_monitor_collects_stats(monkeypatch):
    """Verify SystemMonitor aggregates system metrics correctly."""

    class DummyGPUtil:
        def getGPUs(self):
            return []

    # Provide a dummy GPUtil before importing the monitor module
    monkeypatch.setitem(sys.modules, "GPUtil", DummyGPUtil())

    # Stub monGARS.config to satisfy monitor import
    import types

    dummy_config = types.ModuleType("monGARS.config")
    dummy_config.get_settings = lambda: None
    monkeypatch.setitem(sys.modules, "monGARS.config", dummy_config)

    from monGARS.core.monitor import SystemMonitor, SystemStats

    def fake_gpu_stats(self):
        return {"gpu_usage": 10.0, "gpu_memory_usage": 40.0}

    monkeypatch.setattr(SystemMonitor, "_get_gpu_stats", fake_gpu_stats)
    monkeypatch.setattr("monGARS.core.monitor.psutil.cpu_percent", lambda interval: 1.2)
    monkeypatch.setattr(
        "monGARS.core.monitor.psutil.virtual_memory",
        lambda: type("mem", (), {"percent": 64.0})(),
    )
    monkeypatch.setattr(
        "monGARS.core.monitor.psutil.disk_usage",
        lambda _: type("disk", (), {"percent": 20.0})(),
    )

    monitor = SystemMonitor(update_interval=0)
    stats: SystemStats = asyncio.run(monitor.get_system_stats())

    assert stats.cpu_usage == 1.2
    assert stats.memory_usage == 64.0
    assert stats.disk_usage == 20.0
    assert stats.gpu_usage == 10.0
    assert stats.gpu_memory_usage == 40.0
