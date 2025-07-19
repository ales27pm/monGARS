import asyncio
import os
import sys
import types

import pytest

# Disable metrics exporter during tests
os.environ.setdefault("OTEL_METRICS_ENABLED", "False")
os.environ.setdefault("OTEL_SDK_DISABLED", "True")
metric_module = types.ModuleType(
    "opentelemetry.exporter.otlp.proto.http.metric_exporter"
)
metric_module.OTLPMetricExporter = lambda *a, **k: types.SimpleNamespace(
    export=lambda *x, **y: None
)
sys.modules["opentelemetry.exporter.otlp.proto.http.metric_exporter"] = metric_module

from monGARS.core.caching.tiered_cache import TieredCache


@pytest.mark.asyncio
@pytest.mark.parametrize("key,value", [("a", "b"), ("1", "2"), ("x", "y")])
async def test_tiered_cache_roundtrip(tmp_path, key, value):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()
    await cache.set(key, value)
    assert await cache.get(key) == value


@pytest.mark.asyncio
async def test_tiered_cache_ttl_expiry(tmp_path):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()
    await cache.set("expire", "v", ttl=1)
    await asyncio.sleep(1.1)
    assert await cache.get("expire") is None
