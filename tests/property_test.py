import asyncio
import importlib
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

TieredCache = importlib.import_module("monGARS.core.caching.tiered_cache").TieredCache


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


@pytest.mark.asyncio
async def test_tiered_cache_miss(tmp_path):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()
    assert await cache.get("missing") is None


@pytest.mark.asyncio
async def test_tiered_cache_overwrite(tmp_path):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()
    await cache.set("k", "v1")
    await cache.set("k", "v2")
    assert await cache.get("k") == "v2"
