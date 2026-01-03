import asyncio
import importlib
import os
import sys
import types

import pytest

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("OTEL_METRICS_ENABLED", "False")
os.environ.setdefault("OTEL_SDK_DISABLED", "True")
metric_module = types.ModuleType(
    "opentelemetry.exporter.otlp.proto.http.metric_exporter"
)
metric_module.OTLPMetricExporter = lambda *a, **k: types.SimpleNamespace(
    export=lambda *x, **y: None
)
sys.modules["opentelemetry.exporter.otlp.proto.http.metric_exporter"] = metric_module

module = types.ModuleType("ollama")
sys.modules.setdefault("ollama", module)

llm_module = importlib.import_module("monGARS.core.llm_integration")
CircuitBreaker = llm_module.CircuitBreaker
CircuitBreakerOpenError = llm_module.CircuitBreakerOpenError


@pytest.mark.asyncio
async def test_circuit_breaker_trips_and_recovers():
    cb = CircuitBreaker(fail_max=2, reset_timeout=1)

    async def fail():
        raise RuntimeError("boom")

    async def succeed():
        return 42

    with pytest.raises(RuntimeError):
        await cb.call(fail)
    with pytest.raises(RuntimeError):
        await cb.call(fail)
    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(succeed)

    await asyncio.sleep(1.1)
    result = await cb.call(succeed)
    assert result == 42
