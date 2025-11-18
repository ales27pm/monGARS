from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from monGARS.api.web_api import app
from monGARS.config import configure_telemetry, get_settings
from monGARS.core import monitor

pytestmark = pytest.mark.usefixtures("ensure_test_users")


def _ensure_prometheus_reader() -> None:
    """Configure telemetry once per test session."""

    if getattr(_ensure_prometheus_reader, "_configured", False):
        return

    pytest_module = sys.modules.pop("pytest", None)
    pytest_flag = os.environ.pop("PYTEST_CURRENT_TEST", None)
    try:
        settings = get_settings()
        test_settings = settings.model_copy(
            update={
                "otel_metrics_enabled": False,
                "otel_traces_enabled": False,
                "otel_prometheus_enabled": True,
            }
        )
        configure_telemetry(test_settings)
    finally:
        if pytest_module is not None:
            sys.modules["pytest"] = pytest_module
        if pytest_flag is not None:
            os.environ["PYTEST_CURRENT_TEST"] = pytest_flag

    _ensure_prometheus_reader._configured = True


def test_metrics_endpoint_reports_llm_metrics() -> None:
    _ensure_prometheus_reader()

    with TestClient(app) as client:
        token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
            "access_token"
        ]

        monitor.record_llm_metrics(
            model_id="dolphin-test",
            user_id="researcher",
            conversation_id="conversation-test",
            input_tokens=5,
            output_tokens=7,
            latency_ms=123.0,
            extra_attributes={"request.id": "req-metrics"},
        )
        monitor.LLM_ERROR_COUNTER.add(
            1, {"error.type": "RuntimeError", "model": "dolphin-test"}
        )

        resp = client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

        metrics_map = {
            family.name: family for family in text_string_to_metric_families(resp.text)
        }

        def _find_metric(prefix: str) -> tuple[str, any]:
            for name, family in metrics_map.items():
                if name.startswith(prefix):
                    return name, family
            raise AssertionError(f"Missing metric prefix {prefix}")

        tokens_name, tokens_family = _find_metric("mongars_llm_tokens")
        assert any(sample.value >= 0 for sample in tokens_family.samples)

        duration_name, duration_family = _find_metric("mongars_llm_duration")
        assert any(sample.value >= 0 for sample in duration_family.samples)

        errors_name, errors_family = _find_metric("mongars_llm_errors")
        assert any(sample.value >= 0 for sample in errors_family.samples)
