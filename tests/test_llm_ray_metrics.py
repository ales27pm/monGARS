import httpx
import pytest


@pytest.mark.asyncio
async def test_ray_metrics_success_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("USE_RAY_SERVE", "true")
    monkeypatch.setenv("RAY_SERVE_URL", "http://ray/generate")

    from monGARS.core import llm_integration as module

    latency_records: list[tuple[float, dict[str, object] | None]] = []
    captured_attrs: list[tuple[str | None, dict[str, object]]] = []
    attempt_records: list[tuple[int, dict[str, object] | None]] = []

    monkeypatch.setattr(module, "_RESPONSE_CACHE", module.AsyncTTLCache())
    monkeypatch.setattr(
        module._RAY_REQUEST_COUNTER,
        "add",
        lambda amount, attributes=None: attempt_records.append((amount, attributes)),
    )
    monkeypatch.setattr(
        module._RAY_FAILURE_COUNTER,
        "add",
        lambda amount, attributes=None: None,
    )
    monkeypatch.setattr(
        module._RAY_SCALING_COUNTER,
        "add",
        lambda amount, attributes=None: None,
    )
    monkeypatch.setattr(
        module._RAY_LATENCY_HISTOGRAM,
        "record",
        lambda value, attributes=None: latency_records.append((value, attributes)),
    )

    def fake_metric_attributes(
        self, endpoint: str | None, **extra: object
    ) -> dict[str, object]:
        captured_attrs.append((endpoint, dict(extra)))
        base_endpoint = endpoint or "unknown"
        attributes: dict[str, object] = {"endpoint": base_endpoint}
        attributes.update(extra)
        return attributes

    monkeypatch.setattr(
        module.LLMIntegration, "_ray_metric_attributes", fake_metric_attributes
    )

    class SuccessfulClient:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - shim
            pass

        async def __aenter__(self) -> "SuccessfulClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def post(self, url: str, *, json: dict[str, object]) -> object:
            class Response:
                status_code = 200
                headers: dict[str, str] = {}

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, str]:
                    return {"content": "ray"}

            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", SuccessfulClient)

    llm = module.LLMIntegration()
    llm._metrics_enabled = True
    result = await llm.generate_response("hello")

    assert result["text"] == "ray"
    assert any(extra.get("status") == "success" for _, extra in captured_attrs)
    assert latency_records
    assert all(extra.get("status") != "failure" for _, extra in captured_attrs)
    assert attempt_records
    amount, attributes = attempt_records[0]
    assert amount == 1
    assert attributes is not None and attributes.get("status") == "attempt"


@pytest.mark.asyncio
async def test_ray_metrics_failure_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("USE_RAY_SERVE", "true")
    monkeypatch.setenv("RAY_SERVE_URL", "http://ray/generate")

    from monGARS.core import llm_integration as module

    latency_records: list[tuple[float, dict[str, object] | None]] = []
    captured_attrs: list[tuple[str | None, dict[str, object]]] = []
    failure_records: list[tuple[int, dict[str, object] | None]] = []
    attempt_records: list[tuple[int, dict[str, object] | None]] = []

    monkeypatch.setattr(module, "_RESPONSE_CACHE", module.AsyncTTLCache())
    monkeypatch.setattr(
        module._RAY_REQUEST_COUNTER,
        "add",
        lambda amount, attributes=None: attempt_records.append((amount, attributes)),
    )
    monkeypatch.setattr(
        module._RAY_FAILURE_COUNTER,
        "add",
        lambda amount, attributes=None: failure_records.append((amount, attributes)),
    )
    monkeypatch.setattr(
        module._RAY_SCALING_COUNTER,
        "add",
        lambda amount, attributes=None: None,
    )
    monkeypatch.setattr(
        module._RAY_LATENCY_HISTOGRAM,
        "record",
        lambda value, attributes=None: latency_records.append((value, attributes)),
    )

    def fake_metric_attributes(
        self, endpoint: str | None, **extra: object
    ) -> dict[str, object]:
        captured_attrs.append((endpoint, dict(extra)))
        base_endpoint = endpoint or "unknown"
        attributes: dict[str, object] = {"endpoint": base_endpoint}
        attributes.update(extra)
        return attributes

    monkeypatch.setattr(
        module.LLMIntegration, "_ray_metric_attributes", fake_metric_attributes
    )

    class FailingClient:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - shim
            pass

        async def __aenter__(self) -> "FailingClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def post(self, url: str, *, json: dict[str, object]) -> httpx.Response:
            raise httpx.RequestError("boom", request=httpx.Request("POST", url))

    async def fake_local(self, prompt: str, task_type: str) -> dict[str, str]:
        return {"content": f"local-{task_type}"}

    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)
    monkeypatch.setattr(
        module.LLMIntegration, "_call_local_provider", fake_local, raising=False
    )

    llm = module.LLMIntegration()
    llm._ray_max_scale_cycles = 1
    llm._metrics_enabled = True

    response = await llm.generate_response("prompt")

    assert response["text"] == "local-general"
    assert not latency_records
    assert failure_records
    assert any(extra.get("reason") == "transport_error" for _, extra in captured_attrs)
    assert any(extra.get("reason") == "exhausted" for _, extra in captured_attrs)
    assert attempt_records
    assert all(amount == 1 for amount, _ in attempt_records)
