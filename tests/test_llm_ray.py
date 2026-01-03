import os

import httpx
import pytest


@pytest.mark.asyncio
async def test_llm_integration_uses_ray(monkeypatch):
    os.environ.setdefault("SECRET_KEY", "test-secret")
    os.environ["USE_RAY_SERVE"] = "True"
    os.environ["RAY_SERVE_URL"] = "http://ray/generate"

    from monGARS.core.llm_integration import LLMIntegration

    called = {}

    async def fake_post(self, url, *, json=None, **_kwargs):
        called["url"] = url
        called["json"] = json

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "ray"}

        return Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    llm = LLMIntegration()
    result = await llm.generate_response("hello")

    assert called["url"] == "http://ray/generate"
    ray_prompt = called["json"]["prompt"]
    assert ray_prompt.startswith("<|begin_of_text|>")
    assert "<|system|>" in ray_prompt
    assert "<|user|>" in ray_prompt
    assert "<|assistant|>" in ray_prompt
    assert result["text"] == "ray"


@pytest.mark.asyncio
async def test_llm_integration_falls_back_to_local_on_ray_failure(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("USE_RAY_SERVE", "true")
    monkeypatch.setenv("RAY_SERVE_URL", "http://ray/generate")

    from monGARS.core.llm_integration import LLMIntegration

    class FailingClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, *, json: dict[str, object]) -> httpx.Response:
            raise httpx.RequestError("boom", request=httpx.Request("POST", url))

    async def fake_local(self, prompt: str, task_type: str) -> dict[str, str]:
        fake_local.called = True
        return {"content": "local"}

    fake_local.called = False

    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)
    monkeypatch.setattr(
        LLMIntegration,
        "_call_local_provider",
        fake_local,
        raising=False,
    )

    llm = LLMIntegration()

    result = await llm.generate_response("prompt", task_type="coding")

    assert fake_local.called is True
    assert result["text"] == "local"


@pytest.mark.asyncio
async def test_llm_integration_local_fallback_on_ray_error_payload(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("USE_RAY_SERVE", "true")
    monkeypatch.setenv("RAY_SERVE_URL", "http://ray/generate")

    from monGARS.core.llm_integration import LLMIntegration

    class ErroringClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, *, json: dict[str, object]) -> httpx.Response:
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                content=b'{"error": "not_ready", "detail": "scaling"}',
            )

    async def fake_local(self, prompt: str, task_type: str) -> dict[str, str]:
        fake_local.called = True
        return {"content": f"local-{task_type}"}

    fake_local.called = False

    monkeypatch.setattr(httpx, "AsyncClient", ErroringClient)
    monkeypatch.setattr(
        LLMIntegration,
        "_call_local_provider",
        fake_local,
        raising=False,
    )

    llm = LLMIntegration()

    result = await llm.generate_response("prompt")

    assert fake_local.called is True
    assert result["text"] == "local-general"
