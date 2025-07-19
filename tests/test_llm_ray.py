import os

import httpx
import ollama
import pytest


@pytest.mark.asyncio
async def test_llm_integration_uses_ray(monkeypatch):
    os.environ.setdefault("SECRET_KEY", "test-secret")
    os.environ["USE_RAY_SERVE"] = "True"
    os.environ["RAY_SERVE_URL"] = "http://ray/generate"

    from monGARS.core.llm_integration import LLMIntegration

    called = {}

    async def fake_post(self, url, json=None, timeout=10):
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
    assert called["json"]["prompt"] == "hello"
    assert result["text"] == "ray"


@pytest.mark.asyncio
async def test_llm_integration_sends_task_type(monkeypatch):
    os.environ["USE_RAY_SERVE"] = "True"
    os.environ["RAY_SERVE_URL"] = "http://ray/generate"

    from monGARS.core.llm_integration import LLMIntegration

    called = {}

    async def fake_post(self, url, json=None, timeout=10):
        called["json"] = json

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "ray"}

        return Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    llm = LLMIntegration()
    await llm.generate_response("hi", task_type="coding")

    assert called["json"]["task_type"] == "coding"


@pytest.mark.asyncio
async def test_llm_ray_failure_returns_fallback(monkeypatch):
    os.environ["USE_RAY_SERVE"] = "True"
    os.environ["RAY_SERVE_URL"] = "http://ray/generate"

    from monGARS.core.llm_integration import LLMIntegration

    async def fake_post(self, url, json=None, timeout=10):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    llm = LLMIntegration()
    result = await llm.generate_response("fail")

    assert result["text"] == "Ray Serve unavailable."


@pytest.mark.asyncio
async def test_llm_fallback_to_ollama(monkeypatch):
    os.environ.pop("USE_RAY_SERVE", None)

    from monGARS.core.llm_integration import LLMIntegration

    called = {}

    async def fake_chat(*args, **kwargs):
        called["model"] = kwargs.get("model")
        return {"content": "ollama"}

    async def fake_post(self, url, json=None, timeout=10):
        raise AssertionError("Ray Serve should not be used")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(ollama, "chat", fake_chat)

    llm = LLMIntegration()
    result = await llm.generate_response("hello ollama")

    assert "ollama" in result["text"]
