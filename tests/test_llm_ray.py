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
    assert called["json"]["prompt"] == "hello"
    assert result["text"] == "ray"
