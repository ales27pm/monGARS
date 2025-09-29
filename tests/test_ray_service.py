import httpx
import pytest


@pytest.mark.asyncio
async def test_ray_service_render_response_uses_ollama(monkeypatch):
    from modules import ray_service

    deployment = ray_service.RayLLMDeployment(base_model_path="base")

    called = {}

    class FakeOllama:
        @staticmethod
        def chat(
            *, model: str, messages: list[dict[str, str]], options: dict[str, float]
        ):
            called["model"] = model
            called["messages"] = messages
            called["options"] = options
            return {"message": {"content": "ok"}, "usage": {"eval_count": 1}}

    monkeypatch.setattr(ray_service, "ollama", FakeOllama())

    result = await deployment._render_response(
        "prompt",
        [[0.1, 0.2, 0.3]],
        {"version": "v1", "adapter_path": "/tmp"},
        "general",
    )

    assert result["content"] == "ok"
    assert result["message"]["content"] == "ok"
    assert result["usage"]["model"] == deployment.general_model
    assert called["messages"][0]["content"] == "prompt"


def test_ray_service_encode_prompt_error(monkeypatch):
    from modules import ray_service

    deployment = ray_service.RayLLMDeployment(base_model_path="base")

    def failing_encode(_):
        raise RuntimeError("fail")

    monkeypatch.setattr(deployment.neuron_manager, "encode", failing_encode)

    with pytest.raises(ray_service.RayServeException):
        deployment._encode_prompt("prompt")


@pytest.mark.asyncio
async def test_llm_integration_balances_ray_endpoints(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv("USE_RAY_SERVE", "True")
    monkeypatch.setenv(
        "RAY_SERVE_URL", "http://ray-one/generate,http://ray-two/generate"
    )

    from monGARS.core.llm_integration import LLMIntegration

    llm = LLMIntegration()

    called_urls: list[str] = []

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("monGARS.core.llm_integration.asyncio.sleep", fake_sleep)

    class DummyClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, *, json: dict[str, object]) -> httpx.Response:
            called_urls.append(url)
            if "ray-one" in url and len(called_urls) == 1:
                return httpx.Response(
                    503,
                    request=httpx.Request("POST", url),
                    content=b"scaling",
                    headers={"retry-after": "0"},
                )
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                content=b'{"content": "ray"}',
            )

    monkeypatch.setattr("monGARS.core.llm_integration.httpx.AsyncClient", DummyClient)

    result = await llm._ray_call("hello", "general", None)

    assert result["content"] == "ray"
    assert called_urls.count("http://ray-one/generate") == 1
    assert called_urls.count("http://ray-two/generate") >= 1
