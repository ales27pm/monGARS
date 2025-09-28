import asyncio

import httpx
import pytest

from modules.neurons.registry import update_manifest


def _write_summary(tmp_path, run_name: str) -> dict[str, object]:
    adapter_dir = tmp_path / run_name / "adapter"
    adapter_dir.mkdir(parents=True)
    weights_path = adapter_dir / "weights.json"
    weights_path.write_text(f'{{"run": "{run_name}"}}')
    return {
        "status": "success",
        "artifacts": {
            "adapter": adapter_dir.as_posix(),
            "weights": weights_path.as_posix(),
        },
    }


@pytest.mark.asyncio
async def test_llm_integration_refreshes_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("USE_RAY_SERVE", "True")
    monkeypatch.setenv("RAY_SERVE_URL", "http://ray/generate")
    monkeypatch.setenv("LLM_ADAPTER_REGISTRY_PATH", tmp_path.as_posix())

    update_manifest(tmp_path, _write_summary(tmp_path, "first"))

    captured: list[dict[str, object]] = []

    async def fake_post(self, url, *, json=None, **_kwargs):
        captured.append(json)

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "ray"}

        return Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    from monGARS.core.llm_integration import LLMIntegration

    llm = LLMIntegration()
    result_first = await llm.generate_response("bonjour le monde")
    assert result_first["text"].startswith("ray")
    assert captured and captured[0]["adapter"]["version"]
    first_version = captured[0]["adapter"]["version"]

    await asyncio.sleep(1.1)
    update_manifest(tmp_path, _write_summary(tmp_path, "second"))

    captured.clear()
    result_second = await llm.generate_response("nouvelle demande")
    assert result_second["text"].startswith("ray")
    assert captured
    second_version = captured[0]["adapter"]["version"]
    assert second_version != first_version
    assert captured[0]["adapter"]["adapter_path"].endswith("second/adapter")


@pytest.mark.asyncio
async def test_llm_integration_ignores_corrupt_manifest_without_ray(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("USE_RAY_SERVE", "False")
    monkeypatch.setenv("LLM_ADAPTER_REGISTRY_PATH", tmp_path.as_posix())

    manifest_path = tmp_path / "adapter_manifest.json"
    manifest_path.write_text('{"current": ')

    class DummyOllama:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat(
            self,
            *,
            model: str,
            messages: list[dict[str, str]],
            options: dict[str, object],
        ) -> dict[str, object]:
            self.calls.append(
                {"model": model, "messages": messages, "options": options}
            )
            return {"message": {"content": "local"}}

    from monGARS.core import llm_integration

    dummy = DummyOllama()
    monkeypatch.setattr(llm_integration, "ollama", dummy)

    llm = llm_integration.LLMIntegration()
    result = await llm.generate_response("bonjour")

    assert result["text"] == "local"
    assert dummy.calls
    assert llm._current_adapter_version == "baseline"
