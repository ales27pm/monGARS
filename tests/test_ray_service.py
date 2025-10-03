from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from modules.evolution_engine.orchestrator import EvolutionOrchestrator
from modules.neurons.registry import MANIFEST_FILENAME, load_manifest
from modules.neurons.training.mntp_trainer import TrainingStatus


class DummyNeuronManager:
    """Minimal fake neuron manager used to observe adapter switches in tests."""

    instances: list["DummyNeuronManager"] = []

    def __init__(
        self,
        base_model_path: str,
        default_encoder_path: str | None = None,
        **_: Any,
    ) -> None:
        self.base_model_path = base_model_path
        self.encoder_path = default_encoder_path
        self.switch_calls: list[str] = []
        DummyNeuronManager.instances.append(self)

    def switch_encoder(self, path: str) -> None:
        self.encoder_path = path
        self.switch_calls.append(path)

    def encode(self, prompts: list[str]) -> list[list[float]]:
        return [[0.1, 0.1, 0.1] for _ in prompts]


def _make_success_trainer(*, suffix: str) -> type:
    class SuccessTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            try:
                config_payload = json.loads(Path(training_config_path).read_text())
            except FileNotFoundError:
                config_payload = {"model_name_or_path": "stub"}
            (self.output_dir / "training_config.json").write_text(
                json.dumps(config_payload)
            )

            adapter_dir = self.output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            weights_path = adapter_dir / f"adapter-{suffix}.bin"
            weights_path.write_bytes(f"weights-{suffix}".encode("utf-8"))

            self.summary = {
                "status": TrainingStatus.SUCCESS.value,
                "artifacts": {
                    "adapter": str(adapter_dir),
                    "weights": str(weights_path),
                },
                "metrics": {"training_examples": 1, "run": suffix},
            }

        def train(self) -> dict[str, Any]:
            return self.summary

    return SuccessTrainer


def _run_orchestrator_pipeline(registry_path: Path, trainer_cls: type) -> Path:
    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(registry_path), trainer_cls=trainer_cls
    )
    return Path(orchestrator.trigger_encoder_training_pipeline())


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


def test_deploy_ray_service_raises_when_ray_missing(monkeypatch):
    from modules import ray_service

    monkeypatch.setattr(ray_service, "serve", None)
    monkeypatch.setattr(ray_service, "ray", None)

    with pytest.raises(RuntimeError) as excinfo:
        ray_service.deploy_ray_service()

    assert "Ray Serve is not available" in str(excinfo.value)


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


def test_llm_integration_negative_backoff_entries_are_ignored(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv("USE_RAY_SERVE", "True")
    monkeypatch.setenv("RAY_SERVE_URL", "http://ray/generate")
    monkeypatch.setenv("RAY_SCALING_BACKOFF", "1.5, -2, 3.0")

    from monGARS.core.llm_integration import LLMIntegration

    llm = LLMIntegration()

    assert llm._ray_scaling_backoff == [1.5, 3.0]


def test_llm_integration_invalid_backoff_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv("USE_RAY_SERVE", "True")
    monkeypatch.setenv("RAY_SERVE_URL", "http://ray/generate")
    monkeypatch.setenv("RAY_SCALING_BACKOFF", "oops, 1.0")

    from monGARS.core.llm_integration import LLMIntegration

    llm = LLMIntegration()

    assert llm._ray_scaling_backoff == [0.5, 1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_ray_deployment_refreshes_after_training_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modules import ray_service

    registry_path = tmp_path / "encoders"
    first_trainer = _make_success_trainer(suffix="first")
    first_run = _run_orchestrator_pipeline(registry_path, first_trainer)

    manifest = load_manifest(registry_path)
    assert manifest is not None and manifest.current is not None
    first_payload = manifest.build_payload()

    DummyNeuronManager.instances.clear()
    monkeypatch.setattr(ray_service, "NeuronManager", DummyNeuronManager)

    deployment = ray_service.RayLLMDeployment(
        base_model_path="base", registry_path=str(registry_path)
    )

    assert DummyNeuronManager.instances, "NeuronManager was not instantiated"
    manager = DummyNeuronManager.instances[-1]
    assert manager.encoder_path == first_payload["adapter_path"]
    assert deployment._adapter_payload == first_payload

    second_trainer = _make_success_trainer(suffix="second")
    second_run = _run_orchestrator_pipeline(registry_path, second_trainer)
    second_manifest = load_manifest(registry_path)
    assert second_manifest is not None and second_manifest.current is not None
    second_payload = second_manifest.build_payload()

    manifest_file = registry_path / MANIFEST_FILENAME
    stat_result = manifest_file.stat()
    os.utime(manifest_file, (stat_result.st_atime, stat_result.st_mtime + 1))

    refreshed = await deployment._refresh_adapter(None)

    assert manager.switch_calls, "Expected adapter switch to be triggered"
    assert manager.switch_calls[-1] == second_payload["adapter_path"]
    assert refreshed == second_payload
    assert deployment._adapter_version == second_payload["version"]
    assert second_payload["adapter_path"] == str(second_run / "adapter")


@pytest.mark.asyncio
async def test_ray_deployment_rejects_invalid_adapter_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modules import ray_service

    registry_path = tmp_path / "encoders"
    trainer_cls = _make_success_trainer(suffix="only")
    _run_orchestrator_pipeline(registry_path, trainer_cls)

    manifest = load_manifest(registry_path)
    assert manifest is not None and manifest.current is not None
    payload = manifest.build_payload()

    DummyNeuronManager.instances.clear()
    monkeypatch.setattr(ray_service, "NeuronManager", DummyNeuronManager)

    deployment = ray_service.RayLLMDeployment(
        base_model_path="base", registry_path=str(registry_path)
    )

    manager = DummyNeuronManager.instances[-1]
    assert not manager.switch_calls

    rogue_payload = {
        "adapter_path": str(tmp_path / "rogue"),
        "version": payload["version"],
    }
    result = await deployment._refresh_adapter(rogue_payload)
    assert result == payload
    assert not manager.switch_calls

    mismatched_version = {
        "adapter_path": payload["adapter_path"],
        "version": "unexpected-version",
    }
    result = await deployment._refresh_adapter(mismatched_version)
    assert result == payload
    assert not manager.switch_calls


@pytest.mark.asyncio
async def test_ray_deployment_accepts_multi_replica_payload_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modules import ray_service

    monkeypatch.setenv("SECRET_KEY", "test")
    registry_path = tmp_path / "encoders"
    initial_trainer = _make_success_trainer(suffix="baseline")
    _run_orchestrator_pipeline(registry_path, initial_trainer)

    DummyNeuronManager.instances.clear()
    monkeypatch.setattr(ray_service, "NeuronManager", DummyNeuronManager)

    deployment = ray_service.RayLLMDeployment(
        base_model_path="base", registry_path=str(registry_path)
    )

    manager = DummyNeuronManager.instances[-1]
    assert not manager.switch_calls

    updated_trainer = _make_success_trainer(suffix="multi")
    _run_orchestrator_pipeline(registry_path, updated_trainer)

    manifest = load_manifest(registry_path)
    assert manifest is not None and manifest.current is not None
    payload = manifest.build_payload()

    replica_payload = {
        "adapter_path": payload["adapter_path"],
        "version": payload["version"],
    }

    results = await asyncio.gather(
        deployment._refresh_adapter(replica_payload),
        deployment._refresh_adapter(replica_payload),
    )

    assert all(result == replica_payload for result in results)
    assert manager.switch_calls.count(replica_payload["adapter_path"]) == 1
    assert deployment._adapter_version == payload["version"]
