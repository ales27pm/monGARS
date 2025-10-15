import hashlib
import io
import json

import pytest

from monGARS.config import get_settings
from monGARS.core import adapter_provisioner, model_manager
from monGARS.core.model_manager import LLMModelManager


def _write_config(path, data):
    path.write_text(json.dumps(data))
    return path


def _build_settings(**overrides):
    base = get_settings()
    merged_overrides = {"llm_models_profile": "default", **overrides}
    return base.model_copy(update=merged_overrides)


def test_model_manager_loads_profile_from_config(tmp_path):
    config_data = {
        "profiles": {
            "research": {
                "models": {
                    "general": {
                        "name": "ollama/custom-general",
                        "parameters": {"num_predict": 256},
                    },
                    "coding": {
                        "name": "ollama/custom-coder",
                        "provider": "ollama",
                        "auto_download": "false",
                    },
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    settings = _build_settings(
        llm_models_config_path=config_path,
        llm_models_profile="research",
        llm_general_model="override/general",
    )

    manager = LLMModelManager(settings)

    general = manager.get_model_definition("general")
    assert general.name == "override/general"
    assert manager.get_model_parameters("general")["num_predict"] == 256

    coding = manager.get_model_definition("coding")
    assert coding.name == "ollama/custom-coder"
    assert coding.auto_download is False


def test_model_definition_string_entries_preserve_role(tmp_path):
    config_data = {
        "profiles": {
            "default": {
                "models": {
                    "summarisation": "ollama/summarise",
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    settings = _build_settings(llm_models_config_path=config_path)

    manager = LLMModelManager(settings)

    definition = manager.get_model_definition("summarisation")
    assert definition.role == "summarisation"
    assert definition.name == "ollama/summarise"


@pytest.mark.asyncio
async def test_model_manager_installs_missing_model(monkeypatch, tmp_path):
    config_data = {
        "profiles": {
            "default": {
                "models": {
                    "general": {"name": "custom/general"},
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    settings = _build_settings(llm_models_config_path=config_path)
    manager = LLMModelManager(settings)

    class FakeOllama:
        def __init__(self) -> None:
            self.models: set[str] = set()
            self.pulled: list[str] = []

        def list(self) -> dict[str, object]:
            return {"models": [{"name": name} for name in sorted(self.models)]}

        def pull(self, name: str) -> None:
            self.models.add(name)
            self.pulled.append(name)

    fake = FakeOllama()
    monkeypatch.setattr(model_manager, "ollama", fake)

    report = await manager.ensure_models_installed(["general"], force=True)
    assert report.statuses
    status = report.statuses[0]
    assert status.action == "installed"
    assert fake.pulled == ["custom/general"]


def test_default_profile_exposes_reasoning_role(tmp_path):
    settings = _build_settings(
        llm_adapter_registry_path=tmp_path / "registry",
    )
    manager = LLMModelManager(settings)

    reasoning = manager.get_model_definition("reasoning")
    assert reasoning.name == "dolphin3"
    assert reasoning.adapters
    assert reasoning.adapters[0].target == "dolphin3/reasoning/baseline"


@pytest.mark.asyncio
async def test_adapter_artifacts_copied_into_registry(monkeypatch, tmp_path):
    source_dir = tmp_path / "artifacts" / "adapter"
    source_dir.mkdir(parents=True)
    (source_dir / "adapter_model.safetensors").write_bytes(b"stub")
    config_data = {
        "profiles": {
            "default": {
                "models": {
                    "general": {
                        "name": "custom/general",
                        "adapters": [
                            {
                                "name": "custom-adapter",
                                "source": str(source_dir),
                                "target": "custom/general",
                            }
                        ],
                    }
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    registry_path = tmp_path / "registry"
    settings = _build_settings(
        llm_models_config_path=config_path,
        llm_adapter_registry_path=registry_path,
    )
    manager = LLMModelManager(settings)

    class FakeOllama:
        def __init__(self) -> None:
            self.models: set[str] = set()

        def list(self) -> dict[str, object]:
            return {"models": []}

        def pull(self, name: str) -> None:
            self.models.add(name)

    monkeypatch.setattr(model_manager, "ollama", FakeOllama())

    report = await manager.ensure_models_installed(["general"], force=True)
    adapter_status = next(
        status for status in report.statuses if status.provider == "adapter"
    )
    assert adapter_status.action == "installed"
    installed_file = registry_path / "custom" / "general" / "adapter_model.safetensors"
    assert installed_file.exists()


@pytest.mark.asyncio
async def test_adapter_downloads_remote_payload(monkeypatch, tmp_path):
    remote_url = "https://example.com/adapter_model.safetensors"
    remote_payload = b"remote-adapter"

    class FakeResponse(io.BytesIO):
        def __enter__(self):  # pragma: no cover - context protocol
            self.seek(0)
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - context protocol
            self.close()

    def fake_urlopen(url):
        assert url == remote_url
        return FakeResponse(remote_payload)

    monkeypatch.setattr(adapter_provisioner, "urlopen", fake_urlopen)

    config_data = {
        "profiles": {
            "default": {
                "models": {
                    "general": {
                        "name": "custom/general",
                        "adapters": [
                            {
                                "name": "remote-adapter",
                                "source": remote_url,
                                "target": "custom/general",
                            }
                        ],
                    }
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    registry_path = tmp_path / "registry"
    settings = _build_settings(
        llm_models_config_path=config_path,
        llm_adapter_registry_path=registry_path,
    )
    manager = LLMModelManager(settings)

    class FakeOllama:
        def list(self) -> dict[str, object]:
            return {"models": []}

        def pull(self, name: str) -> None:  # pragma: no cover - defensive stub
            pass

    monkeypatch.setattr(model_manager, "ollama", FakeOllama())

    report = await manager.ensure_models_installed(["general"], force=True)
    adapter_status = next(
        status for status in report.statuses if status.provider == "adapter"
    )
    assert adapter_status.action == "installed"
    installed_file = registry_path / "custom" / "general" / "adapter_model.safetensors"
    assert installed_file.exists()
    assert installed_file.read_bytes() == remote_payload


@pytest.mark.asyncio
async def test_model_manager_skips_download_when_auto_disabled(monkeypatch, tmp_path):
    config_data = {
        "profiles": {
            "default": {
                "models": {
                    "general": {
                        "name": "custom/general",
                        "auto_download": False,
                    }
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    settings = _build_settings(
        llm_models_config_path=config_path,
        llm_models_auto_download=False,
    )
    manager = LLMModelManager(settings)

    class FakeOllama:
        def __init__(self) -> None:
            self.models: set[str] = set()
            self.pull_called = False

        def list(self) -> dict[str, object]:
            return {"models": [{"name": name} for name in sorted(self.models)]}

        def pull(self, _name: str) -> None:
            self.pull_called = True

    fake = FakeOllama()
    monkeypatch.setattr(model_manager, "ollama", fake)

    report = await manager.ensure_models_installed(["general"], force=True)
    status = report.statuses[0]
    assert status.action == "skipped"
    assert status.detail == "auto_download_disabled"
    assert fake.pull_called is False


@pytest.mark.asyncio
async def test_adapter_provision_missing_source(monkeypatch, tmp_path):
    config_data = {
        "profiles": {
            "default": {
                "models": {
                    "general": {
                        "name": "custom/general",
                        "adapters": [
                            {
                                "name": "custom-adapter",
                                "source": str(tmp_path / "missing" / "adapter.bin"),
                                "target": "custom/general",
                            }
                        ],
                    }
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    registry_path = tmp_path / "registry"
    settings = _build_settings(
        llm_models_config_path=config_path,
        llm_adapter_registry_path=registry_path,
    )
    manager = LLMModelManager(settings)

    class FakeOllama:
        def list(self) -> dict[str, object]:
            return {"models": []}

        def pull(self, name: str) -> None:  # pragma: no cover - defensive stub
            pass

    monkeypatch.setattr(model_manager, "ollama", FakeOllama())

    report = await manager.ensure_models_installed(["general"], force=True)
    adapter_status = next(
        status for status in report.statuses if status.provider == "adapter"
    )
    assert adapter_status.action == "error"
    assert adapter_status.detail == "source_missing"


@pytest.mark.asyncio
async def test_adapter_provision_checksum_mismatch(monkeypatch, tmp_path):
    source_dir = tmp_path / "artifacts"
    source_dir.mkdir(parents=True)
    adapter_path = source_dir / "adapter_model.safetensors"
    adapter_path.write_bytes(b"valid")
    bad_checksum = hashlib.sha256(b"invalid").hexdigest()

    config_data = {
        "profiles": {
            "default": {
                "models": {
                    "general": {
                        "name": "custom/general",
                        "adapters": [
                            {
                                "name": "custom-adapter",
                                "source": str(adapter_path),
                                "checksum": bad_checksum,
                                "target": "custom/general",
                            }
                        ],
                    }
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    registry_path = tmp_path / "registry"
    settings = _build_settings(
        llm_models_config_path=config_path,
        llm_adapter_registry_path=registry_path,
    )
    manager = LLMModelManager(settings)

    class FakeOllama:
        def list(self) -> dict[str, object]:
            return {"models": []}

        def pull(self, name: str) -> None:  # pragma: no cover - defensive stub
            pass

    monkeypatch.setattr(model_manager, "ollama", FakeOllama())

    report = await manager.ensure_models_installed(["general"], force=True)
    adapter_status = next(
        status for status in report.statuses if status.provider == "adapter"
    )
    assert adapter_status.action == "error"
    assert adapter_status.detail == "checksum_mismatch"


@pytest.mark.asyncio
async def test_adapter_provision_extraction_failure(monkeypatch, tmp_path):
    source_dir = tmp_path / "artifacts"
    source_dir.mkdir(parents=True)
    adapter_path = source_dir / "adapter_model.safetensors"
    adapter_path.write_bytes(b"valid")

    config_data = {
        "profiles": {
            "default": {
                "models": {
                    "general": {
                        "name": "custom/general",
                        "adapters": [
                            {
                                "name": "custom-adapter",
                                "source": str(adapter_path),
                                "target": "custom/general",
                            }
                        ],
                    }
                }
            }
        }
    }
    config_path = _write_config(tmp_path / "models.json", config_data)
    registry_path = tmp_path / "registry"
    settings = _build_settings(
        llm_models_config_path=config_path,
        llm_adapter_registry_path=registry_path,
    )
    manager = LLMModelManager(settings)

    class FakeOllama:
        def list(self) -> dict[str, object]:
            return {"models": []}

        def pull(self, name: str) -> None:  # pragma: no cover - defensive stub
            pass

    monkeypatch.setattr(model_manager, "ollama", FakeOllama())

    def fail_install(self, source_path, target_path):
        raise RuntimeError("Extraction failed")

    monkeypatch.setattr(
        adapter_provisioner.AdapterProvisioner,
        "_install_from_filesystem",
        fail_install,
    )

    report = await manager.ensure_models_installed(["general"], force=True)
    adapter_status = next(
        status for status in report.statuses if status.provider == "adapter"
    )
    assert adapter_status.action == "error"
    assert adapter_status.detail == "adapter_install_failed"
