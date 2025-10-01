import json

import pytest

from monGARS.config import get_settings
from monGARS.core import model_manager
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
