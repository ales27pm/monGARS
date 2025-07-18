import json
import shutil
from pathlib import Path

import pytest

from modules.evolution_engine.orchestrator import EvolutionOrchestrator


@pytest.fixture()
def temp_dir(tmp_path: Path):
    d = tmp_path / "encoders"
    d.mkdir()
    yield d
    shutil.rmtree(d)


def test_orchestrator_creates_encoder(temp_dir: Path, monkeypatch):
    orchestrator = EvolutionOrchestrator(model_registry_path=str(temp_dir))
    path = orchestrator.trigger_encoder_training_pipeline()
    out = Path(path)
    assert out.exists()
    cfg_file = out / "training_config.json"
    assert cfg_file.exists()
    data = json.loads(cfg_file.read_text())
    assert data["model_name_or_path"] == "mistralai/Mistral-7B-Instruct-v0.2"
