import json
from pathlib import Path

import pytest

from modules.evolution_engine.orchestrator import EvolutionOrchestrator
from modules.neurons.training.mntp_trainer import MNTPTrainer


@pytest.fixture()
def temp_dir(tmp_path: Path):
    d = tmp_path / "encoders"
    d.mkdir()
    yield d


def test_orchestrator_creates_encoder(temp_dir: Path, monkeypatch):
    orchestrator = EvolutionOrchestrator(model_registry_path=str(temp_dir))
    path = orchestrator.trigger_encoder_training_pipeline()
    out = Path(path)
    assert out.exists()

    cfg_file = out / "training_config.json"
    assert cfg_file.exists()
    data = json.loads(cfg_file.read_text())
    assert data["model_name_or_path"] == "mistralai/Mistral-7B-Instruct-v0.2"

    summary_path = out / "training_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "fallback"
    weights_path = out / "adapter" / "fallback_adapter.json"
    assert weights_path.exists()
    weights = json.loads(weights_path.read_text())
    assert weights["rows"] >= 4
    assert weights["cols"] >= 8
    assert weights["matrix"]


def test_mntp_trainer_generates_deterministic_fallback(tmp_path: Path) -> None:
    output_dir = tmp_path / "first"
    trainer = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(output_dir),
    )
    summary = trainer.train()
    assert summary["status"] == "fallback"

    weights_path = output_dir / "adapter" / "fallback_adapter.json"
    weights = json.loads(weights_path.read_text())

    second_dir = tmp_path / "second"
    trainer_repeat = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(second_dir),
    )
    repeat_summary = trainer_repeat.train()
    assert repeat_summary["status"] == "fallback"

    repeat_weights_path = second_dir / "adapter" / "fallback_adapter.json"
    repeat_weights = json.loads(repeat_weights_path.read_text())
    assert repeat_weights == weights
