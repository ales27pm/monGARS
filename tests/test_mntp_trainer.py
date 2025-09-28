import json
from pathlib import Path

import pytest

from modules.evolution_engine.orchestrator import EvolutionOrchestrator
from modules.neurons.training.mntp_trainer import MNTPTrainer, TrainingStatus


@pytest.fixture()
def temp_dir(tmp_path: Path):
    d = tmp_path / "encoders"
    d.mkdir()
    yield d


def _assert_fallback_artifacts(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "training_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == TrainingStatus.FALLBACK.value

    weights_path = output_dir / "adapter" / "fallback_adapter.json"
    assert weights_path.exists()
    weights = json.loads(weights_path.read_text())
    assert weights["rows"] >= 4
    assert weights["cols"] >= 8
    assert weights["matrix"]
    return weights


def test_orchestrator_creates_encoder(temp_dir: Path) -> None:
    orchestrator = EvolutionOrchestrator(model_registry_path=str(temp_dir))
    path = orchestrator.trigger_encoder_training_pipeline()
    out = Path(path)
    assert out.exists()

    cfg_file = out / "training_config.json"
    assert cfg_file.exists()
    data = json.loads(cfg_file.read_text())
    assert data["model_name_or_path"] == "mistralai/Mistral-7B-Instruct-v0.2"

    _assert_fallback_artifacts(out)


def test_mntp_trainer_generates_deterministic_fallback(tmp_path: Path) -> None:
    output_dir = tmp_path / "first"
    trainer = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(output_dir),
    )
    summary = trainer.train()
    assert summary["status"] == TrainingStatus.FALLBACK.value

    weights = _assert_fallback_artifacts(output_dir)

    second_dir = tmp_path / "second"
    trainer_repeat = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(second_dir),
    )
    repeat_summary = trainer_repeat.train()
    assert repeat_summary["status"] == TrainingStatus.FALLBACK.value

    repeat_weights = _assert_fallback_artifacts(second_dir)
    assert repeat_weights == weights


def test_mntp_trainer_missing_config_file(tmp_path: Path) -> None:
    missing_config_path = tmp_path / "missing_config.json"
    trainer = MNTPTrainer(
        training_config_path=str(missing_config_path),
        output_dir=str(tmp_path / "output_missing"),
    )

    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_mntp_trainer_invalid_json(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{invalid_json: true}")
    trainer = MNTPTrainer(
        training_config_path=str(invalid_config),
        output_dir=str(tmp_path / "output_invalid"),
    )

    with pytest.raises(json.JSONDecodeError):
        trainer.train()


def test_mntp_trainer_invalid_numeric_fields(tmp_path: Path) -> None:
    bad_config = {
        "dataset_name": "wikitext",
        "model_name_or_path": "sshleifer/tiny-gpt2",
        "lora_r": "not-an-int",
    }
    bad_config_path = tmp_path / "bad_config.json"
    bad_config_path.write_text(json.dumps(bad_config))

    trainer = MNTPTrainer(
        training_config_path=str(bad_config_path),
        output_dir=str(tmp_path / "output_bad_numeric"),
    )

    with pytest.raises(ValueError):
        trainer.train()
