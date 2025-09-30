import json
from pathlib import Path

import pytest

from modules.evolution_engine.orchestrator import EvolutionOrchestrator
from modules.neurons.registry import MANIFEST_FILENAME, load_manifest
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


def test_orchestrator_surfaces_training_failure(temp_dir: Path) -> None:
    class StubTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            config_payload = {
                "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
            }
            (self.output_dir / "training_config.json").write_text(
                json.dumps(config_payload)
            )

            adapter_dir = self.output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            weights_payload = {
                "rows": 4,
                "cols": 8,
                "matrix": [[0 for _ in range(8)] for _ in range(4)],
            }
            weights_path = adapter_dir / "fallback_adapter.json"
            weights_path.write_text(json.dumps(weights_payload))

            self.summary = {
                "status": TrainingStatus.FALLBACK.value,
                "artifacts": {
                    "adapter": str(adapter_dir),
                    "weights": str(weights_path),
                },
                "metrics": {"training_examples": 0},
                "reason": "missing_dependencies",
            }
            (self.output_dir / "training_summary.json").write_text(
                json.dumps(self.summary)
            )

        def train(self) -> dict[str, object]:
            return self.summary

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=StubTrainer
    )

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.trigger_encoder_training_pipeline()
    assert "unsuccessful" in str(excinfo.value)

    manifest_path = temp_dir / MANIFEST_FILENAME
    assert not manifest_path.exists()

    run_dirs = [path for path in temp_dir.iterdir() if path.is_dir()]
    assert run_dirs, "Expected orchestrator to create an output directory"
    out = run_dirs[0]

    cfg_file = out / "training_config.json"
    assert cfg_file.exists()
    data = json.loads(cfg_file.read_text())
    assert data["model_name_or_path"] == "mistralai/Mistral-7B-Instruct-v0.2"

    _assert_fallback_artifacts(out)


def test_orchestrator_rejects_weights_outside_run(
    temp_dir: Path, tmp_path: Path
) -> None:
    rogue_weights = tmp_path / "rogue.bin"
    rogue_weights.write_bytes(b"1")

    class StubTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            adapter_dir = self.output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}")
            (adapter_dir / "adapter_model.bin").write_bytes(b"0")
            self.summary = {
                "status": TrainingStatus.SUCCESS.value,
                "artifacts": {
                    "adapter": str(adapter_dir),
                    "weights": str(rogue_weights),
                },
                "metrics": {"training_examples": 1},
            }

        def train(self) -> dict[str, object]:
            return self.summary

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=StubTrainer
    )

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.trigger_encoder_training_pipeline()
    assert "weights outside orchestrator output directory" in str(excinfo.value)


def test_mntp_trainer_generates_deterministic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MNTPTrainer, "_deps_available", lambda self: False, raising=False
    )
    output_dir = tmp_path / "first"
    trainer = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(output_dir),
    )
    summary = trainer.train()
    assert summary["status"] == TrainingStatus.FALLBACK.value
    assert summary["reason"] == "missing_dependencies"
    artifacts = summary["artifacts"]
    assert set(artifacts) == {"adapter", "weights"}
    assert artifacts["adapter"].startswith(str(output_dir))

    weights = _assert_fallback_artifacts(output_dir)

    second_dir = tmp_path / "second"
    trainer_repeat = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(second_dir),
    )
    repeat_summary = trainer_repeat.train()
    assert repeat_summary["status"] == TrainingStatus.FALLBACK.value
    assert repeat_summary["reason"] == "missing_dependencies"
    repeat_artifacts = repeat_summary["artifacts"]
    assert set(repeat_artifacts) == {"adapter", "weights"}
    assert repeat_artifacts["adapter"].startswith(str(second_dir))

    repeat_weights = _assert_fallback_artifacts(second_dir)
    assert repeat_weights == weights


def test_mntp_trainer_recovers_from_training_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_deps_available(self) -> bool:
        return True

    def fail_training(self):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        MNTPTrainer, "_deps_available", fake_deps_available, raising=False
    )
    monkeypatch.setattr(MNTPTrainer, "_run_peft_training", fail_training, raising=False)

    output_dir = tmp_path / "failed"
    trainer = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(output_dir),
    )

    summary = trainer.train()

    assert summary["status"] == TrainingStatus.FALLBACK.value
    assert summary["reason"] == "training_failed"
    assert summary["details"] == "synthetic failure"
    _assert_fallback_artifacts(output_dir)


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


def test_persist_model_validates_artifacts(tmp_path: Path) -> None:
    trainer = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(tmp_path / "persist_success"),
    )

    class DummyModel:
        def save_pretrained(self, path: str) -> None:
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "adapter_config.json").write_text("{}")
            (target / "adapter_model.bin").write_bytes(b"0")

    adapter_dir, weights_path = trainer._persist_model(DummyModel())
    assert adapter_dir.exists()
    assert weights_path.exists()


def test_persist_model_raises_when_weights_missing(tmp_path: Path) -> None:
    trainer = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(tmp_path / "persist_failure"),
    )

    class BrokenModel:
        def save_pretrained(self, path: str) -> None:
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "adapter_config.json").write_text("{}")

    with pytest.raises(RuntimeError):
        trainer._persist_model(BrokenModel())
