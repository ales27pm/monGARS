import json
from pathlib import Path
from uuid import UUID

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


def test_orchestrator_updates_manifest_on_success(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_uuid = UUID("00000000-0000-0000-0000-00000000abcd")
    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.uuid4", lambda: fixed_uuid
    )

    class SuccessfulTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            config_payload = {
                "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2"
            }
            (self.output_dir / "training_config.json").write_text(
                json.dumps(config_payload)
            )

            self.adapter_dir = self.output_dir / "adapter"
            self.adapter_dir.mkdir(parents=True, exist_ok=True)
            self.weights_path = self.adapter_dir / "adapter.bin"
            self.weights_path.write_bytes(b"weights")

            self.summary = {
                "status": TrainingStatus.SUCCESS.value,
                "artifacts": {
                    "adapter": str(self.adapter_dir),
                    "weights": str(self.weights_path),
                },
                "metrics": {"training_examples": 4, "loss": 0.1},
            }
            (self.output_dir / "training_summary.json").write_text(
                json.dumps(self.summary)
            )

        def train(self) -> dict[str, object]:
            return self.summary

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=SuccessfulTrainer
    )

    run_path = Path(orchestrator.trigger_encoder_training_pipeline())
    assert run_path.exists()
    assert run_path.name == f"temp-mistral-mntp-step-{fixed_uuid}"

    manifest = load_manifest(temp_dir)
    assert manifest is not None
    assert manifest.current is not None
    assert manifest.current.status == TrainingStatus.SUCCESS.value
    assert manifest.current.summary["metrics"]["training_examples"] == 4

    adapter_dir = manifest.current.resolve_adapter_path(Path(temp_dir))
    weights_path = manifest.current.resolve_weights_path(Path(temp_dir))
    assert adapter_dir == run_path / "adapter"
    assert weights_path == run_path / "adapter" / "adapter.bin"

    latest_link = Path(temp_dir) / "latest"
    if latest_link.exists():
        assert latest_link.resolve() == adapter_dir.resolve()


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


def test_orchestrator_requires_adapter_artifact(temp_dir: Path) -> None:
    class NoAdapterTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.summary = {
                "status": TrainingStatus.SUCCESS.value,
                "artifacts": {},
                "metrics": {},
            }

        def train(self) -> dict[str, object]:
            return self.summary

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=NoAdapterTrainer
    )

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.trigger_encoder_training_pipeline()
    assert "did not return an adapter artifact" in str(excinfo.value)


def test_orchestrator_validates_adapter_path(temp_dir: Path) -> None:
    class MissingAdapterTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.summary = {
                "status": TrainingStatus.SUCCESS.value,
                "artifacts": {
                    "adapter": str(self.output_dir / "adapter" / "missing"),
                },
                "metrics": {},
            }

        def train(self) -> dict[str, object]:
            return self.summary

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=MissingAdapterTrainer
    )

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.trigger_encoder_training_pipeline()
    assert "does not exist" in str(excinfo.value)


def test_orchestrator_propagates_manifest_failures(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SuccessfulTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.adapter_dir = self.output_dir / "adapter"
            self.adapter_dir.mkdir(parents=True, exist_ok=True)
            self.weights_path = self.adapter_dir / "adapter.bin"
            self.weights_path.write_bytes(b"weights")
            self.summary = {
                "status": TrainingStatus.SUCCESS.value,
                "artifacts": {
                    "adapter": str(self.adapter_dir),
                    "weights": str(self.weights_path),
                },
                "metrics": {"training_examples": 1},
            }

        def train(self) -> dict[str, object]:
            return self.summary

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=SuccessfulTrainer
    )

    def explode_manifest(*args, **kwargs):
        raise RuntimeError("manifest write failed")

    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.update_manifest",
        explode_manifest,
    )

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.trigger_encoder_training_pipeline()
    assert "manifest write failed" in str(excinfo.value)

    manifest_path = temp_dir / MANIFEST_FILENAME
    assert not manifest_path.exists()


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


def test_mntp_trainer_curated_training_success(tmp_path: Path) -> None:
    output_dir = tmp_path / "curated"
    trainer = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(output_dir),
    )

    curated_records = [
        {
            "embedding": [0.2, 0.4, 0.6],
            "target": 0.8,
            "confidence": 0.9,
            "source_id": "record-1",
            "text_preview": "example",
        },
        {
            "embedding": [0.1, 0.3, 0.5],
            "target": 0.5,
            "confidence": 0.85,
            "source_id": "record-2",
            "text_preview": "second",
        },
    ]

    summary = trainer.train(curated_records=curated_records)

    assert summary["status"] == TrainingStatus.SUCCESS.value
    assert summary["mode"] == "curated_linear_adapter"
    assert summary["metrics"]["training_examples"] == len(curated_records)
    assert summary["metrics"]["feature_dimension"] == len(
        curated_records[0]["embedding"]
    )

    artifacts = summary["artifacts"]
    adapter_dir = Path(artifacts["adapter"])
    weights_path = Path(artifacts["weights"])
    assert adapter_dir.exists()
    assert weights_path.exists()

    payload = json.loads(weights_path.read_text())
    assert payload["feature_dimension"] == len(curated_records[0]["embedding"])
    expected_epochs = int(trainer.config.get("curated_epochs", 15))
    assert payload["metrics"]["epochs"] == expected_epochs
    assert len(payload["records"]) == len(curated_records)

    summary_path = output_dir / "training_summary.json"
    assert summary_path.exists()
    persisted_summary = json.loads(summary_path.read_text())
    assert persisted_summary["status"] == TrainingStatus.SUCCESS.value
