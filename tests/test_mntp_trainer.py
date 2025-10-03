import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from modules.evolution_engine.energy import EnergyUsageReport
from modules.evolution_engine.orchestrator import EvolutionOrchestrator
from modules.neurons.registry import MANIFEST_FILENAME, load_manifest
from modules.neurons.training.mntp_trainer import MNTPTrainer, TrainingStatus


@pytest.fixture()
def temp_dir(tmp_path: Path):
    d = tmp_path / "encoders"
    d.mkdir()
    yield d


@pytest.fixture()
def fixed_run_uuid(monkeypatch: pytest.MonkeyPatch) -> UUID:
    value = UUID("00000000-0000-0000-0000-00000000abcd")
    monkeypatch.setattr("modules.evolution_engine.orchestrator.uuid4", lambda: value)
    return value


@pytest.fixture()
def orchestrator_factory(temp_dir: Path):
    def _build_orchestrator(
        *, trainer_cls: type, **overrides: Any
    ) -> EvolutionOrchestrator:
        return EvolutionOrchestrator(
            model_registry_path=str(temp_dir), trainer_cls=trainer_cls, **overrides
        )

    return _build_orchestrator


@pytest.fixture()
def stub_energy_tracker():
    calls: list[str] = []

    class StubEnergyTracker:
        def __init__(self) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True
            calls.append("start")

        def stop(self) -> EnergyUsageReport:
            assert self.started, "energy tracker must start before stop"
            calls.append("stop")
            return EnergyUsageReport(
                energy_wh=4321.0,
                duration_seconds=7200.0,
                cpu_seconds=3600.0,
                baseline_cpu_power_watts=55.0,
                backend="stub-energy",
                emissions_grams=12.5,
                carbon_intensity_g_co2_per_kwh=420.0,
            )

    return StubEnergyTracker, calls


def _create_trainer_stub(
    *,
    summary_factory: Callable[[Path, dict[str, object]], dict[str, Any]],
    training_config: dict[str, Any] | None = None,
    setup: Callable[[Path], dict[str, object]] | None = None,
    persist_summary: bool = False,
) -> type:
    class StubTrainer:
        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            if training_config:
                (self.output_dir / "training_config.json").write_text(
                    json.dumps(training_config)
                )

            context: dict[str, object] = {}
            if setup is not None:
                context = setup(self.output_dir)

            self.summary = summary_factory(self.output_dir, context)
            artifacts = self.summary.setdefault("artifacts", {})
            weights_value = artifacts.get("weights")
            if weights_value and "weights_checksum" not in artifacts:
                weights_path = Path(weights_value)
                checksum = (
                    hashlib.sha256(weights_path.read_bytes()).hexdigest()
                    if weights_path.exists()
                    else "stub"
                )
                artifacts["weights_checksum"] = checksum
            if "version" not in self.summary and "weights_checksum" in artifacts:
                self.summary["version"] = artifacts["weights_checksum"]
            if persist_summary:
                (self.output_dir / "training_summary.json").write_text(
                    json.dumps(self.summary)
                )

        def train(self) -> dict[str, object]:
            return self.summary

    return StubTrainer


def _setup_fallback_artifacts(output_dir: Path) -> dict[str, object]:
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    weights_payload = {
        "rows": 4,
        "cols": 8,
        "matrix": [[0 for _ in range(8)] for _ in range(4)],
    }
    weights_path = adapter_dir / "fallback_adapter.json"
    weights_path.write_text(json.dumps(weights_payload))
    return {
        "adapter_dir": adapter_dir,
        "weights_path": weights_path,
    }


def _setup_binary_adapter(output_dir: Path) -> dict[str, object]:
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    weights_path = adapter_dir / "adapter.bin"
    weights_path.write_bytes(b"weights")
    return {"adapter_dir": adapter_dir, "weights_path": weights_path}


TRAINING_CONFIG_PAYLOAD = {"model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2"}


def _configure_manifest_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode_manifest(*_: object, **__: object) -> None:
        raise RuntimeError("manifest write failed")

    monkeypatch.setattr(
        "modules.evolution_engine.orchestrator.update_manifest",
        explode_manifest,
    )


def _assert_manifest_missing(temp_dir: Path) -> None:
    manifest_path = temp_dir / MANIFEST_FILENAME
    assert not manifest_path.exists()


def _assert_fallback_artifacts(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "training_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == TrainingStatus.FALLBACK.value
    assert "version" in summary
    artifacts = summary["artifacts"]
    assert artifacts["weights_checksum"] == summary["version"]

    weights_path = output_dir / "adapter" / "fallback_adapter.json"
    assert weights_path.exists()
    weights = json.loads(weights_path.read_text())
    assert weights["rows"] >= 4
    assert weights["cols"] >= 8
    assert weights["matrix"]
    return weights


def test_orchestrator_surfaces_training_failure(temp_dir: Path) -> None:
    fallback_trainer = _create_trainer_stub(
        training_config=TRAINING_CONFIG_PAYLOAD,
        setup=_setup_fallback_artifacts,
        persist_summary=True,
        summary_factory=lambda _, ctx: {
            "status": TrainingStatus.FALLBACK.value,
            "artifacts": {
                "adapter": str(ctx["adapter_dir"]),
                "weights": str(ctx["weights_path"]),
            },
            "metrics": {"training_examples": 0},
            "reason": "missing_dependencies",
        },
    )

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=fallback_trainer
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
    temp_dir: Path, fixed_run_uuid: UUID
) -> None:
    successful_trainer = _create_trainer_stub(
        training_config=TRAINING_CONFIG_PAYLOAD,
        setup=_setup_binary_adapter,
        persist_summary=True,
        summary_factory=lambda _, ctx: {
            "status": TrainingStatus.SUCCESS.value,
            "artifacts": {
                "adapter": str(ctx["adapter_dir"]),
                "weights": str(ctx["weights_path"]),
            },
            "metrics": {"training_examples": 4, "loss": 0.1},
        },
    )

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=successful_trainer
    )

    run_path = Path(orchestrator.trigger_encoder_training_pipeline())
    assert run_path.exists()
    assert run_path.name == f"temp-mistral-mntp-step-{fixed_run_uuid}"

    manifest = load_manifest(temp_dir)
    assert manifest is not None
    assert manifest.current is not None
    assert manifest.current.status == TrainingStatus.SUCCESS.value
    assert manifest.current.summary["metrics"]["training_examples"] == 4

    adapter_dir = manifest.current.resolve_adapter_path(Path(temp_dir))
    weights_path = manifest.current.resolve_weights_path(Path(temp_dir))
    assert adapter_dir == run_path / "adapter"
    assert weights_path == run_path / "adapter" / "adapter.bin"

    energy_report = run_path / "energy_report.json"
    assert energy_report.exists()
    energy_payload = json.loads(energy_report.read_text())
    assert "energy_wh" in energy_payload

    summary_payload = json.loads((run_path / "training_summary.json").read_text())
    assert summary_payload["metrics"]["energy_wh"] == pytest.approx(
        energy_payload["energy_wh"], rel=1e-3, abs=1e-4
    )

    latest_link = Path(temp_dir) / "latest"
    assert latest_link.exists(), "The 'latest' symlink was not created"
    assert latest_link.is_symlink()
    assert latest_link.resolve() == adapter_dir.resolve()


def test_orchestrator_rejects_weights_outside_run(
    temp_dir: Path, tmp_path: Path
) -> None:
    rogue_weights = tmp_path / "rogue.bin"
    rogue_weights.write_bytes(b"1")

    rogue_weights_trainer = _create_trainer_stub(
        setup=_setup_binary_adapter,
        summary_factory=lambda _, ctx: {
            "status": TrainingStatus.SUCCESS.value,
            "artifacts": {
                "adapter": str(ctx["adapter_dir"]),
                "weights": str(rogue_weights),
            },
            "metrics": {"training_examples": 1},
        },
    )

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=rogue_weights_trainer
    )

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.trigger_encoder_training_pipeline()
    assert "weights outside orchestrator output directory" in str(excinfo.value)


@pytest.mark.parametrize(
    (
        "trainer_cls",
        "configure",
        "post_check",
        "expected_message",
    ),
    [
        pytest.param(
            _create_trainer_stub(
                summary_factory=lambda _, __: {
                    "status": TrainingStatus.SUCCESS.value,
                    "artifacts": {},
                    "metrics": {},
                }
            ),
            None,
            None,
            "did not return an adapter artifact",
            id="missing-adapter",
        ),
        pytest.param(
            _create_trainer_stub(
                summary_factory=lambda output_dir, __: {
                    "status": TrainingStatus.SUCCESS.value,
                    "artifacts": {
                        "adapter": str(output_dir / "adapter" / "missing"),
                    },
                    "metrics": {},
                }
            ),
            None,
            None,
            "does not exist",
            id="invalid-adapter-path",
        ),
        pytest.param(
            _create_trainer_stub(
                training_config=TRAINING_CONFIG_PAYLOAD,
                setup=_setup_binary_adapter,
                summary_factory=lambda _, ctx: {
                    "status": TrainingStatus.SUCCESS.value,
                    "artifacts": {
                        "adapter": str(ctx["adapter_dir"]),
                        "weights": str(ctx["weights_path"]),
                    },
                    "metrics": {"training_examples": 1},
                },
            ),
            _configure_manifest_failure,
            _assert_manifest_missing,
            "manifest write failed",
            id="manifest-write-failure",
        ),
    ],
)
def test_orchestrator_failure_scenarios(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer_cls: type,
    configure: Callable[[pytest.MonkeyPatch], None] | None,
    post_check: Callable[[Path], None] | None,
    expected_message: str,
) -> None:
    if configure is not None:
        configure(monkeypatch)

    orchestrator = EvolutionOrchestrator(
        model_registry_path=str(temp_dir), trainer_cls=trainer_cls
    )

    with pytest.raises(RuntimeError) as excinfo:
        orchestrator.trigger_encoder_training_pipeline()
    assert expected_message in str(excinfo.value)

    if post_check is not None:
        post_check(temp_dir)


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
    assert set(artifacts) == {"adapter", "weights", "weights_checksum"}
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
    assert set(repeat_artifacts) == {"adapter", "weights", "weights_checksum"}
    assert repeat_artifacts["adapter"].startswith(str(second_dir))

    repeat_weights = _assert_fallback_artifacts(second_dir)
    assert repeat_weights == weights


def test_orchestrator_records_energy_metrics_for_long_running_jobs(
    temp_dir: Path,
    fixed_run_uuid: UUID,
    orchestrator_factory,
    stub_energy_tracker,
) -> None:
    StubEnergyTracker, calls = stub_energy_tracker

    successful_trainer = _create_trainer_stub(
        training_config=TRAINING_CONFIG_PAYLOAD,
        setup=_setup_binary_adapter,
        persist_summary=True,
        summary_factory=lambda _, ctx: {
            "status": TrainingStatus.SUCCESS.value,
            "artifacts": {
                "adapter": str(ctx["adapter_dir"]),
                "weights": str(ctx["weights_path"]),
            },
            "metrics": {"training_examples": 8, "loss": 0.05},
        },
    )

    orchestrator = orchestrator_factory(
        trainer_cls=successful_trainer, energy_tracker_factory=StubEnergyTracker
    )

    run_path = Path(orchestrator.trigger_encoder_training_pipeline())
    assert run_path.name == f"temp-mistral-mntp-step-{fixed_run_uuid}"
    summary_payload = json.loads((run_path / "training_summary.json").read_text())

    assert calls == ["start", "stop"], "Energy tracker lifecycle was not invoked"
    assert summary_payload["metrics"]["training_examples"] == 8
    assert summary_payload["metrics"]["run_duration_seconds"] == pytest.approx(7200.0)
    assert summary_payload["metrics"]["cpu_seconds"] == pytest.approx(3600.0)
    assert summary_payload["metrics"]["energy_wh"] == pytest.approx(4321.0)
    assert summary_payload["metrics"]["energy_backend"] == "stub-energy"
    assert summary_payload["telemetry"]["energy"]["backend"] == "stub-energy"
    assert summary_payload["telemetry"]["energy"]["duration_seconds"] == pytest.approx(
        7200.0
    )

    energy_report_path = run_path / "energy_report.json"
    assert energy_report_path.exists()
    energy_report_payload = json.loads(energy_report_path.read_text())
    assert energy_report_payload["backend"] == "stub-energy"
    assert energy_report_payload["duration_seconds"] == pytest.approx(7200.0)
    assert energy_report_payload["cpu_seconds"] == pytest.approx(3600.0)


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
    assert summary["version"]

    artifacts = summary["artifacts"]
    adapter_dir = Path(artifacts["adapter"])
    weights_path = Path(artifacts["weights"])
    assert adapter_dir.exists()
    assert weights_path.exists()
    assert artifacts["weights_checksum"] == summary["version"]

    payload = json.loads(weights_path.read_text())
    assert payload["feature_dimension"] == len(curated_records[0]["embedding"])
    expected_epochs = int(trainer.config.get("curated_epochs", 15))
    assert payload["metrics"]["epochs"] == expected_epochs
    assert len(payload["records"]) == len(curated_records)

    summary_path = output_dir / "training_summary.json"
    assert summary_path.exists()
    persisted_summary = json.loads(summary_path.read_text())
    assert persisted_summary["status"] == TrainingStatus.SUCCESS.value


def test_mntp_trainer_curated_training_rejects_invalid_records(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "curated_failure"
    trainer = MNTPTrainer(
        training_config_path="configs/training/mntp_mistral_config.json",
        output_dir=str(output_dir),
    )

    invalid_curated_records = [
        {"target": 0.5},
        {"embedding": [], "target": 0.6},
    ]

    with pytest.raises(ValueError) as excinfo:
        trainer.train(curated_records=invalid_curated_records)
    assert "No valid curated records" in str(excinfo.value)

    summary_path = output_dir / "training_summary.json"
    assert not summary_path.exists()
