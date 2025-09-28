import json
from pathlib import Path

import pytest

from monGARS.core.self_training import SelfTrainingEngine


@pytest.fixture(autouse=True)
def stub_embedding_system(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbeddingSystem:
        def __init__(self) -> None:
            self.encodes: list[str] = []

        async def encode(self, text: str) -> tuple[list[float], bool]:
            self.encodes.append(text)
            return [float(len(text))], False

    monkeypatch.setattr(
        "monGARS.core.self_training.EmbeddingSystem",
        DummyEmbeddingSystem,
    )


@pytest.fixture()
def trainer_stub() -> type:
    class DummyTrainer:
        runs: list[list[dict[str, object]]] = []

        def __init__(self, training_config_path: str, output_dir: str) -> None:
            self.training_config_path = training_config_path
            self.output_dir = Path(output_dir)

        def train(self, curated_records=None):  # type: ignore[override]
            records = list(curated_records or [])
            DummyTrainer.runs.append(records)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            weights_path = self.output_dir / "weights.json"
            weights_path.write_text(json.dumps({"records": len(records)}))
            summary = {
                "status": "success",
                "metrics": {
                    "training_examples": len(records),
                    "loss": 0.05,
                },
                "artifacts": {
                    "weights": str(weights_path),
                },
            }
            return summary

    DummyTrainer.runs = []
    return DummyTrainer


@pytest.mark.asyncio
async def test_run_training_cycle_creates_version(
    tmp_path: Path, trainer_stub: type
) -> None:
    engine = SelfTrainingEngine(
        trainer_cls=trainer_stub,
        dataset_root=str(tmp_path / "datasets"),
        model_registry_path=str(tmp_path / "models"),
    )
    await engine.training_queue.put(
        {"text": "trainable", "confidence": 0.95, "id": "row-1"}
    )
    await engine.training_queue.put({"text": "ignored", "confidence": 0.4})
    await engine._run_training_cycle()
    assert "v1" in engine.model_versions
    version = engine.model_versions["v1"]
    assert version["data_count"] == 1
    assert version["summary"]["metrics"]["training_examples"] == 1
    dataset_file = version["dataset"]["dataset_file"]
    assert Path(dataset_file).exists()
    assert trainer_stub.runs and trainer_stub.runs[0][0]["text_preview"] == "trainable"


@pytest.mark.asyncio
async def test_run_training_cycle_no_data(tmp_path: Path, trainer_stub: type) -> None:
    engine = SelfTrainingEngine(
        trainer_cls=trainer_stub,
        dataset_root=str(tmp_path / "datasets"),
        model_registry_path=str(tmp_path / "models"),
    )
    await engine._run_training_cycle()
    assert engine.model_versions == {}


@pytest.mark.asyncio
async def test_run_training_cycle_skips_low_confidence_only(
    tmp_path: Path, trainer_stub: type
) -> None:
    engine = SelfTrainingEngine(
        training_threshold=0.9,
        trainer_cls=trainer_stub,
        dataset_root=str(tmp_path / "datasets"),
        model_registry_path=str(tmp_path / "models"),
    )
    await engine.training_queue.put({"text": "low", "confidence": 0.1})
    await engine.training_queue.put({"text": "unknown"})
    await engine._run_training_cycle()
    assert engine.model_versions == {}


@pytest.mark.asyncio
async def test_run_training_cycle_handles_non_numeric_confidence(
    tmp_path: Path, trainer_stub: type
) -> None:
    engine = SelfTrainingEngine(
        training_threshold=0.5,
        trainer_cls=trainer_stub,
        dataset_root=str(tmp_path / "datasets"),
        model_registry_path=str(tmp_path / "models"),
    )
    await engine.training_queue.put({"text": "valid", "confidence": 0.8})
    await engine.training_queue.put({"text": "bad", "confidence": "not-a-number"})
    await engine._run_training_cycle()
    assert engine.model_versions["v1"]["data_count"] == 1
