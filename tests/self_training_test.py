import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
            return {
                "status": "success",
                "metrics": {
                    "training_examples": len(records),
                    "loss": 0.05,
                },
                "artifacts": {
                    "weights": str(weights_path),
                },
            }

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
    assert version["dataset"]["version"] == 1


@pytest.mark.asyncio
async def test_run_training_cycle_handles_multiple_records(
    tmp_path: Path, trainer_stub: type
) -> None:
    engine = SelfTrainingEngine(
        trainer_cls=trainer_stub,
        dataset_root=str(tmp_path / "datasets"),
        model_registry_path=str(tmp_path / "models"),
    )
    await engine.training_queue.put({"text": "first", "confidence": 0.9, "id": "row-1"})
    await engine.training_queue.put(
        {"prompt": "second", "confidence": 0.85, "id": "row-2"}
    )
    await engine._run_training_cycle()

    assert len(trainer_stub.runs) == 1
    assert len(trainer_stub.runs[0]) == 2
    summary = engine.model_versions["v1"]["summary"]
    assert summary["metrics"]["training_examples"] == 2


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


@pytest.mark.asyncio
async def test_run_training_cycle_skips_missing_confidence(
    tmp_path: Path, trainer_stub: type
) -> None:
    engine = SelfTrainingEngine(
        training_threshold=0.5,
        trainer_cls=trainer_stub,
        dataset_root=str(tmp_path / "datasets"),
        model_registry_path=str(tmp_path / "models"),
    )
    await engine.training_queue.put({"text": "valid", "confidence": 0.7})
    await engine.training_queue.put({"text": "no-confidence"})
    await engine._run_training_cycle()

    assert len(trainer_stub.runs) == 1
    assert len(trainer_stub.runs[0]) == 1
    assert trainer_stub.runs[0][0]["text_preview"] == "valid"


@pytest.mark.asyncio
async def test_run_training_cycle_skips_missing_text_fields(
    tmp_path: Path, trainer_stub: type
) -> None:
    engine = SelfTrainingEngine(
        trainer_cls=trainer_stub,
        dataset_root=str(tmp_path / "datasets"),
        model_registry_path=str(tmp_path / "models"),
    )
    await engine.training_queue.put({"confidence": 0.95, "id": "missing"})
    await engine.training_queue.put({"response": "train", "confidence": 0.9})
    await engine._run_training_cycle()

    assert len(trainer_stub.runs) == 1
    assert len(trainer_stub.runs[0]) == 1
    assert trainer_stub.runs[0][0]["text_preview"] == "train"


@pytest.mark.asyncio
async def test_dataset_catalog_and_pii_scrubbing(
    tmp_path: Path, trainer_stub: type
) -> None:
    engine = SelfTrainingEngine(
        trainer_cls=trainer_stub,
        dataset_root=str(tmp_path / "datasets"),
        model_registry_path=str(tmp_path / "models"),
    )
    await engine.training_queue.put(
        {
            "text": "Reach me at person@example.com",
            "confidence": 0.92,
            "id": "user-123",
        }
    )
    await engine._run_training_cycle()

    version = engine.model_versions["v1"]
    dataset_meta = version["dataset"]
    dataset_file = Path(dataset_meta["dataset_file"])
    assert dataset_file.exists()

    with dataset_file.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    assert rows and "[REDACTED_EMAIL]" in rows[0]["text_preview"]
    assert trainer_stub.runs[0][0]["text_preview"].endswith("[REDACTED_EMAIL]")
    assert dataset_meta["version"] == 1

    catalog_path = tmp_path / "datasets" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["latest_version"] == 1
    assert catalog["versions"][0]["run_id"] == dataset_meta["run_id"]


def test_collect_internal_reasoning_records_handles_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyHippo:
        MAX_HISTORY = 5

        def __init__(self, enable_scheduler: bool = False) -> None:
            self.enable_scheduler = enable_scheduler
            self.calls: list[tuple[str, int]] = []

        async def history(self, user_id: str, limit: int):  # type: ignore[override]
            self.calls.append((user_id, limit))
            return [
                SimpleNamespace(
                    query="Please calculate 2 + 2",  # matches reasoning keywords
                    response="<reasoning>2+2=4</reasoning><answer>4</answer>",
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "monGARS.core.hippocampus",
        SimpleNamespace(Hippocampus=DummyHippo),
    )

    def fake_get_running_loop():
        return object()

    monkeypatch.setattr(
        "monGARS.core.self_training.asyncio.get_running_loop", fake_get_running_loop
    )

    engine = SelfTrainingEngine()
    records = engine._collect_internal_reasoning_records(limit=1)

    assert records
    assert records[0]["answer"] == "4"
    assert records[0]["metadata"]["source"] == "hippocampus"
