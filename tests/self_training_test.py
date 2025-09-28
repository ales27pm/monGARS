import pytest

from monGARS.core.self_training import SelfTrainingEngine


@pytest.fixture(autouse=True)
def stub_embedding_system(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbeddingSystem:
        def __init__(self) -> None:
            self.encodes: list[str] = []

        async def encode(self, text: str) -> tuple[list[float], bool]:
            self.encodes.append(text)
            return [0.0], False

    monkeypatch.setattr(
        "monGARS.core.self_training.EmbeddingSystem",
        DummyEmbeddingSystem,
    )


@pytest.mark.asyncio
async def test_run_training_cycle_creates_version():
    engine = SelfTrainingEngine()
    await engine.training_queue.put({"data": 1, "confidence": 0.95})
    await engine.training_queue.put({"data": 2, "confidence": 0.4})
    await engine._run_training_cycle()
    assert "v1" in engine.model_versions
    assert engine.model_versions["v1"]["data_count"] == 1


@pytest.mark.asyncio
async def test_run_training_cycle_no_data():
    engine = SelfTrainingEngine()
    await engine._run_training_cycle()
    assert engine.model_versions == {}


@pytest.mark.asyncio
async def test_run_training_cycle_skips_low_confidence_only():
    engine = SelfTrainingEngine(training_threshold=0.9)
    await engine.training_queue.put({"data": "low", "confidence": 0.1})
    await engine.training_queue.put({"data": "unknown"})
    await engine._run_training_cycle()
    assert engine.model_versions == {}
