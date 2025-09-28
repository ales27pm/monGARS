import pytest

from monGARS.core.cortex import curiosity_engine as curiosity_module
from monGARS.core.cortex.curiosity_engine import CuriosityEngine


@pytest.mark.asyncio
async def test_vector_similarity_uses_embeddings_with_history(monkeypatch):
    engine = CuriosityEngine()

    async def fake_encode(text: str) -> list[float]:
        if "quantum" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]

    monkeypatch.setattr(curiosity_module, "select", None)
    monkeypatch.setattr(curiosity_module, "ConversationHistory", None)
    monkeypatch.setattr(curiosity_module, "async_session_factory", None)
    monkeypatch.setattr(engine.embedding_system, "encode", fake_encode)

    history = [
        "Quantum computing basics",
        "Quantum supremacy milestones",
        "Understanding classical computing",
    ]

    similar = await engine._vector_similarity_search(
        "Quantum computing advantages",
        history,
    )

    assert similar == 2


@pytest.mark.asyncio
async def test_vector_similarity_falls_back_to_tokens(monkeypatch):
    engine = CuriosityEngine()

    async def failing_encode(text: str) -> list[float]:
        raise RuntimeError("no embedding available")

    monkeypatch.setattr(curiosity_module, "select", None)
    monkeypatch.setattr(curiosity_module, "ConversationHistory", None)
    monkeypatch.setattr(curiosity_module, "async_session_factory", None)
    monkeypatch.setattr(engine.embedding_system, "encode", failing_encode)

    history = [
        "Quantum computing overview",
        "Daily weather forecast",
    ]

    similar = await engine._vector_similarity_search(
        "Quantum computing basics",
        history,
    )

    assert similar == 1
