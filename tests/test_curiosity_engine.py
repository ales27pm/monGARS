import os

os.environ.setdefault("SECRET_KEY", "test-secret")

import pytest

from monGARS.core.cortex import curiosity_engine as curiosity_module
from monGARS.core.cortex.curiosity_engine import CuriosityEngine


def _enable_vector_mode(engine: CuriosityEngine) -> None:
    engine.embedding_system._model_dependency_available = True  # type: ignore[attr-defined]
    engine.embedding_system._using_fallback_embeddings = False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_vector_similarity_uses_embeddings_with_history(monkeypatch):
    engine = CuriosityEngine()

    async def fake_encode(text: str) -> list[float]:
        engine.embedding_system._using_fallback_embeddings = False  # type: ignore[attr-defined]
        lowered = text.lower()
        if "quantum" in lowered:
            return [1.0, 0.0]
        if "classical" in lowered:
            return [0.0, 1.0]
        return [0.0, -1.0]

    _enable_vector_mode(engine)
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

    dissimilar = await engine._vector_similarity_search(
        "Completely unrelated topic",
        history,
    )
    assert dissimilar == 0

    engine.knowledge_gap_threshold = 1.0
    identical = await engine._vector_similarity_search(
        "Quantum computing basics",
        history,
    )
    assert identical == 1


@pytest.mark.asyncio
async def test_vector_similarity_handles_empty_and_no_overlap(monkeypatch):
    engine = CuriosityEngine()
    _enable_vector_mode(engine)

    async def fake_encode(text: str) -> list[float]:
        engine.embedding_system._using_fallback_embeddings = False  # type: ignore[attr-defined]
        lowered = text.lower()
        if "quantum" in lowered:
            return [1.0, 0.0]
        if "classical" in lowered:
            return [0.0, 1.0]
        return [0.0, -1.0]

    monkeypatch.setattr(curiosity_module, "select", None)
    monkeypatch.setattr(curiosity_module, "ConversationHistory", None)
    monkeypatch.setattr(curiosity_module, "async_session_factory", None)
    monkeypatch.setattr(engine.embedding_system, "encode", fake_encode)

    assert (
        await engine._vector_similarity_search(
            "Quantum computing basics",
            [],
        )
        == 0
    )

    assert (
        await engine._vector_similarity_search(
            "Quantum computing basics",
            ["Classical mechanics introduction"],
        )
        == 0
    )


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


@pytest.mark.asyncio
async def test_vector_similarity_fallback_both_layers_fail(monkeypatch):
    engine = CuriosityEngine()

    async def failing_encode(text: str) -> list[float]:
        raise RuntimeError("no embedding available")

    def zero_token_similarity(
        query_terms: set[str], history_candidates: list[str]
    ) -> int:
        return 0

    monkeypatch.setattr(curiosity_module, "select", None)
    monkeypatch.setattr(curiosity_module, "ConversationHistory", None)
    monkeypatch.setattr(curiosity_module, "async_session_factory", None)
    monkeypatch.setattr(engine.embedding_system, "encode", failing_encode)
    monkeypatch.setattr(engine, "_count_token_similarity", zero_token_similarity)

    similar = await engine._vector_similarity_search(
        "Unrelated topic",
        [
            "Classical computing basics",
            "Quantum entanglement explained",
            "Machine learning introduction",
        ],
    )

    assert similar == 0
