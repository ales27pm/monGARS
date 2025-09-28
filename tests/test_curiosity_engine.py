import os

os.environ.setdefault("SECRET_KEY", "test-secret")

from types import SimpleNamespace

import pytest

from monGARS.core.cortex import curiosity_engine as curiosity_module
from monGARS.core.cortex.curiosity_engine import CuriosityEngine


def _enable_vector_mode(engine: CuriosityEngine) -> None:
    engine.embedding_system._model_dependency_available = True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_vector_similarity_uses_embeddings_with_history(monkeypatch):
    engine = CuriosityEngine()

    async def fake_encode(text: str) -> tuple[list[float], bool]:
        lowered = text.lower()
        if "quantum" in lowered:
            return [1.0, 0.0], False
        if "classical" in lowered:
            return [0.0, 1.0], False
        return [0.0, -1.0], False

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

    async def fake_encode(text: str) -> tuple[list[float], bool]:
        lowered = text.lower()
        if "quantum" in lowered:
            return [1.0, 0.0], False
        if "classical" in lowered:
            return [0.0, 1.0], False
        return [0.0, -1.0], False

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

    async def failing_encode(text: str) -> tuple[list[float], bool]:
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

    async def failing_encode(text: str) -> tuple[list[float], bool]:
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


def test_curiosity_engine_initialises_from_settings(monkeypatch):
    monkeypatch.setattr(
        curiosity_module.settings, "curiosity_similarity_threshold", 0.42
    )
    monkeypatch.setattr(
        curiosity_module.settings, "curiosity_minimum_similar_history", 7
    )
    monkeypatch.setattr(curiosity_module.settings, "curiosity_graph_gap_cutoff", 3)

    engine = CuriosityEngine()

    assert engine.knowledge_gap_threshold == 0.42
    assert engine.similar_history_threshold == 7
    assert engine.graph_gap_cutoff == 3


@pytest.mark.asyncio
async def test_detect_gaps_respects_graph_gap_cutoff(monkeypatch):
    engine = CuriosityEngine()
    engine.similar_history_threshold = 0
    engine.graph_gap_cutoff = 3

    async def fake_vector_similarity(*args, **kwargs) -> int:
        return 0

    async def always_missing(_entity: str) -> bool:
        return False

    async def fake_research(query: str) -> str:
        raise AssertionError(f"Research should not be triggered for: {query}")

    engine._perform_research = fake_research  # type: ignore[assignment]
    monkeypatch.setattr(engine, "_vector_similarity_search", fake_vector_similarity)
    monkeypatch.setattr(engine, "_check_entity_in_kg", always_missing)

    engine.nlp = lambda text: SimpleNamespace(
        ents=[SimpleNamespace(text="Entité inconnue")]
    )

    result = await engine.detect_gaps({"last_query": "Qu'est-ce que MonGARS?"})

    assert result == {"status": "sufficient_knowledge"}
