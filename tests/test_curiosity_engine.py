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

    engine.similarity_threshold = 1.0
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

    assert engine.similarity_threshold == 0.42
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


@pytest.mark.asyncio
async def test_perform_research_records_document_service_channel(monkeypatch):
    engine = CuriosityEngine()

    calls: list[tuple[int, dict[str, str]]] = []

    def record(amount: int, attributes: dict[str, str]) -> None:
        calls.append((amount, attributes.copy()))

    monkeypatch.setattr(curiosity_module._external_research_counter, "add", record)

    class SuccessfulResponse:
        def __init__(self) -> None:
            self._payload = {"documents": [{"summary": "Résumé pertinent"}]}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class SuccessfulClient:
        def __init__(
            self, *args, **kwargs
        ) -> None:  # pragma: no cover - signature parity
            pass

        async def __aenter__(self) -> "SuccessfulClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, *args, **kwargs) -> SuccessfulResponse:
            return SuccessfulResponse()

    async def fail_search(_query: str) -> str:
        raise AssertionError(
            "Iris fallback should not trigger when documents are returned"
        )

    monkeypatch.setattr(curiosity_module.httpx, "AsyncClient", SuccessfulClient)
    monkeypatch.setattr(engine.iris, "search", fail_search)

    result = await engine._perform_research("Test de recherche")

    assert "Résumé pertinent" in result
    assert calls == [(1, {"channel": "document_service"})]


@pytest.mark.asyncio
async def test_perform_research_records_iris_fallback_channel(monkeypatch):
    engine = CuriosityEngine()

    calls: list[tuple[int, dict[str, str]]] = []

    def record(amount: int, attributes: dict[str, str]) -> None:
        calls.append((amount, attributes.copy()))

    monkeypatch.setattr(curiosity_module._external_research_counter, "add", record)

    class EmptyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"documents": []}

    class EmptyClient:
        def __init__(
            self, *args, **kwargs
        ) -> None:  # pragma: no cover - signature parity
            pass

        async def __aenter__(self) -> "EmptyClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, *args, **kwargs) -> EmptyResponse:
            return EmptyResponse()

    async def iris_result(query: str) -> str:
        return f"Résultat Iris pour {query}"

    monkeypatch.setattr(curiosity_module.httpx, "AsyncClient", EmptyClient)
    monkeypatch.setattr(engine.iris, "search", iris_result)

    result = await engine._perform_research("Nouvelle requête")

    assert "Résultat Iris" in result
    assert calls == [
        (1, {"channel": "document_service"}),
        (1, {"channel": "iris"}),
    ]
