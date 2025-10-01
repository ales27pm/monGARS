import asyncio
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")

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
async def test_call_result_method_logs_and_recovers_from_errors(caplog):
    engine = CuriosityEngine()

    class FaultyResult:
        def data(self) -> None:
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG):
        value = await engine._call_result_method(FaultyResult(), "data")

    assert value is None
    assert "boom" in caplog.text


@pytest.mark.asyncio
async def test_coerce_row_handles_unexpected_iterable_shapes(caplog):
    engine = CuriosityEngine()

    class OddRow:
        def data(self):  # noqa: ANN001 - interface mimics driver row
            return [1, 2, 3]

    with caplog.at_level(logging.DEBUG):
        coerced = await engine._coerce_row(OddRow())

    assert coerced == {}
    assert "coercion_error" in caplog.text


@pytest.mark.asyncio
async def test_detect_gaps_respects_graph_gap_cutoff(monkeypatch):
    engine = CuriosityEngine()
    engine.similar_history_threshold = 0
    engine.graph_gap_cutoff = 3

    async def fake_vector_similarity(*args, **kwargs) -> int:
        return 0

    async def always_missing_batch(entities):
        return {entity: False for entity in entities}

    async def fake_research(query: str) -> str:
        raise AssertionError(f"Research should not be triggered for: {query}")

    engine._perform_research = fake_research  # type: ignore[assignment]
    monkeypatch.setattr(engine, "_vector_similarity_search", fake_vector_similarity)
    monkeypatch.setattr(engine, "_check_entities_in_kg_batch", always_missing_batch)

    engine.nlp = lambda text: SimpleNamespace(
        ents=[SimpleNamespace(text="Entité inconnue")]
    )

    result = await engine.detect_gaps({"last_query": "Qu'est-ce que MonGARS?"})

    assert result == {"status": "sufficient_knowledge"}


@pytest.mark.asyncio
async def test_check_entities_in_kg_batch_handles_empty_and_malformed_lists(
    monkeypatch,
):
    engine = CuriosityEngine()

    calls: list[list[str]] = []

    async def fake_query(entities: Sequence[str]) -> dict[str, bool]:
        normalised = list(entities)
        calls.append(normalised)
        return {entity: True for entity in normalised}

    monkeypatch.setattr(engine, "_query_kg_entities", fake_query)

    assert await engine._check_entities_in_kg_batch([]) == {}
    assert calls == []

    assert await engine._check_entities_in_kg_batch(["   ", "\t", "\n"]) == {}
    assert calls == []

    result = await engine._check_entities_in_kg_batch(["Paris", None, 123])

    assert result == {"Paris": True}
    assert calls == [["paris"]]


@pytest.mark.asyncio
async def test_batch_lookup_uses_cache():
    engine = CuriosityEngine()

    class RecordingResult:
        def __init__(self, entities: list[str]) -> None:
            self._entities = entities

        async def data(self) -> list[dict[str, object]]:
            return [
                {"normalized": entity, "exists": entity == "paris"}
                for entity in self._entities
            ]

    class RecordingSession:
        def __init__(self) -> None:
            self.run_calls: list[list[str]] = []

        async def __aenter__(self) -> "RecordingSession":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def run(self, _query: str, *, entities: list[str]) -> RecordingResult:
            self.run_calls.append(list(entities))
            return RecordingResult(entities)

    session = RecordingSession()
    engine.embedding_system.driver = SimpleNamespace(session=lambda: session)

    first_lookup = await engine._check_entities_in_kg_batch(["Paris", "Lyon", "Paris"])

    assert first_lookup == {"Paris": True, "Lyon": False}
    assert session.run_calls == [["paris", "lyon"]]

    cached_lookup = await engine._check_entities_in_kg_batch(["Lyon"])

    assert cached_lookup == {"Lyon": False}
    assert session.run_calls == [["paris", "lyon"]]


@pytest.mark.asyncio
async def test_batch_lookup_shares_inflight_queries():
    engine = CuriosityEngine()

    started = asyncio.Event()
    release = asyncio.Event()

    class RecordingResult:
        def __init__(self, entities):
            self._entities = entities

        async def data(self) -> list[dict[str, object]]:
            return [
                {"normalized": entity, "exists": entity == "paris"}
                for entity in self._entities
            ]

    class RecordingSession:
        def __init__(self) -> None:
            self.run_calls: list[list[str]] = []

        async def __aenter__(self) -> "RecordingSession":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def run(self, _query: str, *, entities: list[str]) -> RecordingResult:
            self.run_calls.append(list(entities))
            started.set()
            await release.wait()
            return RecordingResult(entities)

    session = RecordingSession()
    engine.embedding_system.driver = SimpleNamespace(session=lambda: session)

    async def first_lookup() -> dict[str, bool]:
        return await engine._check_entities_in_kg_batch(["Paris", "Lyon"])

    async def second_lookup() -> dict[str, bool]:
        await started.wait()
        return await engine._check_entities_in_kg_batch(["Paris"])

    first_task = asyncio.create_task(first_lookup())
    await started.wait()
    second_task = asyncio.create_task(second_lookup())
    await asyncio.sleep(0)
    release.set()

    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert first_result == {"Paris": True, "Lyon": False}
    assert second_result == {"Paris": True}
    assert session.run_calls == [["paris", "lyon"]]


@pytest.mark.asyncio
async def test_batch_lookup_returns_false_on_driver_errors():
    engine = CuriosityEngine()

    class ErrorSession:
        async def __aenter__(self) -> "ErrorSession":
            return self

        async def __aexit__(
            self,
            exc_type,
            exc,
            tb,
        ) -> None:
            return None

        async def run(self, _query: str, *, entities: list[str]) -> None:
            raise RuntimeError("Simulated session error")

    engine.embedding_system.driver = SimpleNamespace(session=lambda: ErrorSession())

    result = await engine._check_entities_in_kg_batch(["Berlin", "Madrid"])

    assert result == {"Berlin": False, "Madrid": False}


@pytest.mark.asyncio
async def test_consume_result_rows_supports_multiple_interfaces():
    engine = CuriosityEngine()

    async def assert_rows(result_obj) -> None:
        rows = await engine._consume_result_rows(result_obj)
        assert rows == [
            {"normalized": "paris", "exists": True},
            {"normalized": "lyon", "exists": False},
        ]

    class AsyncDataResult:
        async def data(self) -> list[dict[str, object]]:
            return [
                {"normalized": "paris", "exists": True},
                {"normalized": "lyon", "exists": False},
            ]

    await assert_rows(AsyncDataResult())

    class Record:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def data(self) -> dict[str, object]:
            return self._payload

    class RecordsResult:
        def records(self) -> list[Record]:
            return [
                Record({"normalized": "paris", "exists": True}),
                Record({"normalized": "lyon", "exists": False}),
            ]

    await assert_rows(RecordsResult())

    class AsyncIteratorRecord:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def data(self) -> dict[str, object]:
            return self._payload

    class AsyncIterableResult:
        def __init__(self) -> None:
            self._records = [
                AsyncIteratorRecord({"normalized": "paris", "exists": True}),
                AsyncIteratorRecord({"normalized": "lyon", "exists": False}),
            ]
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._records):
                raise StopAsyncIteration
            record = self._records[self._index]
            self._index += 1
            return record

    await assert_rows(AsyncIterableResult())


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


@pytest.mark.asyncio
async def test_perform_research_uses_cache(monkeypatch):
    cache_events: list[tuple[int, dict[str, str]]] = []
    research_events: list[tuple[int, dict[str, str]]] = []

    def record_cache(amount: int, attributes: dict[str, str]) -> None:
        cache_events.append((amount, attributes.copy()))

    def record_research(amount: int, attributes: dict[str, str]) -> None:
        research_events.append((amount, attributes.copy()))

    response_calls: list[str] = []

    class CachedResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"documents": [{"summary": "Résumé depuis le service"}]}

    class RecordingClient:
        async def post(self, url: str, *, json: dict, timeout: int) -> CachedResponse:
            response_calls.append(json["query"])
            return CachedResponse()

        async def __aenter__(self) -> "RecordingClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    async def client_factory():  # pragma: no cover - helper for context manager
        client = RecordingClient()
        try:
            yield client
        finally:
            return

    engine = CuriosityEngine(http_client_factory=asynccontextmanager(client_factory))

    monkeypatch.setattr(curiosity_module._research_cache_counter, "add", record_cache)
    monkeypatch.setattr(
        curiosity_module._external_research_counter, "add", record_research
    )
    monkeypatch.setattr(engine.iris, "search", lambda query: "")

    result_first = await engine._perform_research("Analyse des réseaux")
    result_second = await engine._perform_research("Analyse des réseaux")

    assert result_first == result_second
    assert response_calls == ["Analyse des réseaux"]
    assert cache_events == [
        (1, {"event": "miss"}),
        (1, {"event": "hit"}),
    ]
    assert research_events == [(1, {"channel": "document_service"})]


def test_summarise_documents_filters_invalid_payloads():
    engine = CuriosityEngine()

    documents = [
        {"summary": " Première synthèse. "},
        {"summary": ""},
        {"not_summary": "ignored"},
        "invalid",
    ]

    assert engine._summarise_documents(documents) == "Première synthèse."
