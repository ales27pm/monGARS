from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Iterable

import pytest
from sqlalchemy.exc import OperationalError

from monGARS.config import get_settings
from monGARS.core.persistence import PersistenceRepository
from monGARS.init_db import reset_database


class _StubEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.calls: list[tuple[str, str | None]] = []

    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        self.calls.append((text, instruction))
        return list(self.vector), False


class _SequenceEmbedder:
    def __init__(self, vectors: Iterable[list[float]]) -> None:
        self._iter = iter(vectors)
        self.calls: list[str] = []

    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        self.calls.append(text)
        vector = next(self._iter, [0.0, 0.0, 0.0])
        return list(vector), False


class _DummySession:
    async def __aenter__(self) -> "_DummySession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def in_transaction(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_persistence_retries_and_surfaces_connection_failure():
    attempts: list[int] = []

    @asynccontextmanager
    async def failing_session_factory():
        yield _DummySession()

    repo = PersistenceRepository(session_factory=failing_session_factory)

    async def failing_operation(_session):
        attempts.append(1)
        raise OperationalError("SELECT 1", {}, RuntimeError("db down"))

    with pytest.raises(OperationalError):
        await repo._execute_with_retry(
            failing_operation, operation_name="failing_operation"
        )

    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_save_history_entry_records_embedding() -> None:
    await reset_database()
    embedder = _StubEmbedder([0.5, 0.25, 0.75])
    repo = PersistenceRepository(embedder=embedder)

    await repo.save_history_entry(
        user_id="vector-user", query="hello", response="world"
    )

    history = await repo.get_history("vector-user", limit=1)
    assert history
    settings = get_settings()
    assert len(history[0].vector) == settings.llm2vec_vector_dimensions
    assert history[0].vector[:3] == pytest.approx([0.5, 0.25, 0.75])
    assert all(component == 0.0 for component in history[0].vector[3:])
    assert embedder.calls


@pytest.mark.asyncio
async def test_vector_search_history_falls_back_without_pgvector() -> None:
    await reset_database()
    embedder = _StubEmbedder()
    repo = PersistenceRepository(embedder=embedder)

    matches = await repo.vector_search_history("no-vector", "query text")
    assert matches == []
    assert embedder.calls


@pytest.mark.asyncio
async def test_vector_search_history_python_fallback_orders_by_distance() -> None:
    await reset_database()
    embedder = _SequenceEmbedder(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    repo = PersistenceRepository(embedder=embedder)

    await repo.save_history_entry(
        user_id="python-fallback",
        query="q1",
        response="r1",
    )
    await repo.save_history_entry(
        user_id="python-fallback",
        query="q2",
        response="r2",
    )
    await repo.save_history_entry(
        user_id="python-fallback",
        query="q3",
        response="r3",
    )

    matches = await repo.vector_search_history("python-fallback", "fresh query")
    assert matches
    # The first match should correspond to the vector [1, 0, 0] with a cosine distance of 0.
    assert matches[0].record.query == "q1"
    assert pytest.approx(matches[0].distance, abs=1e-6) == 0.0
