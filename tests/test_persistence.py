from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Iterable

import pytest
from sqlalchemy.exc import OperationalError

from monGARS.config import Settings
from monGARS.core.embeddings import EmbeddingBackendError, EmbeddingIdentity
from monGARS.core.persistence import PersistenceRepository
from monGARS.db.models import ConversationHistory
from monGARS.init_db import dispose_database, reset_database


@pytest.fixture(scope="module", autouse=True)
def _dispose_persistence_database_after_suite():
    """Close SQLAlchemy's pooled aiosqlite worker after this module."""

    yield
    asyncio.run(dispose_database())


class _StubEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.embedding_identity = EmbeddingIdentity(
            backend="test",
            model="stub-model",
            revision="rev-1",
            dimension=len(self.vector),
        )
        self.calls: list[tuple[str, str | None]] = []

    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        self.calls.append((text, instruction))
        return list(self.vector), False


class _SequenceEmbedder:
    def __init__(self, vectors: Iterable[list[float]]) -> None:
        prepared = list(vectors)
        self._iter = iter(prepared)
        self.embedding_identity = EmbeddingIdentity(
            backend="test",
            model="sequence-model",
            revision="rev-1",
            dimension=len(prepared[0]) if prepared else 3,
        )
        self.calls: list[str] = []

    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        self.calls.append(text)
        vector = next(self._iter, [0.0, 0.0, 0.0])
        return list(vector), False


class _ErroringEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        self.calls.append(text)
        raise EmbeddingBackendError("backend down")


class _FallbackEmbedder(_StubEmbedder):
    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        self.calls.append((text, instruction))
        return list(self.vector), True


class _DummySession:
    async def __aenter__(self) -> "_DummySession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def in_transaction(self) -> bool:
        return False


def test_unsupported_3072_dimension_ann_index_is_not_declared() -> None:
    index_names = {index.name for index in ConversationHistory.__table__.indexes}

    assert "ix_conversation_history_vector_cosine" not in index_names


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
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = _StubEmbedder([0.5, 0.25, 0.75])
    repo = PersistenceRepository(embedder=embedder, settings=settings)

    await repo.save_history_entry(
        user_id="vector-user", query="hello", response="world"
    )

    history = await repo.get_history("vector-user", limit=1)
    assert history
    assert len(history[0].vector) == settings.llm2vec_vector_dimensions
    assert history[0].vector == pytest.approx([0.5, 0.25, 0.75])
    assert history[0].embedding_identity == embedder.embedding_identity.storage_key
    assert embedder.calls


@pytest.mark.asyncio
async def test_vector_search_history_falls_back_without_pgvector() -> None:
    await reset_database()
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = _StubEmbedder()
    repo = PersistenceRepository(embedder=embedder, settings=settings)

    matches = await repo.vector_search_history("no-vector", "query text")
    assert matches == []
    assert embedder.calls


@pytest.mark.asyncio
async def test_vector_search_history_python_fallback_orders_by_distance() -> None:
    await reset_database()
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = _SequenceEmbedder(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    repo = PersistenceRepository(embedder=embedder, settings=settings)

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


@pytest.mark.asyncio
async def test_save_history_entry_skips_vector_when_embedding_unavailable() -> None:
    await reset_database()
    embedder = _ErroringEmbedder()
    repo = PersistenceRepository(embedder=embedder)

    await repo.save_history_entry(
        user_id="error-user", query="trouble", response="still stored"
    )

    history = await repo.get_history("error-user", limit=1)
    assert history
    assert history[0].vector is None
    assert embedder.calls


@pytest.mark.asyncio
async def test_save_history_entry_rejects_synthetic_fallback_vector() -> None:
    await reset_database()
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = _FallbackEmbedder([0.1, 0.2, 0.3])
    repo = PersistenceRepository(embedder=embedder, settings=settings)

    await repo.save_history_entry(
        user_id="fallback-user",
        query="query",
        response="stored without synthetic vector",
    )

    history = await repo.get_history("fallback-user", limit=1)
    assert history
    assert history[0].vector is None


@pytest.mark.asyncio
async def test_save_history_entry_rejects_dimension_mismatch_without_resize() -> None:
    await reset_database()
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = _StubEmbedder([0.1, 0.2])
    repo = PersistenceRepository(embedder=embedder, settings=settings)

    await repo.save_history_entry(
        user_id="dimension-user",
        query="query",
        response="stored without mismatched vector",
    )

    history = await repo.get_history("dimension-user", limit=1)
    assert history
    assert history[0].vector is None


@pytest.mark.asyncio
async def test_vector_search_rejects_embedding_identity_change() -> None:
    await reset_database()
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = _StubEmbedder([1.0, 0.0, 0.0])
    repo = PersistenceRepository(embedder=embedder, settings=settings)
    await repo.save_history_entry(
        user_id="identity-user",
        query="first",
        response="record",
    )
    embedder.embedding_identity = EmbeddingIdentity(
        backend="test",
        model="different-model",
        revision="rev-2",
        dimension=3,
    )

    matches = await repo.vector_search_history("identity-user", "query")

    assert matches == []


@pytest.mark.asyncio
async def test_vector_search_excludes_different_persisted_identity_after_restart() -> (
    None
):
    await reset_database()
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    first_embedder = _StubEmbedder([1.0, 0.0, 0.0])
    first_repo = PersistenceRepository(embedder=first_embedder, settings=settings)
    await first_repo.save_history_entry(
        user_id="identity-restart-user",
        query="first",
        response="record",
    )

    second_embedder = _StubEmbedder([1.0, 0.0, 0.0])
    second_embedder.embedding_identity = EmbeddingIdentity(
        backend="test",
        model="replacement-model",
        revision="rev-2",
        dimension=3,
    )
    second_repo = PersistenceRepository(embedder=second_embedder, settings=settings)

    matches = await second_repo.vector_search_history(
        "identity-restart-user",
        "query",
    )

    assert matches == []


def test_normalise_vector_rejects_non_finite_and_wrong_dimensions() -> None:
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    repo = PersistenceRepository(settings=settings, enable_embeddings=False)

    assert repo._normalise_vector([1.0, 2.0]) is None
    assert repo._normalise_vector([1.0, 2.0, 3.0, 4.0]) is None
    assert repo._normalise_vector([1.0, float("nan"), 3.0]) is None
    assert repo._normalise_vector([1.0, float("inf"), 3.0]) is None
    assert repo._normalise_vector([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_database_disposal_is_idempotent_and_engine_remains_reusable() -> None:
    await reset_database()
    await dispose_database()
    await dispose_database()
    await reset_database()
