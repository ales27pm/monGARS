"""Tests for the LLM2Vec embedder utilities."""

import math

import pytest

from monGARS.config import Settings
from monGARS.core.embeddings import EmbeddingBackendError, LLM2VecEmbedder


class _RecordingManager:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        self.calls.append(list(texts))
        base_vector = [float(len(self.calls)), 42.0, 84.0, 168.0]
        return [base_vector for _ in texts]


class _FailingManager:
    def __init__(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        raise RuntimeError("embedding backend unavailable")


class _PartialManager:
    def __init__(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float] | None]:
        return [[], None]


class _NotReadyManager:
    def __init__(self) -> None:
        self._ready = False
        self.calls: int = 0

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 2.0, 3.0] for _ in texts]


class _NonFiniteManager:
    def __init__(self) -> None:
        self._ready = True
        self.return_value = float("nan")

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        return [[self.return_value, 1.0, 2.0] for _ in texts]


@pytest.mark.asyncio
async def test_encode_batch_chunks_requests_and_normalises_dimensions() -> None:
    settings = Settings(
        llm2vec_max_batch_size=2,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _RecordingManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payloads = [f"text-{idx}" for idx in range(5)]
    result = await embedder.encode_batch(payloads)

    assert len(result.vectors) == len(payloads)
    assert all(len(vector) == 3 for vector in result.vectors)
    assert manager.calls == [payloads[:2], payloads[2:4], payloads[4:]]
    assert result.used_fallback is False


@pytest.mark.asyncio
async def test_embed_text_raises_on_backend_failure() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=_FailingManager
    )

    with pytest.raises(EmbeddingBackendError):
        await embedder.embed_text("hello world")


@pytest.mark.asyncio
async def test_encode_batch_generates_deterministic_fallback_vectors() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=5,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _PartialManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payloads = ["alpha", "beta"]
    first = await embedder.encode_batch(payloads)
    second = await embedder.encode_batch(payloads)

    assert first.used_fallback is True
    assert len(first.vectors) == len(payloads)
    assert first.vectors == second.vectors
    assert {len(vector) for vector in first.vectors} == {5}
    for vector in first.vectors:
        magnitude = math.sqrt(sum(component * component for component in vector))
        assert magnitude == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_encode_batch_skips_backend_for_blank_inputs() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=4,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _RecordingManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payloads = ["", "   "]
    batch = await embedder.encode_batch(payloads)

    assert batch.used_fallback is True
    assert manager.calls == []
    assert all(len(vector) == 4 for vector in batch.vectors)


@pytest.mark.asyncio
async def test_encode_batch_uses_fallback_when_manager_not_ready() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _NotReadyManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    batch = await embedder.encode_batch(["alpha", "beta"])

    assert batch.used_fallback is True
    assert manager.calls == 0
    assert all(len(vector) == 3 for vector in batch.vectors)


@pytest.mark.asyncio
async def test_encode_batch_fallback_depends_on_instruction_context() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=5,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _PartialManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payload = ["shared-text"]
    without_instruction = await embedder.encode_batch(payload)
    with_instruction = await embedder.encode_batch(
        payload, instruction="represent for troubleshooting"
    )

    assert without_instruction.used_fallback is True
    assert with_instruction.used_fallback is True
    assert without_instruction.vectors[0] != with_instruction.vectors[0]


@pytest.mark.asyncio
async def test_encode_batch_fallback_triggers_on_non_finite_values() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _NonFiniteManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    # Test fallback for NaN values
    batch = await embedder.encode_batch(["alpha"])

    assert batch.used_fallback is True
    assert len(batch.vectors) == 1
    assert len(batch.vectors[0]) == 3
    magnitude = math.sqrt(sum(component * component for component in batch.vectors[0]))
    assert magnitude == pytest.approx(1.0)

    # Test fallback for positive infinity values
    manager.return_value = float("inf")
    batch_inf = await embedder.encode_batch(["alpha"])

    assert batch_inf.used_fallback is True
    assert len(batch_inf.vectors) == 1
    assert len(batch_inf.vectors[0]) == 3
    magnitude_inf = math.sqrt(
        sum(component * component for component in batch_inf.vectors[0])
    )
    assert magnitude_inf == pytest.approx(1.0)

    # Test fallback for negative infinity values
    manager.return_value = float("-inf")
    batch_ninf = await embedder.encode_batch(["alpha"])

    assert batch_ninf.used_fallback is True
    assert len(batch_ninf.vectors) == 1
    assert len(batch_ninf.vectors[0]) == 3
    magnitude_ninf = math.sqrt(
        sum(component * component for component in batch_ninf.vectors[0])
    )
    assert magnitude_ninf == pytest.approx(1.0)
