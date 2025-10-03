"""Tests for the LLM2Vec embedder utilities."""

import pytest

from monGARS.config import Settings
from monGARS.core.embeddings import EmbeddingBackendError, LLM2VecEmbedder


class _RecordingManager:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.is_ready = True

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        self.calls.append(list(texts))
        base_vector = [float(len(self.calls)), 42.0, 84.0, 168.0]
        return [base_vector for _ in texts]


class _FailingManager:
    def __init__(self) -> None:
        self.is_ready = True

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        raise RuntimeError("embedding backend unavailable")


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
