"""Utilities for generating high-fidelity embeddings with LLM2Vec."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from modules.neurons.core import NeuronManager
from monGARS.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingBatch:
    """Container describing an embedding request outcome."""

    vectors: list[list[float]]
    used_fallback: bool


class EmbeddingBackendError(RuntimeError):
    """Raised when the embedding backend cannot produce vectors."""


class LLM2VecEmbedder:
    """Thin asynchronous wrapper around :class:`modules.neurons.core.NeuronManager`."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        neuron_manager_factory: Callable[[], NeuronManager] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._manager_factory = neuron_manager_factory or self._default_manager_factory
        self._manager: NeuronManager | None = None
        self._manager_lock = asyncio.Lock()
        concurrency = max(1, int(self._settings.llm2vec_max_concurrency))
        self._semaphore = asyncio.Semaphore(concurrency)

    async def encode_batch(
        self, texts: Sequence[str], *, instruction: str | None = None
    ) -> EmbeddingBatch:
        """Return embeddings for ``texts`` using LLM2Vec with graceful fallbacks."""

        if not texts:
            return EmbeddingBatch(vectors=[], used_fallback=False)

        cleaned: list[str] = [str(text) for text in texts]
        manager = await self._ensure_manager()
        prompt = (
            instruction
            if instruction is not None
            else self._settings.llm2vec_instruction
        )

        aggregate_vectors: list[list[float]] = []
        used_fallback = False
        batch_size = max(1, int(self._settings.llm2vec_max_batch_size))

        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]
            chunk_used_fallback = False

            try:
                async with self._semaphore:
                    raw_vectors = await asyncio.to_thread(manager.encode, chunk, prompt)
            except Exception as exc:  # pragma: no cover - unexpected backend errors
                logger.exception(
                    "llm2vec.encode.failed", extra={"chunk_size": len(chunk)}
                )
                raise EmbeddingBackendError("LLM2Vec encode failed") from exc

            # Normalise vectors while preserving ordering and chunk length.
            normalised_vectors: list[list[float]] = []
            for raw_vector in list(raw_vectors or [])[: len(chunk)]:
                prepared = self._normalise_dimensions(raw_vector)
                if prepared is None:
                    chunk_used_fallback = True
                    normalised_vectors.append([])
                else:
                    normalised_vectors.append(prepared)

            if len(normalised_vectors) < len(chunk):
                chunk_used_fallback = True
                normalised_vectors.extend(
                    [] for _ in range(len(chunk) - len(normalised_vectors))
                )

            aggregate_vectors.extend(normalised_vectors)
            used_fallback = used_fallback or chunk_used_fallback or not manager.is_ready

        if used_fallback:
            logger.debug("llm2vec.fallback_embeddings", extra={"count": len(cleaned)})
        return EmbeddingBatch(vectors=aggregate_vectors, used_fallback=used_fallback)

    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        """Return a single embedding vector for ``text``."""

        batch = await self.encode_batch([text], instruction=instruction)
        if not batch.vectors or not batch.vectors[0]:
            return [], batch.used_fallback
        return batch.vectors[0], batch.used_fallback

    async def _ensure_manager(self) -> NeuronManager:
        if self._manager is not None:
            return self._manager
        async with self._manager_lock:
            if self._manager is not None:
                return self._manager
            manager = await asyncio.to_thread(self._manager_factory)
            self._manager = manager
            return manager

    def _default_manager_factory(self) -> NeuronManager:
        options = {
            "device_map": self._settings.llm2vec_device_map,
            "torch_dtype": self._settings.llm2vec_torch_dtype,
        }
        filtered_options = {k: v for k, v in options.items() if v is not None}
        return NeuronManager(
            base_model_path=self._settings.llm2vec_base_model,
            default_encoder_path=self._settings.llm2vec_encoder,
            fallback_dimensions=self._settings.llm2vec_vector_dimensions,
            llm2vec_options=filtered_options,
        )

    def _normalise_dimensions(
        self, vector: Sequence[float] | None
    ) -> list[float] | None:
        if vector is None:
            return None
        if hasattr(vector, "tolist"):
            vector = vector.tolist()  # type: ignore[assignment]
        try:
            values = [float(component) for component in vector]
        except (TypeError, ValueError):
            return None

        if not values:
            return None

        dimensions = int(self._settings.llm2vec_vector_dimensions)
        if len(values) > dimensions:
            values = values[:dimensions]
        elif len(values) < dimensions:
            values.extend(0.0 for _ in range(dimensions - len(values)))
        return values


@lru_cache(maxsize=1)
def get_llm2vec_embedder() -> LLM2VecEmbedder:
    """Return a cached embedder instance for reuse across services."""

    return LLM2VecEmbedder()


__all__ = [
    "EmbeddingBackendError",
    "EmbeddingBatch",
    "LLM2VecEmbedder",
    "get_llm2vec_embedder",
]
