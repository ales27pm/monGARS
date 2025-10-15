"""Utilities for generating high-fidelity embeddings with LLM2Vec."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
from collections import OrderedDict
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
        self._fallback_cache_size = 256
        self._fallback_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()

    async def encode_batch(
        self, texts: Sequence[str], *, instruction: str | None = None
    ) -> EmbeddingBatch:
        """Return embeddings for ``texts`` using LLM2Vec with graceful fallbacks."""

        if not texts:
            return EmbeddingBatch(vectors=[], used_fallback=False)

        cleaned: list[str] = [str(text) for text in texts]
        prompt = (
            instruction
            if instruction is not None
            else self._settings.llm2vec_instruction
        )

        if all(not text.strip() for text in cleaned):
            vectors = [self._fallback_vector(prompt, text) for text in cleaned]
            return EmbeddingBatch(vectors=vectors, used_fallback=True)

        manager = await self._ensure_manager()

        aggregate_vectors: list[list[float]] = []
        used_fallback = False
        batch_size = max(1, int(self._settings.llm2vec_max_batch_size))

        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]
            (
                chunk_vectors,
                backend_indices,
                backend_payloads,
                chunk_used_fallback,
            ) = self._partition_chunk(chunk, prompt)

            raw_sequence: Sequence[Sequence[float]] | None = None
            manager_ready = self._is_manager_ready(manager)

            if backend_payloads and manager_ready:
                try:
                    async with self._semaphore:
                        raw_sequence = await asyncio.to_thread(
                            manager.encode, backend_payloads, prompt
                        )
                except Exception as exc:  # pragma: no cover - unexpected backend errors
                    logger.exception(
                        "llm2vec.encode.failed",
                        extra={
                            "chunk_size": len(chunk),
                            "backend_size": len(backend_payloads),
                        },
                    )
                    raise EmbeddingBackendError("LLM2Vec encode failed") from exc
            elif backend_payloads and not manager_ready:
                chunk_used_fallback = True
                self._assign_fallbacks(chunk, chunk_vectors, backend_indices, prompt)

            (
                chunk_vectors,
                merge_used_fallback,
            ) = self._merge_backend_results(
                chunk,
                chunk_vectors,
                backend_indices,
                raw_sequence,
                prompt,
            )
            chunk_used_fallback = chunk_used_fallback or merge_used_fallback

            aggregate_vectors.extend(chunk_vectors)  # type: ignore[arg-type]
            used_fallback = used_fallback or chunk_used_fallback or not manager_ready

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

        if any(not math.isfinite(component) for component in values):
            logger.warning(
                "llm2vec.embedding.non_finite",
                extra={"component_count": len(values), "values": values},
            )
            return None

        dimensions = int(self._settings.llm2vec_vector_dimensions)
        if len(values) > dimensions:
            values = values[:dimensions]
        elif len(values) < dimensions:
            values.extend(0.0 for _ in range(dimensions - len(values)))
        return values

    def _fallback_vector(self, instruction: str, text: str) -> list[float]:
        cache_key = (instruction, text)
        cached = self._fallback_cache.get(cache_key)
        if cached is not None:
            self._fallback_cache.move_to_end(cache_key)
            return list(cached)

        dimensions = max(1, int(self._settings.llm2vec_vector_dimensions))
        serialized = json.dumps(cache_key, ensure_ascii=False, separators=(",", ":"))
        secret = self._settings.SECRET_KEY.encode("utf-8")
        digest = hmac.new(secret, serialized.encode("utf-8"), hashlib.sha256).digest()
        repeated = (digest * ((dimensions // len(digest)) + 1))[:dimensions]
        values = [(byte / 255.0) * 2 - 1 for byte in repeated]
        magnitude = math.sqrt(sum(component * component for component in values))
        if magnitude == 0:
            return [0.0] * dimensions
        normalised = [component / magnitude for component in values]
        self._fallback_cache[cache_key] = list(normalised)
        if len(self._fallback_cache) > self._fallback_cache_size:
            self._fallback_cache.popitem(last=False)
        return normalised

    def _partition_chunk(self, chunk: list[str], prompt: str) -> tuple[
        list[list[float] | None],
        list[int],
        list[str],
        bool,
    ]:
        chunk_vectors: list[list[float] | None] = [None] * len(chunk)
        backend_indices: list[int] = []
        backend_payloads: list[str] = []
        used_fallback = False

        for index, text in enumerate(chunk):
            if not text.strip():
                used_fallback = True
                chunk_vectors[index] = self._fallback_vector(prompt, text)
            else:
                backend_indices.append(index)
                backend_payloads.append(text)

        return chunk_vectors, backend_indices, backend_payloads, used_fallback

    def _assign_fallbacks(
        self,
        chunk: list[str],
        chunk_vectors: list[list[float] | None],
        backend_indices: list[int],
        prompt: str,
    ) -> None:
        for index in backend_indices:
            chunk_vectors[index] = self._fallback_vector(prompt, chunk[index])

    def _merge_backend_results(
        self,
        chunk: list[str],
        chunk_vectors: list[list[float] | None],
        backend_indices: list[int],
        raw_sequence: Sequence[Sequence[float]] | None,
        prompt: str,
    ) -> tuple[list[list[float]], bool]:
        used_fallback = False
        prepared_sequence = list(raw_sequence or [])

        for relative_index, index in enumerate(backend_indices):
            candidate = (
                prepared_sequence[relative_index]
                if relative_index < len(prepared_sequence)
                else None
            )
            prepared = self._normalise_dimensions(candidate)
            if prepared is None:
                used_fallback = True
                chunk_vectors[index] = self._fallback_vector(prompt, chunk[index])
            else:
                chunk_vectors[index] = prepared

        for index, vector in enumerate(chunk_vectors):
            if vector is None:
                used_fallback = True
                chunk_vectors[index] = self._fallback_vector(prompt, chunk[index])

        return chunk_vectors, used_fallback

    def _is_manager_ready(self, manager: NeuronManager) -> bool:
        attr = getattr(manager, "is_ready", None)
        try:
            return bool(attr()) if callable(attr) else bool(attr)
        except Exception:  # pragma: no cover - defensive guard
            return False


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
