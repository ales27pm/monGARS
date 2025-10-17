"""Utilities for generating high-fidelity embeddings with LLM2Vec and Dolphin 3."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib
import importlib.util
import json
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency
    from httpx import HTTPError as HTTPXError
except Exception:  # pragma: no cover - httpx may be unavailable
    HTTPXError = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from requests import HTTPError as RequestsHTTPError
except Exception:  # pragma: no cover - requests may be unavailable
    RequestsHTTPError = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from huggingface_hub.utils import HfHubHTTPError
except Exception:  # pragma: no cover - huggingface_hub may be unavailable
    HfHubHTTPError = None  # type: ignore[assignment]

from modules.neurons.core import NeuronManager
from monGARS.config import Settings, get_settings
from monGARS.core.constants import DEFAULT_EMBEDDING_BACKEND
from monGARS.core.embedding_backends import normalise_embedding_backend
from monGARS.core.inference_utils import (
    prepare_tokenizer_inputs,
    render_chat_prompt_from_text,
)

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - dependency absent in some environments
    SentenceTransformer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _exception_tuple(*candidates: object) -> tuple[type[BaseException], ...]:
    result: list[type[BaseException]] = []
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            result.append(candidate)
    return tuple(result)


@dataclass(slots=True)
class EmbeddingBatch:
    """Container describing an embedding request outcome."""

    vectors: list[list[float]]
    used_fallback: bool


class EmbeddingBackendError(RuntimeError):
    """Raised when the embedding backend cannot produce vectors."""


_KNOWN_MANAGER_EXCEPTIONS = _exception_tuple(
    EmbeddingBackendError,
    RuntimeError,
    OSError,
    ConnectionError,
    asyncio.TimeoutError,
    RequestsHTTPError,
    HTTPXError,
    HfHubHTTPError,
)

_OLLAMA_TRANSPORT_ERRORS = _exception_tuple(
    ConnectionError,
    OSError,
    asyncio.TimeoutError,
    RequestsHTTPError,
    HTTPXError,
)


class LLM2VecEmbedder:
    """Thin asynchronous wrapper around :class:`modules.neurons.core.NeuronManager`."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        settings: Settings | None = None,
        neuron_manager_factory: Callable[[], NeuronManager] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._backend = self._resolve_backend(
            backend
            or getattr(self._settings, "embedding_backend", DEFAULT_EMBEDDING_BACKEND)
        )
        self._manager_factory = neuron_manager_factory or self._default_manager_factory
        self._manager: NeuronManager | None = None
        self._manager_lock = asyncio.Lock()
        concurrency = max(1, int(self._settings.llm2vec_max_concurrency))
        self._semaphore = asyncio.Semaphore(concurrency)
        self._fallback_cache_size = 256
        self._fallback_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._encode_batch_cache_size = 256
        self._encode_batch_cache: OrderedDict[
            tuple[str, tuple[str, ...]], EmbeddingBatch
        ] = OrderedDict()
        self._encode_batch_cache_lock = threading.Lock()
        self._ollama_module: Any | None = None
        self._ollama_client: Any | None = None
        self._ollama_client_lock = asyncio.Lock()
        self._transformers_model: Any | None = None
        self._transformers_model_lock = asyncio.Lock()

    @property
    def backend(self) -> str:
        """Return the active embedding backend."""

        return self._backend

    async def encode_batch(
        self, texts: Sequence[str], *, instruction: str | None = None
    ) -> EmbeddingBatch:
        """Return embeddings for ``texts`` using LLM2Vec with graceful fallbacks."""

        cleaned: list[str] = [str(text) for text in texts]
        prompt = (
            instruction
            if instruction is not None
            else self._settings.llm2vec_instruction
        )

        cache_key = self._encode_batch_cache_key(prompt, cleaned)
        cached = self._get_cached_batch(cache_key)
        if cached is not None:
            return cached

        if not cleaned:
            batch = EmbeddingBatch(vectors=[], used_fallback=False)
            self._set_cached_batch(cache_key, batch)
            return batch

        if all(not text.strip() for text in cleaned):
            vectors = [self._fallback_vector(prompt, text) for text in cleaned]
            batch = EmbeddingBatch(vectors=vectors, used_fallback=True)
            self._set_cached_batch(cache_key, batch)
            return batch

        if self._backend == "ollama":
            batch = await self._encode_with_ollama(cleaned, prompt)
        elif self._backend == "transformers":
            batch = await self._encode_with_transformers(cleaned, prompt)
        else:
            batch = await self._encode_with_huggingface(cleaned, prompt)

        self._set_cached_batch(cache_key, batch)
        return batch

    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        """Return a single embedding vector for ``text``."""

        batch = await self.encode_batch([text], instruction=instruction)
        if not batch.vectors or not batch.vectors[0]:
            return [], batch.used_fallback
        return batch.vectors[0], batch.used_fallback

    async def _encode_with_huggingface(
        self, cleaned: Sequence[str], prompt: str
    ) -> EmbeddingBatch:
        manager = await self._ensure_manager()

        aggregate_vectors: list[list[float]] = []
        used_fallback = False
        batch_size = max(1, int(self._settings.llm2vec_max_batch_size))

        for start in range(0, len(cleaned), batch_size):
            chunk = list(cleaned[start : start + batch_size])
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
                except _KNOWN_MANAGER_EXCEPTIONS as exc:
                    logger.exception(
                        "llm2vec.encode.failed",
                        extra={
                            "chunk_size": len(chunk),
                            "backend_size": len(backend_payloads),
                        },
                    )
                    raise EmbeddingBackendError(
                        "embedding backend unavailable"
                    ) from exc
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

    async def _encode_with_ollama(
        self, cleaned: Sequence[str], prompt: str
    ) -> EmbeddingBatch:
        aggregate_vectors: list[list[float]] = []
        used_fallback = False
        batch_size = max(1, int(self._settings.llm2vec_max_batch_size))

        for start in range(0, len(cleaned), batch_size):
            chunk = list(cleaned[start : start + batch_size])
            (
                chunk_vectors,
                backend_indices,
                backend_payloads,
                chunk_used_fallback,
            ) = self._partition_chunk(chunk, prompt)

            raw_sequence: Sequence[Sequence[float]] | None = None

            if backend_payloads:
                try:
                    async with self._semaphore:
                        raw_sequence = await self._ollama_embed(backend_payloads)
                except EmbeddingBackendError:
                    chunk_used_fallback = True
                    self._assign_fallbacks(
                        chunk, chunk_vectors, backend_indices, prompt
                    )

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
            used_fallback = used_fallback or chunk_used_fallback

        if used_fallback:
            logger.debug("llm2vec.fallback_embeddings", extra={"count": len(cleaned)})
        return EmbeddingBatch(vectors=aggregate_vectors, used_fallback=used_fallback)

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
            "dtype": self._settings.llm2vec_torch_dtype,
        }
        filtered_options = {k: v for k, v in options.items() if v is not None}
        return NeuronManager(
            base_model_path=self._settings.llm2vec_base_model,
            default_encoder_path=self._settings.llm2vec_encoder,
            fallback_dimensions=self._settings.llm2vec_vector_dimensions,
            llm2vec_options=filtered_options,
        )

    def _resolve_backend(self, configured: str | None) -> str:
        return normalise_embedding_backend(
            configured,
            default=DEFAULT_EMBEDDING_BACKEND,
            logger=logger,
            log_event="llm2vec.embedding_backend.invalid",
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
        rendered = render_chat_prompt_from_text(
            text,
            system_prompt=instruction,
            include_assistant_stub=False,
        ).chatml
        cache_key = (instruction, rendered)
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

    def _encode_batch_cache_key(
        self, prompt: str, cleaned: Sequence[str]
    ) -> tuple[str, tuple[str, ...]]:
        return (prompt, tuple(cleaned))

    def _clone_embedding_batch(self, batch: EmbeddingBatch) -> EmbeddingBatch:
        return EmbeddingBatch(
            vectors=[list(vector) for vector in batch.vectors],
            used_fallback=batch.used_fallback,
        )

    def _get_cached_batch(
        self, cache_key: tuple[str, tuple[str, ...]]
    ) -> EmbeddingBatch | None:
        with self._encode_batch_cache_lock:
            cached = self._encode_batch_cache.get(cache_key)
            if cached is None:
                return None
            self._encode_batch_cache.move_to_end(cache_key)
            return self._clone_embedding_batch(cached)

    def _set_cached_batch(
        self, cache_key: tuple[str, tuple[str, ...]], batch: EmbeddingBatch
    ) -> None:
        with self._encode_batch_cache_lock:
            self._encode_batch_cache[cache_key] = self._clone_embedding_batch(batch)
            self._encode_batch_cache.move_to_end(cache_key)
            if len(self._encode_batch_cache) > self._encode_batch_cache_size:
                self._encode_batch_cache.popitem(last=False)

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
                backend_payloads.append(
                    render_chat_prompt_from_text(
                        text,
                        system_prompt=prompt,
                        include_assistant_stub=False,
                    ).chatml
                )

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

    async def _ollama_embed(self, payloads: Sequence[str]) -> Sequence[Sequence[float]]:
        module = self._ensure_ollama_module()
        if module is None:
            logger.error("llm2vec.ollama.module_missing")
            raise EmbeddingBackendError("Ollama client is not available")

        client = await self._ensure_ollama_client(module)
        if client is None:
            logger.error("llm2vec.ollama.client_unavailable")
            raise EmbeddingBackendError("Ollama client is not available")

        model_name = self._resolve_ollama_model()
        dimensions = self._resolve_ollama_dimensions()
        client_errors = self._ollama_error_types(module)
        expected_errors = client_errors + _OLLAMA_TRANSPORT_ERRORS
        embed_kwargs: dict[str, Any] = {
            "model": model_name,
            "input": list(payloads),
        }
        if dimensions is not None:
            embed_kwargs["dimensions"] = dimensions

        if expected_errors:
            try:
                response = await asyncio.to_thread(client.embed, **embed_kwargs)
            except expected_errors as exc:  # type: ignore[misc]
                logger.exception(
                    "llm2vec.ollama.embed_failed",
                    extra={"payload_count": len(payloads), "model": model_name},
                )
                raise EmbeddingBackendError("Ollama embeddings failed") from exc
        else:
            response = await asyncio.to_thread(client.embed, **embed_kwargs)

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, Mapping):
            embeddings = response.get("embeddings")

        if not embeddings:
            logger.error(
                "llm2vec.ollama.empty_response",
                extra={"payload_count": len(payloads), "model": model_name},
            )
            raise EmbeddingBackendError("Ollama embeddings missing")

        return embeddings

    @staticmethod
    def _ollama_error_types(module: Any) -> tuple[type[BaseException], ...]:
        return _exception_tuple(
            getattr(module, "OllamaError", None),
            getattr(module, "RequestError", None),
            getattr(module, "ResponseError", None),
            getattr(module, "StreamError", None),
        )

    async def _encode_with_transformers(
        self, cleaned: Sequence[str], prompt: str
    ) -> EmbeddingBatch:
        try:
            model = await self._ensure_transformers_model()
        except EmbeddingBackendError:
            logger.warning(
                "llm2vec.transformers.model_unavailable",
                extra={"payload_count": len(cleaned)},
            )
            vectors = [self._fallback_vector(prompt, text) for text in cleaned]
            return EmbeddingBatch(vectors=vectors, used_fallback=True)

        aggregate_vectors: list[list[float]] = []
        used_fallback = False
        batch_size = max(1, int(self._settings.llm2vec_max_batch_size))

        for start in range(0, len(cleaned), batch_size):
            chunk = list(cleaned[start : start + batch_size])
            chunk_vectors: list[list[float] | None] = [None] * len(chunk)
            encode_indices: list[int] = []
            encode_payloads: list[str] = []

            for index, text in enumerate(chunk):
                if not text.strip():
                    chunk_vectors[index] = self._fallback_vector(prompt, text)
                    used_fallback = True
                else:
                    encode_indices.append(index)
                    encode_payloads.append(str(text))

            raw_vectors: Sequence[Sequence[float]] | None = None
            if encode_payloads:
                try:
                    async with self._semaphore:
                        raw_vectors = await asyncio.to_thread(
                            self._transformers_encode_sync,
                            model,
                            encode_payloads,
                        )
                except Exception as exc:  # pragma: no cover - encode failures are rare
                    logger.exception(
                        "llm2vec.transformers.encode_failed",
                        extra={
                            "chunk_size": len(chunk),
                            "payload_count": len(encode_payloads),
                        },
                    )
                    raw_vectors = None

            for relative_index, index in enumerate(encode_indices):
                candidate = (
                    raw_vectors[relative_index]
                    if raw_vectors is not None and relative_index < len(raw_vectors)
                    else None
                )
                prepared = self._normalise_dimensions(candidate)
                if prepared is None:
                    chunk_vectors[index] = self._fallback_vector(prompt, chunk[index])
                    used_fallback = True
                else:
                    chunk_vectors[index] = prepared

            for index, vector in enumerate(chunk_vectors):
                if vector is None:
                    chunk_vectors[index] = self._fallback_vector(prompt, chunk[index])
                    used_fallback = True

            aggregate_vectors.extend(chunk_vectors)  # type: ignore[arg-type]

        return EmbeddingBatch(vectors=aggregate_vectors, used_fallback=used_fallback)

    def _ensure_ollama_module(self) -> Any | None:
        if self._ollama_module is not None:
            return self._ollama_module
        spec = importlib.util.find_spec("ollama")
        if spec is None:
            return None
        module = importlib.import_module("ollama")
        self._ollama_module = module
        return module

    async def _ensure_ollama_client(self, module: Any) -> Any | None:
        if self._ollama_client is not None:
            return self._ollama_client
        async with self._ollama_client_lock:
            if self._ollama_client is not None:
                return self._ollama_client
            host_value = getattr(self._settings, "ollama_host", None)
            host = str(host_value) if host_value else None
            self._ollama_client = module.Client(host=host)
        return self._ollama_client

    def _resolve_ollama_model(self) -> str:
        model = getattr(self._settings, "ollama_embedding_model", None)
        if not model:
            raise EmbeddingBackendError("Ollama embedding model is not configured")
        return str(model).strip()

    async def _ensure_transformers_model(self) -> Any:
        if self._transformers_model is not None:
            return self._transformers_model

        if SentenceTransformer is None:
            raise EmbeddingBackendError(
                "sentence-transformers library is not available"
            )

        async with self._transformers_model_lock:
            if self._transformers_model is not None:
                return self._transformers_model

            model_name = getattr(
                self._settings,
                "transformers_embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            try:
                model = await asyncio.to_thread(SentenceTransformer, str(model_name))
            except Exception as exc:  # pragma: no cover - model load errors are rare
                logger.exception(
                    "llm2vec.transformers.model_load_failed",
                    extra={"model": model_name},
                )
                raise EmbeddingBackendError(
                    "Failed to load transformers embedding model"
                ) from exc
            self._transformers_model = model
            return model

    @staticmethod
    def _transformers_encode_sync(
        model: Any, payloads: Sequence[str]
    ) -> list[list[float]]:
        result = model.encode(  # type: ignore[call-arg]
            list(payloads), convert_to_numpy=True, normalize_embeddings=False
        )
        if hasattr(result, "tolist"):
            return result.tolist()
        return [[float(component) for component in sequence] for sequence in result]

    def _resolve_ollama_dimensions(self) -> int | None:
        try:
            return int(self._settings.llm2vec_vector_dimensions)
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
            return None


class Dolphin3Embedder:
    """Generate embeddings by mean-pooling Dolphin 3.0 hidden states."""

    DEFAULT_MODEL_ID = "dphn/Dolphin3.0-Llama3.1-8B"
    DEFAULT_MAX_LENGTH = 4096
    DEFAULT_BATCH_SIZE = 2
    DEFAULT_VECTOR_DIMENSION = 3072

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_id: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        target_dimension: int | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_id = (
            model_id
            or getattr(self._settings, "dolphin3_embedding_model_id", None)
            or self.DEFAULT_MODEL_ID
        )
        self._max_length = max(
            1,
            int(
                max_length
                or getattr(
                    self._settings,
                    "dolphin3_embedding_max_length",
                    self.DEFAULT_MAX_LENGTH,
                )
            ),
        )
        self._batch_size = max(
            1,
            int(
                batch_size
                or getattr(
                    self._settings,
                    "dolphin3_embedding_batch_size",
                    self.DEFAULT_BATCH_SIZE,
                )
            ),
        )
        self._target_dimension = max(
            1,
            int(
                target_dimension
                or getattr(
                    self._settings,
                    "dolphin3_embedding_vector_dimensions",
                    self.DEFAULT_VECTOR_DIMENSION,
                )
            ),
        )
        self._device_preference = device or getattr(
            self._settings, "dolphin3_embedding_device", None
        )
        self._torch_dtype_config = torch_dtype or getattr(
            self._settings, "dolphin3_embedding_torch_dtype", None
        )
        self._torch_module: Any | None = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._model_lock = threading.Lock()

    @property
    def vector_dimension(self) -> int:
        """Return the configured output dimensionality."""

        return self._target_dimension

    @property
    def device(self):  # noqa: ANN201 - torch device type resolved dynamically
        """Return the resolved torch device used for inference."""

        if self._device is not None:
            return self._device
        self._ensure_model_components()
        return self._device

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length used during tokenisation."""

        return self._max_length

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for ``texts`` using Dolphin 3 mean pooling."""

        if not texts:
            return []

        torch_module, model, tokenizer = self._ensure_model_components()
        device = self.device
        results: list[list[float]] = []
        system_prompt = getattr(self._settings, "llm2vec_instruction", None)

        for start in range(0, len(texts), self._batch_size):
            chunk = [str(text) for text in texts[start : start + self._batch_size]]
            formatted_chunk = [
                render_chat_prompt_from_text(
                    text,
                    system_prompt=system_prompt,
                    include_assistant_stub=False,
                ).chatml
                for text in chunk
            ]
            prepared_inputs, _ = prepare_tokenizer_inputs(
                tokenizer,
                formatted_chunk,
                max_length=self._max_length,
                device=device,
                padding=True,
                truncation=True,
            )
            with torch_module.inference_mode():
                outputs = model(**prepared_inputs, output_hidden_states=True)

            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise EmbeddingBackendError(
                    "Dolphin 3 model did not return hidden states"
                )

            final_hidden = hidden_states[-1]
            attention_mask = prepared_inputs.get("attention_mask")
            if attention_mask is None:
                mask = torch_module.ones(
                    final_hidden.shape[:2],
                    dtype=final_hidden.dtype,
                    device=final_hidden.device,
                )
            else:
                mask = attention_mask.to(final_hidden.dtype)
            mask = mask.unsqueeze(-1)

            masked_hidden = final_hidden * mask
            token_counts = mask.sum(dim=1).clamp_min(1.0)
            pooled = torch_module.nan_to_num(
                masked_hidden.sum(dim=1) / token_counts,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            for vector in pooled:
                results.append(self._prepare_output_vector(vector, torch_module))

        return results

    def _ensure_model_components(self):  # noqa: ANN201 - runtime torch types
        if self._model is not None and self._tokenizer is not None:
            return self._torch_module, self._model, self._tokenizer

        with self._model_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._torch_module, self._model, self._tokenizer

            torch_module = self._load_torch()
            tokenizer_cls, model_cls = self._load_transformers_classes()

            device = self._resolve_device(torch_module)
            dtype = self._resolve_torch_dtype(torch_module, device)

            tokenizer = tokenizer_cls.from_pretrained(
                self._model_id,
                use_fast=True,
                trust_remote_code=True,
            )

            model_options: dict[str, Any] = {"trust_remote_code": True}
            if dtype is not None:
                model_options["dtype"] = dtype

            model = model_cls.from_pretrained(self._model_id, **model_options)
            model.eval()
            if device is not None and getattr(model, "hf_device_map", None) is None:
                model.to(device)

            self._torch_module = torch_module
            self._tokenizer = tokenizer
            self._model = model
            self._device = device

        return self._torch_module, self._model, self._tokenizer

    def _prepare_output_vector(self, tensor, torch_module):  # noqa: ANN201
        vector = tensor.detach().to(torch_module.float32)
        if vector.device.type != "cpu":
            vector = vector.cpu()

        components = vector.shape[-1]
        if components > self._target_dimension:
            vector = vector[: self._target_dimension]
        elif components < self._target_dimension:
            pad = torch_module.zeros(
                self._target_dimension - components,
                dtype=vector.dtype,
                device=vector.device,
            )
            vector = torch_module.cat((vector, pad), dim=0)

        return vector.tolist()

    def _load_torch(self):  # noqa: ANN201 - runtime torch module
        if self._torch_module is not None:
            return self._torch_module

        spec = importlib.util.find_spec("torch")
        if spec is None:
            raise EmbeddingBackendError(
                "PyTorch is required for Dolphin 3 embeddings but is not installed"
            )
        torch_module = importlib.import_module("torch")
        self._torch_module = torch_module
        return torch_module

    def _load_transformers_classes(self) -> tuple[Any, Any]:
        spec = importlib.util.find_spec("transformers")
        if spec is None:
            raise EmbeddingBackendError(
                "transformers is required for Dolphin 3 embeddings but is not installed"
            )
        transformers_module = importlib.import_module("transformers")
        tokenizer_cls = getattr(transformers_module, "AutoTokenizer", None)
        model_cls = getattr(transformers_module, "AutoModelForCausalLM", None)
        if tokenizer_cls is None or model_cls is None:
            raise EmbeddingBackendError(
                "transformers does not provide AutoTokenizer/AutoModelForCausalLM"
            )
        return tokenizer_cls, model_cls

    def _resolve_device(self, torch_module):  # noqa: ANN201
        if self._device_preference:
            return torch_module.device(self._device_preference)
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        mps_available = getattr(torch_module.backends, "mps", None)
        if mps_available and mps_available.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    def _resolve_torch_dtype(self, torch_module, device):  # noqa: ANN201
        if not self._torch_dtype_config:
            return None

        configured = str(self._torch_dtype_config).strip()
        if not configured:
            return None

        alias_map = {
            "fp16": "float16",
            "half": "float16",
            "float16": "float16",
            "float32": "float32",
            "fp32": "float32",
            "float": "float32",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
        }
        candidates = {
            configured,
            configured.lower(),
            alias_map.get(configured.lower(), ""),
        }
        dtype = None
        for candidate in candidates:
            if not candidate:
                continue
            attribute = getattr(torch_module, candidate, None)
            if attribute is not None and getattr(attribute, "__class__", None):
                dtype = attribute
                break

        if dtype is None:
            logger.warning(
                "dolphin3.embedding.dtype_unresolved",
                extra={"value": self._torch_dtype_config},
            )
            return None

        if device.type == "cpu" and getattr(dtype, "__str__", lambda: "")() not in {
            "torch.float32",
            "torch.float64",
        }:
            logger.warning(
                "dolphin3.embedding.dtype_cpu_override",
                extra={"requested": self._torch_dtype_config},
            )
            return torch_module.float32

        return dtype


@lru_cache(maxsize=1)
def get_llm2vec_embedder() -> LLM2VecEmbedder:
    """Return a cached embedder instance for reuse across services."""

    return LLM2VecEmbedder()


@lru_cache(maxsize=1)
def get_dolphin3_embedder() -> Dolphin3Embedder:
    """Return a cached Dolphin 3 embedder instance."""

    return Dolphin3Embedder()


__all__ = [
    "EmbeddingBackendError",
    "EmbeddingBatch",
    "Dolphin3Embedder",
    "LLM2VecEmbedder",
    "get_dolphin3_embedder",
    "get_llm2vec_embedder",
]
