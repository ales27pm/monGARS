"""Utilities for generating high-fidelity embeddings with LLM2Vec and Dolphin-X1."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import logging
import math
import sys
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Mapping, Optional

import numpy as np

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

from monGARS.config import Settings, get_settings
from monGARS.core.constants import DEFAULT_EMBEDDING_BACKEND
from monGARS.core.embedding_backends import normalise_embedding_backend
from monGARS.core.inference_utils import (
    prepare_tokenizer_inputs,
    render_chat_prompt_from_text,
)
from monGARS.core.llm_integration import LLMRuntimeError, UnifiedLLMRuntime

try:  # pragma: no cover - optional heavy dependency
    import torch
except ImportError:  # pragma: no cover - dependency absent in some environments
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - dependency absent in some environments
    AutoModel = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

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


_TRANSFORMERS_COMPONENT_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, int]] = {}
_TRANSFORMERS_COMPONENT_LOCK = threading.Lock()


_SENTENCE_TRANSFORMER_CACHE: dict[str, Any] = {}
_SENTENCE_TRANSFORMER_LOCK = threading.Lock()


_KNOWN_MANAGER_EXCEPTIONS = _exception_tuple(
    EmbeddingBackendError,
    LLMRuntimeError,
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


def _resolve_transformers_device(
    settings: Settings,
):
    if torch is None:  # pragma: no cover - dependency missing
        raise EmbeddingBackendError("transformers backend requires torch")

    requested = getattr(settings, "transformers_embedding_device", None)
    if requested:
        try:
            device = torch.device(str(requested))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise EmbeddingBackendError(
                f"Invalid transformers embedding device: {requested!r}"
            ) from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise EmbeddingBackendError("CUDA requested but not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")

    return torch.device("cpu")


def _ensure_transformers_components(settings: Settings) -> tuple[Any, Any, Any, int]:
    model_id = str(getattr(settings, "transformers_embedding_model", "").strip())
    if not model_id:
        raise EmbeddingBackendError("transformers embedding model is not configured")

    if (
        AutoTokenizer is None or AutoModelForCausalLM is None
    ):  # pragma: no cover - dependency missing
        raise EmbeddingBackendError("transformers library is not available")
    if torch is None:  # pragma: no cover - dependency missing
        raise EmbeddingBackendError("PyTorch is required for transformers embeddings")

    device = _resolve_transformers_device(settings)
    cache_key = (model_id, str(device))

    with _TRANSFORMERS_COMPONENT_LOCK:
        cached = _TRANSFORMERS_COMPONENT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - network/config errors are rare
        logger.exception(
            "llm2vec.transformers.tokenizer_load_failed",
            extra={"model": model_id},
        )
        raise EmbeddingBackendError("Failed to load transformers tokenizer") from exc

    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token", None) is None:
        raise EmbeddingBackendError("Tokenizer does not define a pad token")
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"

    if device.type == "cuda":
        dtype = torch.float16
    elif device.type == "mps":  # pragma: no cover - macOS specific
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = None
    load_errors: list[Exception] = []
    loaders: tuple[tuple[str, Any | None], ...] = (
        ("AutoModel", AutoModel),
        ("AutoModelForCausalLM", AutoModelForCausalLM),
    )

    for loader_name, loader in loaders:
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            break
        except Exception as exc:  # pragma: no cover - network/config errors are rare
            load_errors.append(exc)
            logger.debug(
                "llm2vec.transformers.model_loader_failed",
                extra={"model": model_id, "loader": loader_name},
                exc_info=exc,
            )

    if model is None:
        last_error = load_errors[-1] if load_errors else None
        if last_error is not None:
            logger.error(
                "llm2vec.transformers.model_load_failed",
                extra={"model": model_id},
                exc_info=(
                    last_error.__class__,
                    last_error,
                    last_error.__traceback__,
                ),
            )
        else:
            logger.error(
                "llm2vec.transformers.model_loader_unavailable",
                extra={"model": model_id},
            )
        raise EmbeddingBackendError(
            "Failed to load transformers embedding model"
        ) from last_error

    model.to(device)
    model.eval()

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        raise EmbeddingBackendError("Model configuration missing hidden_size")

    components = (tokenizer, model, device, int(hidden_size))
    with _TRANSFORMERS_COMPONENT_LOCK:
        _TRANSFORMERS_COMPONENT_CACHE[cache_key] = components
    return components


def _format_instruction_texts(
    texts: Sequence[str], instruction: Optional[str]
) -> list[str]:
    candidates = [str(text) for text in texts]
    if instruction is None:
        return candidates
    instruction_text = str(instruction).strip()
    if not instruction_text:
        return candidates
    return [f"{instruction_text}\n\n{candidate}" for candidate in candidates]


def _encode_with_transformers(
    texts: List[str],
    instruction: Optional[str],
    settings: Settings | None = None,
    *,
    normalise: bool = True,
) -> np.ndarray:
    resolved_settings = settings or get_settings()
    tokenizer, model, device, hidden_size = _ensure_transformers_components(
        resolved_settings
    )

    if not texts:
        return np.zeros((0, hidden_size), dtype=np.float32)

    if torch is None:  # pragma: no cover - dependency missing
        raise EmbeddingBackendError("PyTorch is required for transformers embeddings")

    batch_size = max(
        1, int(getattr(resolved_settings, "transformers_embedding_batch_size", 1))
    )
    max_length = max(
        1, int(getattr(resolved_settings, "transformers_embedding_max_length", 1))
    )
    system_prompt = instruction

    pooled_results: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        chunk = [str(text) for text in texts[slice(start, start + batch_size)]]
        formatted = _format_instruction_texts(chunk, system_prompt)
        try:
            encoded = tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        except Exception as exc:  # pragma: no cover - tokenizer failures are rare
            logger.exception(
                "llm2vec.transformers.tokenizer_encode_failed",
                extra={"chunk_size": len(chunk), "max_length": max_length},
            )
            raise EmbeddingBackendError(
                "Tokenization failed for transformers embeddings"
            ) from exc

        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise EmbeddingBackendError(
                "Transformers model did not return hidden states"
            )

        final_hidden = hidden_states[-1]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            mask = torch.ones(
                final_hidden.shape[:2],
                dtype=final_hidden.dtype,
                device=final_hidden.device,
            )
        else:
            mask = attention_mask.to(final_hidden.dtype)
        mask = mask.unsqueeze(-1)

        masked_hidden = final_hidden * mask
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = masked_hidden.sum(dim=1) / token_counts

        if normalise:
            if hasattr(torch, "linalg") and hasattr(torch.linalg, "norm"):
                norms = torch.linalg.norm(pooled, ord=2, dim=1, keepdim=True)
            else:  # pragma: no cover - compatibility fallback
                norms = torch.norm(pooled, p=2, dim=1, keepdim=True)
            safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
            pooled = pooled / safe_norms

        pooled_results.append(
            pooled.detach().cpu().numpy().astype(np.float32, copy=False)
        )

    return (
        np.concatenate(pooled_results, axis=0)
        if pooled_results
        else np.zeros((0, hidden_size), dtype=np.float32)
    )


def _encode_with_sentence_transformers(
    texts: List[str], instruction: Optional[str], settings: Settings
) -> np.ndarray:
    spec = importlib.util.find_spec("sentence_transformers")
    if spec is None:
        raise ImportError("sentence_transformers not installed")
    module = importlib.import_module("sentence_transformers")
    model_cls = getattr(module, "SentenceTransformer", None)
    if model_cls is None:
        raise EmbeddingBackendError(
            "sentence_transformers does not expose SentenceTransformer"
        )
    model_name = getattr(settings, "transformers_embedding_model", None)
    if not model_name:
        raise EmbeddingBackendError("transformers_embedding_model must be configured")
    with _SENTENCE_TRANSFORMER_LOCK:
        model = _SENTENCE_TRANSFORMER_CACHE.get(model_name)
        if model is None:
            model = model_cls(model_name)
            _SENTENCE_TRANSFORMER_CACHE[model_name] = model
    formatted = _format_instruction_texts(texts, instruction)
    embeddings = model.encode(
        formatted, convert_to_numpy=True, normalize_embeddings=False
    )
    return np.asarray(embeddings, dtype=np.float32)


class LLM2VecEmbedder:
    """Generate embeddings using the configured backend with deterministic fallbacks."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        settings: Settings | None = None,
        runtime: UnifiedLLMRuntime | None = None,
        neuron_manager_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        configured_backend = backend or getattr(
            self._settings, "embedding_backend", DEFAULT_EMBEDDING_BACKEND
        )
        self._backend = normalise_embedding_backend(configured_backend, logger=logger)
        self._runtime = runtime or UnifiedLLMRuntime.instance(self._settings)
        self._vector_dimensions = max(
            1, int(getattr(self._settings, "llm2vec_vector_dimensions", 4096))
        )
        self._batch_size = max(
            1, int(getattr(self._settings, "llm2vec_max_batch_size", 8))
        )
        self._cache_size = max(
            1, int(getattr(self._settings, "llm2vec_cache_size", 128))
        )
        self._cache: OrderedDict[tuple[str, tuple[str, ...]], EmbeddingBatch] = (
            OrderedDict()
        )
        self._cache_lock = asyncio.Lock()
        concurrency = max(1, int(getattr(self._settings, "llm2vec_max_concurrency", 4)))
        self._semaphore = asyncio.Semaphore(concurrency)
        self._neuron_manager_factory = neuron_manager_factory
        self._neuron_manager: Any | None = None
        self._neuron_manager_lock = asyncio.Lock()
        self._ollama_module = None
        self._ollama_client = None
        self._ollama_client_lock = asyncio.Lock()
        self._dolphin_client = None
        self._dolphin_client_lock = asyncio.Lock()
        self._dolphin_dimension: int | None = None

    @property
    def backend(self) -> str:
        """Return the active embedding backend label."""

        return self._backend

    async def encode_batch(
        self, texts: Sequence[str], *, instruction: str | None = None
    ) -> EmbeddingBatch:
        """Return embeddings for ``texts`` using the selected backend."""

        prompt = instruction
        if prompt is None and self._backend != "transformers":
            prompt = getattr(self._settings, "llm2vec_instruction", None)
        resolved_prompt = str(prompt or "")
        cleaned = [str(text) for text in texts]
        cache_key = (resolved_prompt, tuple(cleaned))
        cached = await self._get_cached_batch(cache_key)
        if cached is not None:
            return cached
        if not cleaned:
            batch = EmbeddingBatch(vectors=[], used_fallback=False)
            await self._set_cached_batch(cache_key, batch)
            return batch

        if all(not candidate.strip() for candidate in cleaned):
            fallback = [
                self._fallback_vector(resolved_prompt, candidate)
                for candidate in cleaned
            ]
            batch = EmbeddingBatch(vectors=fallback, used_fallback=True)
            await self._set_cached_batch(cache_key, batch)
            return batch

        vectors: list[list[float]] = []
        used_fallback = False
        for start in range(0, len(cleaned), self._batch_size):
            chunk = cleaned[slice(start, start + self._batch_size)]
            try:
                chunk_vectors, chunk_fallback = await self._dispatch_backend(
                    chunk, resolved_prompt
                )
            except EmbeddingBackendError:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception(
                    "llm2vec.backend.chunk_failed",
                    extra={"backend": self._backend, "size": len(chunk)},
                )
                chunk_vectors = [
                    self._fallback_vector(resolved_prompt, text) for text in chunk
                ]
                chunk_fallback = True
            vectors.extend(chunk_vectors)
            used_fallback = used_fallback or chunk_fallback

        batch = EmbeddingBatch(vectors=vectors, used_fallback=used_fallback)
        await self._set_cached_batch(cache_key, batch)
        return batch

    async def embed_text(
        self, text: str, *, instruction: str | None = None
    ) -> tuple[list[float], bool]:
        """Return a single embedding vector for ``text``."""

        batch = await self.encode_batch([text], instruction=instruction)
        vector = batch.vectors[0] if batch.vectors else []
        return vector, batch.used_fallback

    async def _get_cached_batch(
        self, cache_key: tuple[str, tuple[str, ...]]
    ) -> EmbeddingBatch | None:
        async with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
            return cached

    async def _set_cached_batch(
        self, cache_key: tuple[str, tuple[str, ...]], batch: EmbeddingBatch
    ) -> None:
        async with self._cache_lock:
            self._cache[cache_key] = batch
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _fallback_vector(self, instruction: str, text: str) -> list[float]:
        seed = f"{instruction}\u0000{text}".encode("utf-8", "ignore")
        digest = hashlib.sha256(seed).digest()
        required = self._vector_dimensions
        values: list[float] = []
        buffer = digest
        index = 0
        while len(values) < required:
            if index >= len(buffer):
                buffer = hashlib.sha256(buffer).digest()
                index = 0
            byte = buffer[index]
            index += 1
            scaled = (byte / 255.0) * 2.0 - 1.0
            values.append(scaled)
        norm = math.sqrt(sum(component * component for component in values)) or 1.0
        return [component / norm for component in values]

    async def _dispatch_backend(
        self, chunk: Sequence[str], prompt: str
    ) -> tuple[list[list[float]], bool]:
        backend = self._backend
        if backend == "dolphin-x1-llm2vec":
            return await self._encode_with_dolphin_service(chunk, prompt)
        if backend == "ollama":
            return await self._encode_with_ollama(chunk, prompt)
        if backend == "transformers":
            return await self._encode_with_transformers_backend(chunk, prompt)
        return await self._encode_with_neuron_manager(chunk, prompt)

    async def _encode_with_neuron_manager(
        self, chunk: Sequence[str], prompt: str
    ) -> tuple[list[list[float]], bool]:
        rendered = self._render_chatml_batch(chunk, prompt)
        manager = await self._ensure_neuron_manager()
        if manager is None or not self._manager_ready(manager):
            return self._fallback_vectors(prompt, chunk), True

        try:
            async with self._semaphore:
                vectors = await asyncio.to_thread(
                    manager.encode,
                    rendered,
                    prompt,
                )
        except _KNOWN_MANAGER_EXCEPTIONS as exc:
            raise EmbeddingBackendError("Embedding backend unavailable") from exc

        return self._normalise_vectors(vectors, chunk, prompt)

    async def _ensure_neuron_manager(self) -> Any | None:
        if self._neuron_manager is not None:
            return self._neuron_manager

        async with self._neuron_manager_lock:
            if self._neuron_manager is not None:
                return self._neuron_manager

            factory = (
                self._neuron_manager_factory or self._default_neuron_manager_factory
            )
            try:
                manager = factory()
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "llm2vec.manager.initialisation_failed",
                    extra={"error": str(exc)},
                )
                manager = self._runtime_manager()
            self._neuron_manager = manager
        return self._neuron_manager

    def _default_neuron_manager_factory(self) -> Any:
        spec = importlib.util.find_spec("modules.neurons.core")
        if spec is None:
            raise RuntimeError("NeuronManager module not available")
        module = importlib.import_module("modules.neurons.core")
        manager_cls = getattr(module, "NeuronManager", None)
        if manager_cls is None:
            raise RuntimeError("NeuronManager class unavailable")

        llm2vec_options = {
            "device_map": getattr(self._settings, "llm2vec_device_map", None),
            "torch_dtype": getattr(self._settings, "llm2vec_torch_dtype", None),
            "tokenizer_name": getattr(self._settings, "llm2vec_tokenizer_name", None),
            "tokenizer_revision": getattr(
                self._settings, "llm2vec_tokenizer_revision", None
            ),
            "revision": getattr(self._settings, "llm2vec_revision", None),
            "loader": getattr(self._settings, "llm2vec_loader", None),
            "trust_remote_code": getattr(
                self._settings, "llm2vec_trust_remote_code", None
            ),
            "use_safetensors": getattr(self._settings, "llm2vec_use_safetensors", None),
        }
        encode_options = {
            "pooling_strategy": getattr(
                self._settings, "llm2vec_pooling_strategy", None
            )
        }

        return manager_cls(
            getattr(self._settings, "llm2vec_base_model", "nomic-ai/llm2vec-large"),
            getattr(self._settings, "llm2vec_encoder", None),
            fallback_dimensions=self._vector_dimensions,
            llm2vec_options=llm2vec_options,
            encode_options=encode_options,
        )

    def _runtime_manager(self) -> Any:
        runtime = self._runtime

        class _RuntimeManager:
            def is_ready(self) -> bool:
                return True

            def encode(
                self, texts: Sequence[str], instruction: str
            ) -> list[list[float]]:
                del instruction
                return runtime.embed(list(texts))

        return _RuntimeManager()

    def _normalise_vectors(
        self,
        vectors: Sequence[Sequence[float]] | Any,
        chunk: Sequence[str],
        prompt: str,
        *,
        expected_dimension: int | None = None,
    ) -> tuple[list[list[float]], bool]:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
            return self._fallback_vectors(prompt, chunk), True
        if len(vectors) != len(chunk):
            return self._fallback_vectors(prompt, chunk), True

        processed: list[list[float]] = []
        for raw, original in zip(vectors, chunk, strict=True):
            vector = self._coerce_vector(raw, expected_dimension)
            if vector is None:
                return self._fallback_vectors(prompt, chunk), True
            processed.append(vector)
        return processed, False

    def _coerce_vector(
        self, vector: Sequence[float] | Any, expected_dimension: int | None
    ) -> list[float] | None:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            return None
        converted: list[float] = []
        for component in vector:
            try:
                value = float(component)
            except (TypeError, ValueError):
                return None
            if math.isnan(value) or math.isinf(value):
                return None
            converted.append(value)
        return self._resize_vector(converted, expected_dimension)

    def _resize_vector(
        self, vector: Sequence[float], expected_dimension: int | None
    ) -> list[float]:
        dimension = expected_dimension or self._vector_dimensions
        if len(vector) > dimension:
            return list(vector[:dimension])
        if len(vector) < dimension:
            padding = [0.0] * (dimension - len(vector))
            return list(vector) + padding
        return list(vector)

    def _fallback_vectors(
        self, instruction: str, texts: Sequence[str]
    ) -> list[list[float]]:
        return [self._fallback_vector(instruction, text) for text in texts]

    def _render_chatml_batch(self, chunk: Sequence[str], prompt: str) -> list[str]:
        system_prompt = prompt or getattr(self._settings, "llm2vec_instruction", "")
        return [
            render_chat_prompt_from_text(
                text,
                system_prompt=system_prompt,
                include_assistant_stub=False,
            ).chatml
            for text in chunk
        ]

    def _manager_ready(self, manager: Any) -> bool:
        ready_attr = getattr(manager, "is_ready", None)
        if callable(ready_attr):
            try:
                return bool(ready_attr())
            except Exception:  # pragma: no cover - defensive fallback
                return False
        return bool(ready_attr)

    async def _encode_with_transformers_backend(
        self, chunk: Sequence[str], prompt: str
    ) -> tuple[list[list[float]], bool]:
        def _run_transformers():
            return _encode_with_transformers(
                list(chunk), prompt, self._settings, normalise=False
            )

        def _run_sentence_transformers():
            return _encode_with_sentence_transformers(
                list(chunk), prompt, self._settings
            )

        try:
            matrix = await asyncio.to_thread(_run_sentence_transformers)
        except ImportError:
            logger.info("llm2vec.transformers.sentence_transformers_missing")
            matrix = await asyncio.to_thread(_run_transformers)
        except EmbeddingBackendError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "llm2vec.transformers.sentence_transformers_failed",
                extra={"error": str(exc)},
            )
            matrix = await asyncio.to_thread(_run_transformers)

        return self._normalise_vectors(matrix, chunk, prompt)

    async def _encode_with_dolphin_service(
        self, chunk: Sequence[str], prompt: str
    ) -> tuple[list[list[float]], bool]:
        module = self._ensure_httpx_module()
        module, client = await self._ensure_dolphin_client(module)
        try:
            response = await client.post("/embed", json={"texts": list(chunk)})
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(
                "llm2vec.dolphin.request_failed",
                extra={"error": str(exc)},
            )
            return self._fallback_vectors(prompt, chunk), True

        embeddings = None
        dimension = None
        if isinstance(payload, Mapping):
            embeddings = payload.get("embeddings")
            raw_dimension = payload.get("dimension")
            if isinstance(raw_dimension, int) and raw_dimension > 0:
                dimension = raw_dimension
                self._dolphin_dimension = raw_dimension

        if not isinstance(embeddings, Sequence):
            return self._fallback_vectors(prompt, chunk), True

        return self._normalise_vectors(
            embeddings,
            chunk,
            prompt,
            expected_dimension=dimension or self._dolphin_dimension,
        )

    def _ensure_httpx_module(self):  # noqa: ANN201 - dynamic import helper
        existing = sys.modules.get("httpx")
        if existing is not None:
            return existing
        spec = importlib.util.find_spec("httpx")
        if spec is None:
            raise EmbeddingBackendError(
                "httpx is required for dolphin service embeddings"
            )
        return importlib.import_module("httpx")

    async def _ensure_dolphin_client(self, module):  # noqa: ANN201 - runtime type
        if self._dolphin_client is not None:
            return module, self._dolphin_client

        async with self._dolphin_client_lock:
            if self._dolphin_client is not None:
                return module, self._dolphin_client
            timeout = getattr(
                self._settings, "dolphin_x1_llm2vec_service_timeout", 30.0
            )
            timeout_config = None
            timeout_cls = getattr(module, "Timeout", None)
            if timeout_cls is not None:
                timeout_config = timeout_cls(timeout, connect=timeout)
            headers = None
            token = getattr(self._settings, "dolphin_x1_llm2vec_service_token", None)
            if token:
                headers = {"Authorization": f"Bearer {token}"}
            client = module.AsyncClient(
                base_url=str(self._settings.dolphin_x1_llm2vec_service_url),
                timeout=timeout_config or timeout,
                headers=headers,
            )
            response = await client.get("/health")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, Mapping):
                dimension = payload.get("dimension")
                if isinstance(dimension, int) and dimension > 0:
                    self._dolphin_dimension = dimension
            self._dolphin_client = client
        return module, client

    async def _encode_with_ollama(
        self, chunk: Sequence[str], prompt: str
    ) -> tuple[list[list[float]], bool]:
        module = self._ensure_ollama_module()
        client = await self._ensure_ollama_client(module)
        model = getattr(self._settings, "ollama_embedding_model", None)
        if not model:
            raise EmbeddingBackendError("ollama_embedding_model must be configured")
        try:
            response = await asyncio.to_thread(
                client.embed,
                model=model,
                input=list(chunk),
            )
        except _OLLAMA_TRANSPORT_ERRORS as exc:
            logger.warning(
                "llm2vec.ollama.request_failed",
                extra={"error": str(exc)},
            )
            return self._fallback_vectors(prompt, chunk), True

        embeddings = None
        if isinstance(response, Mapping):
            embeddings = response.get("embeddings")
        if not isinstance(embeddings, Sequence):
            return self._fallback_vectors(prompt, chunk), True

        dimensions = getattr(
            self._settings, "ollama_embedding_dimensions", self._vector_dimensions
        )
        return self._normalise_vectors(
            embeddings, chunk, prompt, expected_dimension=int(dimensions)
        )

    def _ensure_ollama_module(self):  # noqa: ANN201 - dynamic import helper
        if self._ollama_module is not None:
            logger.debug("llm2vec.ollama.module_import.reuse")
            return self._ollama_module
        logger.debug("llm2vec.ollama.module_import.start")
        spec = importlib.util.find_spec("ollama")
        if spec is None:
            raise EmbeddingBackendError("ollama backend requested but module missing")
        module = importlib.import_module("ollama")
        self._ollama_module = module
        logger.debug("llm2vec.ollama.module_import.success")
        return module

    async def _ensure_ollama_client(self, module=None):  # noqa: ANN201 - runtime type
        if module is None:
            module = self._ensure_ollama_module()
        if self._ollama_client is not None:
            return self._ollama_client
        async with self._ollama_client_lock:
            if self._ollama_client is not None:
                return self._ollama_client
            host = getattr(self._settings, "ollama_host", None)
            logger.debug(
                "llm2vec.ollama.client.initialising",
                extra={"host": host},
            )
            client = module.Client(host=host)
            self._ollama_client = client
            logger.debug(
                "llm2vec.ollama.client.initialised",
                extra={"host": host},
            )
        return self._ollama_client


class DolphinX1Embedder:
    """Generate embeddings by mean-pooling Dolphin-X1-8B hidden states."""

    DEFAULT_MODEL_ID = "dphn/Dolphin-X1-8B"
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
            or getattr(self._settings, "dolphin_x1_embedding_model_id", None)
            or getattr(self._settings, "dolphin3_embedding_model_id", None)
            or self.DEFAULT_MODEL_ID
        )
        self._max_length = max(
            1,
            int(
                max_length
                or getattr(
                    self._settings,
                    "dolphin_x1_embedding_max_length",
                    getattr(
                        self._settings,
                        "dolphin3_embedding_max_length",
                        self.DEFAULT_MAX_LENGTH,
                    ),
                )
            ),
        )
        self._batch_size = max(
            1,
            int(
                batch_size
                or getattr(
                    self._settings,
                    "dolphin_x1_embedding_batch_size",
                    getattr(
                        self._settings,
                        "dolphin3_embedding_batch_size",
                        self.DEFAULT_BATCH_SIZE,
                    ),
                )
            ),
        )
        self._target_dimension = max(
            1,
            int(
                target_dimension
                or getattr(
                    self._settings,
                    "dolphin_x1_embedding_vector_dimensions",
                    getattr(
                        self._settings,
                        "dolphin3_embedding_vector_dimensions",
                        self.DEFAULT_VECTOR_DIMENSION,
                    ),
                )
            ),
        )
        self._device_preference = device or (
            getattr(self._settings, "dolphin_x1_embedding_device", None)
            or getattr(self._settings, "dolphin3_embedding_device", None)
        )
        self._torch_dtype_config = torch_dtype or getattr(
            self._settings,
            "dolphin_x1_embedding_torch_dtype",
            getattr(self._settings, "dolphin3_embedding_torch_dtype", None),
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
        """Return embeddings for ``texts`` using Dolphin-X1 mean pooling."""

        if not texts:
            return []

        torch_module, model, tokenizer = self._ensure_model_components()
        device = self.device
        results: list[list[float]] = []
        system_prompt = getattr(self._settings, "llm2vec_instruction", None)

        for start in range(0, len(texts), self._batch_size):
            chunk = [
                str(text) for text in texts[slice(start, start + self._batch_size)]
            ]
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
                    "Dolphin-X1 model did not return hidden states"
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
                "PyTorch is required for Dolphin-X1 embeddings but is not installed"
            )
        torch_module = importlib.import_module("torch")
        self._torch_module = torch_module
        return torch_module

    def _load_transformers_classes(self) -> tuple[Any, Any]:
        spec = importlib.util.find_spec("transformers")
        if spec is None:
            raise EmbeddingBackendError(
                "transformers is required for Dolphin-X1 embeddings but is not installed"
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
                "dolphin_x1.embedding.dtype_unresolved",
                extra={"value": self._torch_dtype_config},
            )
            return None

        if device.type == "cpu" and getattr(dtype, "__str__", lambda: "")() not in {
            "torch.float32",
            "torch.float64",
        }:
            logger.warning(
                "dolphin_x1.embedding.dtype_cpu_override",
                extra={"requested": self._torch_dtype_config},
            )
            return torch_module.float32

        return dtype


# Backwards compatibility alias maintained for callers importing the legacy name.
Dolphin3Embedder = DolphinX1Embedder


@lru_cache(maxsize=1)
def get_llm2vec_embedder() -> LLM2VecEmbedder:
    """Return a cached embedder instance for reuse across services."""

    return LLM2VecEmbedder()


@lru_cache(maxsize=1)
def get_dolphin_x1_embedder() -> DolphinX1Embedder:
    """Return a cached Dolphin-X1 embedder instance."""

    return DolphinX1Embedder()


@lru_cache(maxsize=1)
def get_dolphin3_embedder() -> DolphinX1Embedder:
    """Deprecated alias for :func:`get_dolphin_x1_embedder`."""

    warnings.warn(
        "get_dolphin3_embedder is deprecated; use get_dolphin_x1_embedder instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_dolphin_x1_embedder()


__all__ = [
    "EmbeddingBackendError",
    "EmbeddingBatch",
    "DolphinX1Embedder",
    "Dolphin3Embedder",
    "LLM2VecEmbedder",
    "_encode_with_transformers",
    "get_dolphin_x1_embedder",
    "get_dolphin3_embedder",
    "get_llm2vec_embedder",
]
