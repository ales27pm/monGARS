"""Neural utilities used by the cognition stack."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import math
import os
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

try:  # pragma: no cover - optional dependency
    from neo4j.async_driver import AsyncGraphDatabase
except ImportError:  # pragma: no cover - driver not installed in tests
    AsyncGraphDatabase = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - tests run without the heavy model
    SentenceTransformer = None  # type: ignore[assignment]

from monGARS.core.caching.tiered_cache import TieredCache

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MODEL_DIMENSION = 384
_EMPTY_CACHE_KEY = "<EMPTY>"


class _NoOpResult:
    async def single(self) -> dict[str, Any]:
        return {"exists": False}


class _NoOpSession:
    async def __aenter__(self) -> "_NoOpSession":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        return None

    async def run(self, *args: Any, **kwargs: Any) -> _NoOpResult:
        return _NoOpResult()


class _NoOpDriver:
    def session(self) -> _NoOpSession:
        return _NoOpSession()

    async def close(self) -> None:
        return None


class EmbeddingSystem:
    """Encode text and provide an optional knowledge-graph driver.

    The production configuration loads a :class:`SentenceTransformer` model and a
    Neo4j driver when the required dependencies are available. Both dependencies
    are optional to keep test environments lightweight; deterministic fallbacks
    are used otherwise.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache_ttl: int = 900,
        driver: Any | None = None,
        fallback_dimensions: int | None = None,
        cache_max_entries: int = 1024,
    ) -> None:
        self._model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", _DEFAULT_MODEL
        )
        self._model: SentenceTransformer | None = None
        self._model_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        self._cache_ttl = cache_ttl
        self._cache = TieredCache()
        self._cache_index: OrderedDict[str, None] = OrderedDict()
        self._cache_max_entries = max(1, cache_max_entries)
        self._explicit_fallback_dimensions = fallback_dimensions is not None
        if fallback_dimensions is None:
            self._fallback_dimensions = _DEFAULT_MODEL_DIMENSION
        else:
            self._fallback_dimensions = max(1, fallback_dimensions)
        self._model_dependency_available = SentenceTransformer is not None
        self._using_fallback_embeddings = not self._model_dependency_available
        self.driver = driver or self._create_driver()

    async def close(self) -> None:
        close_callable = getattr(self.driver, "close", None)
        if not close_callable:
            return
        result = close_callable()
        if inspect.isawaitable(result):
            await result

    async def encode(self, text: str) -> list[float]:
        """Return an embedding for *text* with caching and graceful fallbacks."""

        normalized = text.strip()
        cache_key = normalized or _EMPTY_CACHE_KEY

        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        vector: list[float]
        fallback_triggered = False
        if not normalized:
            vector = self._fallback_embedding(cache_key)
        elif SentenceTransformer is None:
            logger.debug(
                "SentenceTransformer not available; using fallback embedding for '%s'",
                cache_key,
            )
            vector = self._fallback_embedding(cache_key)
            fallback_triggered = True
        else:
            try:
                model = await self._ensure_model()
            except Exception as exc:  # pragma: no cover - model load failures are rare
                logger.warning(
                    "Failed to load embedding model '%s': %s", self._model_name, exc
                )
                vector = self._fallback_embedding(cache_key)
                fallback_triggered = True
            else:
                try:
                    encoded = await asyncio.to_thread(
                        model.encode,
                        normalized,
                        normalize_embeddings=True,
                    )
                    if isinstance(encoded, Iterable):
                        try:
                            vector = [float(value) for value in encoded]
                        except (TypeError, ValueError) as exc:
                            raise TypeError(
                                "Model returned embedding with non-numeric values"
                            ) from exc
                    else:
                        # existing fallback or error path
                        raise TypeError("Model returned non-iterable embedding")
                except Exception as exc:  # pragma: no cover - model failures are rare
                    logger.warning("Embedding failed for '%s': %s", normalized, exc)
                    vector = self._fallback_embedding(cache_key)
                    fallback_triggered = True

        if fallback_triggered:
            self._using_fallback_embeddings = True
        elif normalized:
            self._using_fallback_embeddings = False

        await self._store_cache(cache_key, vector)
        return vector

    @property
    def is_model_available(self) -> bool:
        """Return ``True`` when the real embedding model dependency is available."""

        return self._model_dependency_available

    @property
    def using_fallback_embeddings(self) -> bool:
        """Return ``True`` when recent encodes relied on deterministic fallbacks."""

        return self._using_fallback_embeddings

    async def _ensure_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:  # pragma: no cover - safeguarded by caller
            raise RuntimeError("SentenceTransformer dependency missing")
        async with self._model_lock:
            if self._model is None:
                logger.info("Loading embedding model '%s'", self._model_name)
                self._model = await asyncio.to_thread(
                    SentenceTransformer, self._model_name
                )
                if not self._explicit_fallback_dimensions:
                    try:
                        dimension = int(self._model.get_sentence_embedding_dimension())
                    except AttributeError:
                        logger.debug(
                            "Model '%s' does not expose an embedding dimension; keeping fallback size at %s",
                            self._model_name,
                            self._fallback_dimensions,
                        )
                    else:
                        if dimension > 0:
                            self._fallback_dimensions = dimension
        return self._model

    async def _get_cached(self, key: str) -> list[float] | None:
        async with self._cache_lock:
            cached = await self._cache.get(key)
            if cached is None:
                self._cache_index.pop(key, None)
                return None
            logger.debug("Embedding cache hit for '%s'", key)
            self._record_cache_key(key)
            return [float(value) for value in cached]

    async def _store_cache(self, key: str, vector: list[float]) -> None:
        evictions: list[str] = []
        async with self._cache_lock:
            await self._cache.set(key, list(vector), ttl=self._cache_ttl)
            if key in self._cache_index:
                self._cache_index.move_to_end(key)
            else:
                self._cache_index[key] = None
            while len(self._cache_index) > self._cache_max_entries:
                oldest, _ = self._cache_index.popitem(last=False)
                if oldest != key:
                    evictions.append(oldest)
        for victim in evictions:
            await self._evict(victim)

    def _record_cache_key(self, key: str) -> None:
        if key in self._cache_index:
            self._cache_index.move_to_end(key)
        else:
            self._cache_index[key] = None

    async def _evict(self, key: str) -> None:
        caches = getattr(self._cache, "caches", [])
        for cache in caches:
            delete = getattr(cache, "delete", None)
            if delete is None:
                continue
            try:
                result = delete(key)
                if inspect.isawaitable(result):
                    await result
            except (
                Exception
            ) as exc:  # pragma: no cover - cache eviction failures are rare
                logger.debug(
                    "Failed to evict key '%s' from %s: %s",
                    key,
                    cache.__class__.__name__,
                    exc,
                )

    def _create_driver(self) -> Any:
        if AsyncGraphDatabase is None:
            logger.debug("Neo4j driver not installed; using no-op driver")
            return _NoOpDriver()

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not (uri and user and password):
            logger.debug("Neo4j credentials missing; using no-op driver")
            return _NoOpDriver()
        try:
            driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            logger.info("Connected to Neo4j at %s", uri)
            return driver
        except Exception as exc:  # pragma: no cover - driver not present in tests
            logger.warning("Failed to initialise Neo4j driver: %s", exc)
            return _NoOpDriver()

    def _fallback_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        required = self._fallback_dimensions
        repeated = (digest * ((required // len(digest)) + 1))[:required]
        vector = [(byte / 255.0) * 2 - 1 for byte in repeated]
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return [0.0] * required
        return [value / magnitude for value in vector]
