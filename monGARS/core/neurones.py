"""Neural utilities used by the cognition stack."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

try:  # pragma: no cover - optional dependency
    from neo4j.async_driver import AsyncGraphDatabase
except ImportError:  # pragma: no cover - driver not installed in tests
    AsyncGraphDatabase = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - tests run without the heavy model
    SentenceTransformer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


@dataclass(slots=True)
class _CacheEntry:
    expires_at: float
    vector: list[float]


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
        fallback_dimensions: int = 32,
    ) -> None:
        self._model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", _DEFAULT_MODEL
        )
        self._model: SentenceTransformer | None = None
        self._model_lock = asyncio.Lock()
        self._cache: dict[str, _CacheEntry] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_ttl = cache_ttl
        self._fallback_dimensions = max(1, fallback_dimensions)
        self._now = time.monotonic
        self.driver = driver or self._create_driver()

    async def close(self) -> None:
        close_callable = getattr(self.driver, "close", None)
        if not close_callable:
            return
        try:
            result = close_callable()
        except TypeError:
            # The driver may expose ``close`` as an async method defined on the
            # class rather than a bound coroutine function.
            result = close_callable(self.driver)  # type: ignore[misc]
        if inspect.isawaitable(result):
            await result

    async def encode(self, text: str) -> list[float]:
        """Return an embedding for *text* with caching and graceful fallbacks."""

        normalized = text.strip()
        if not normalized:
            return [0.0] * self._fallback_dimensions

        cached = await self._get_cached(normalized)
        if cached is not None:
            return cached

        vector: list[float]
        if SentenceTransformer is None:
            logger.debug(
                "SentenceTransformer not available; using fallback embedding for '%s'",
                normalized,
            )
            vector = self._fallback_embedding(normalized)
        else:
            model = await self._ensure_model()
            try:
                encoded = await asyncio.to_thread(
                    model.encode,
                    normalized,
                    normalize_embeddings=True,
                )
                if isinstance(encoded, Iterable):
                    vector = [float(value) for value in encoded]
                else:
                    raise TypeError("Model returned non-iterable embedding")
            except Exception as exc:  # pragma: no cover - model failures are rare
                logger.warning("Embedding failed for '%s': %s", normalized, exc)
                vector = self._fallback_embedding(normalized)

        await self._store_cache(normalized, vector)
        return vector

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
        return self._model

    async def _get_cached(self, text: str) -> list[float] | None:
        async with self._cache_lock:
            entry = self._cache.get(text)
            if entry and entry.expires_at > self._now():
                logger.debug("Embedding cache hit for '%s'", text)
                return entry.vector
            if entry:
                del self._cache[text]
        return None

    async def _store_cache(self, text: str, vector: list[float]) -> None:
        async with self._cache_lock:
            self._cache[text] = _CacheEntry(
                expires_at=self._now() + self._cache_ttl,
                vector=list(vector),
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
        return [(byte / 255.0) * 2 - 1 for byte in repeated]
