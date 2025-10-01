from __future__ import annotations

import asyncio
import inspect
import logging
import math
import types
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, Callable

import httpx
import spacy
from cachetools import TTLCache

try:  # pragma: no cover - optional dependency at import time
    from sqlalchemy import select
except ImportError:  # pragma: no cover - SQLAlchemy not installed in tests
    select = None  # type: ignore[assignment]

from opentelemetry import metrics

from monGARS.config import get_settings
from monGARS.core.iris import Iris
from monGARS.core.neurones import EmbeddingSystem

try:  # pragma: no cover - optional dependency at import time
    from ...init_db import ConversationHistory, async_session_factory
except (ImportError, AttributeError):  # pragma: no cover - database optional
    ConversationHistory = None  # type: ignore[assignment]
    async_session_factory = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
settings = get_settings()
meter = metrics.get_meter(__name__)
_external_research_counter = meter.create_counter(
    "curiosity_external_research_requests",
    unit="1",
    description="Number of external research requests initiated by the curiosity engine.",
)
_kg_lookup_counter = meter.create_counter(
    "curiosity_kg_lookup_events",
    unit="1",
    description="Knowledge graph lookup hits and misses for curiosity gap detection.",
)
_research_cache_counter = meter.create_counter(
    "curiosity_research_cache_events",
    unit="1",
    description="Cache events for external research queries triggered by the curiosity engine.",
)

AsyncClientFactory = Callable[[], AsyncIterator[httpx.AsyncClient]]


def _tokenize(text: str) -> set[str]:
    """Return a lower-cased token set without empty strings."""

    return {token for token in text.lower().split() if token}


class CuriosityEngine:
    """Detect knowledge gaps and trigger research fetches when required."""

    _MAX_HISTORY_CANDIDATES = 50
    _HISTORY_KEY_PRIORITY: tuple[str, ...] = ("query", "message", "prompt", "text")

    def __init__(
        self,
        iris: Iris | None = None,
        *,
        http_client_factory: AsyncClientFactory | None = None,
    ) -> None:
        """Initialise the curiosity engine with NLP and embedding utilities."""

        self.embedding_system = EmbeddingSystem()
        self.similarity_threshold = settings.curiosity_similarity_threshold
        self.similar_history_threshold = max(
            0, settings.curiosity_minimum_similar_history
        )
        self.graph_gap_cutoff = max(1, settings.curiosity_graph_gap_cutoff)
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:  # pragma: no cover - optional model
            logger.warning(
                "spaCy model unavailable; falling back to rule-based entity detection.",
            )

            def _dummy(text: str) -> types.SimpleNamespace:
                return types.SimpleNamespace(ents=[])

            self.nlp = _dummy
        self.iris = iris or Iris()
        self._kg_cache_ttl = max(5, getattr(settings, "curiosity_kg_cache_ttl", 300))
        self._kg_cache_max_entries = max(
            16, getattr(settings, "curiosity_kg_cache_max_entries", 512)
        )
        self._kg_cache: TTLCache[str, bool] = TTLCache(
            maxsize=self._kg_cache_max_entries,
            ttl=self._kg_cache_ttl,
        )
        self._kg_cache_lock = asyncio.Lock()
        self._kg_inflight_lock = asyncio.Lock()
        self._kg_inflight: dict[str, asyncio.Future[bool]] = {}
        self._research_cache_ttl = max(
            30, getattr(settings, "curiosity_research_cache_ttl", 900)
        )
        self._research_cache_max_entries = max(
            8, getattr(settings, "curiosity_research_cache_max_entries", 256)
        )
        self._research_cache: TTLCache[str, str] = TTLCache(
            maxsize=self._research_cache_max_entries,
            ttl=self._research_cache_ttl,
        )
        self._research_cache_lock = asyncio.Lock()

        if http_client_factory is None:

            @asynccontextmanager
            async def default_http_client_factory() -> AsyncIterator[httpx.AsyncClient]:
                async with httpx.AsyncClient() as client:
                    yield client

            self._http_client_factory = default_http_client_factory
        else:
            self._http_client_factory = http_client_factory

    async def detect_gaps(self, conversation_context: dict) -> dict:
        """Identify knowledge gaps using previous history and entity checks."""

        try:
            last_query = conversation_context.get("last_query", "").strip()
            if not last_query:
                return {"status": "sufficient_knowledge"}
            history_queries = self._extract_history_queries(
                conversation_context.get("history")
            )
            similar_count = await self._vector_similarity_search(
                last_query, history_queries
            )
            if (
                self.similar_history_threshold
                and similar_count >= self.similar_history_threshold
            ):
                return {"status": "sufficient_knowledge"}
            entities = await self._extract_entities(last_query)
            entity_presence = await self._check_entities_in_kg_batch(entities)
            missing_entities = [
                entity for entity in entities if not entity_presence.get(entity, False)
            ]
            if len(missing_entities) >= self.graph_gap_cutoff:
                research_query = self._formulate_research_query(
                    missing_entities, last_query
                )
                additional_context = await self._perform_research(research_query)
                return {
                    "status": "insufficient_knowledge",
                    "additional_context": additional_context,
                    "research_query": research_query,
                }
            return {"status": "sufficient_knowledge"}
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.error(
                "curiosity.detect_gaps.error",
                exc_info=True,
                extra={
                    "has_context": bool(conversation_context),
                    "error": str(exc),
                },
            )
            return {"status": "sufficient_knowledge"}
        except Exception:
            logger.exception(
                "curiosity.detect_gaps.unexpected_error",
                extra={"has_context": bool(conversation_context)},
            )
            raise

    async def _vector_similarity_search(
        self, query_text: str, fallback_history: Sequence[str] | None = None
    ) -> int:
        """Count previous queries similar to ``query_text`` using vectors when available."""

        query_terms = _tokenize(query_text)
        if not query_terms:
            return 0

        db_history = await self._load_recent_queries_from_db(limit=25)
        combined_history: list[object] = list(db_history)
        if fallback_history:
            combined_history.extend(fallback_history)

        history_candidates = self._prepare_history_candidates(
            combined_history,
            exclude=query_text,
            limit=self._MAX_HISTORY_CANDIDATES,
        )
        if not history_candidates:
            return 0

        return await self._count_similarities(
            query_text, history_candidates, query_terms
        )

    async def _load_recent_queries_from_db(self, *, limit: int) -> list[str]:
        if (
            select is None
            or ConversationHistory is None
            or async_session_factory is None
        ):
            logger.debug(
                "Vector similarity search skipped; persistence dependencies missing.",
            )
            return []
        try:
            session_factory = async_session_factory()
        except TypeError:
            logger.debug("Vector similarity fallback due to invalid session factory")
            return []
        try:
            async with session_factory as session:
                result = await session.execute(
                    select(ConversationHistory.query)
                    .order_by(ConversationHistory.timestamp.desc())
                    .limit(limit)
                )
        except Exception as exc:  # pragma: no cover - DB optional
            logger.debug("Vector similarity fallback due to DB error: %s", exc)
            return []

        history: list[str] = []
        try:
            scalars = result.scalars()
            candidate_rows = scalars.all()
            rows: Iterable[str]
            if inspect.isawaitable(candidate_rows):
                rows = await candidate_rows  # type: ignore[assignment]
            else:
                rows = candidate_rows  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            rows = [row[0] for row in result if row and isinstance(row[0], str)]
        for previous in rows:
            if not isinstance(previous, str):
                continue
            if cleaned := previous.strip():
                history.append(cleaned)
        return history

    async def _count_similarities(
        self,
        query_text: str,
        history_candidates: Sequence[str],
        query_terms: set[str],
    ) -> int:
        if not history_candidates:
            return 0

        if not self.embedding_system.is_model_available:
            logger.debug("Vector similarity skipped; embedding model unavailable")
            return self._count_token_similarity(query_terms, history_candidates)

        try:
            query_vector, query_used_fallback = await self.embedding_system.encode(
                query_text
            )
        except Exception as exc:  # pragma: no cover - embedding optional
            logger.debug(
                "Vector similarity fallback due to embedding error: %s",
                exc,
            )
            return self._count_token_similarity(query_terms, history_candidates)

        if query_used_fallback:
            logger.debug("Vector similarity fallback due to query embedding fallback")
            return self._count_token_similarity(query_terms, history_candidates)

        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            logger.debug(
                "Vector similarity fallback due to zero-length query embedding"
            )
            return self._count_token_similarity(query_terms, history_candidates)

        try:
            history_results = await asyncio.gather(
                *(self.embedding_system.encode(item) for item in history_candidates)
            )
        except Exception as exc:  # pragma: no cover - embedding optional
            logger.debug(
                "Vector similarity fallback due to embedding error: %s",
                exc,
            )
            return self._count_token_similarity(query_terms, history_candidates)

        if len(history_results) != len(history_candidates):
            logger.debug(
                "Vector similarity fallback due to embedding count mismatch",
            )
            return self._count_token_similarity(query_terms, history_candidates)

        similar = 0
        for _candidate_text, (history_vector, history_used_fallback) in zip(
            history_candidates, history_results
        ):
            if history_used_fallback:
                logger.debug(
                    "Vector similarity fallback due to history embedding fallback",
                )
                return self._count_token_similarity(query_terms, history_candidates)
            if len(history_vector) != len(query_vector):
                logger.debug(
                    "Vector similarity fallback due to embedding length mismatch",
                )
                return self._count_token_similarity(query_terms, history_candidates)
            other_norm = math.sqrt(sum(value * value for value in history_vector))
            if other_norm == 0:
                continue
            dot = sum(
                q_value * h_value
                for q_value, h_value in zip(query_vector, history_vector)
            )
            similarity = dot / (query_norm * other_norm)
            if similarity >= self.similarity_threshold:
                similar += 1
        return similar

    def _count_token_similarity(
        self, query_terms: set[str], history_candidates: Iterable[str]
    ) -> int:
        similar = 0
        for previous in history_candidates:
            previous_terms = _tokenize(previous)
            if not previous_terms:
                continue
            overlap = query_terms.intersection(previous_terms)
            similarity = len(overlap) / len(query_terms)
            if similarity >= self.similarity_threshold:
                similar += 1
        return similar

    async def _extract_entities(self, query: str) -> list[str]:
        """Extract entities for *query* without blocking the event loop."""

        cleaned = query.strip()
        if not cleaned:
            return []

        if inspect.iscoroutinefunction(self.nlp):
            doc = await self.nlp(cleaned)
        else:
            doc = await asyncio.to_thread(self.nlp, cleaned)
        entities: list[str] = []
        for ent in getattr(doc, "ents", []):
            text = getattr(ent, "text", "")
            if not isinstance(text, str):
                continue
            if cleaned_text := text.strip():
                entities.append(cleaned_text)
        return entities

    def _extract_history_queries(self, history: Iterable[object] | None) -> list[str]:
        if not history:
            return []
        return [
            entry
            for raw_entry in history
            if (entry := self._normalise_history_entry(raw_entry)) is not None
        ]

    def _prepare_history_candidates(
        self,
        history: Iterable[object],
        *,
        exclude: str,
        limit: int,
    ) -> list[str]:
        """Normalise, deduplicate, and bound the history list for similarity checks."""

        deduplicated: list[str] = []
        seen: set[str] = set()
        exclude_key = exclude.strip().lower()
        for raw_entry in history:
            candidate = self._normalise_history_entry(raw_entry)
            if candidate is None:
                continue
            key = candidate.lower()
            if key == exclude_key or key in seen:
                continue
            seen.add(key)
            deduplicated.append(candidate)
            if len(deduplicated) >= limit:
                break
        return deduplicated

    def _normalise_history_entry(self, entry: object) -> str | None:
        """Extract a string query from history entries.

        The extraction order for mapping-based entries prioritises the
        ``query`` field, followed by ``message``, ``prompt``, and ``text`` to
        reflect how conversation records are persisted today. The precedence is
        documented to avoid ambiguity when multiple keys are present.
        """

        candidate: str | None = None
        if isinstance(entry, str):
            candidate = entry
        elif isinstance(entry, dict):
            for key in self._HISTORY_KEY_PRIORITY:
                value = entry.get(key)
                if isinstance(value, str):
                    candidate = value
                    break
        elif isinstance(entry, (list, tuple)) and entry:
            first = entry[0]
            if isinstance(first, str):
                candidate = first
        if candidate and (cleaned := candidate.strip()):
            return cleaned
        return None

    async def _check_entities_in_kg_batch(
        self, entities: Sequence[str]
    ) -> dict[str, bool]:
        """Return entity presence flags using batching and an LRU cache."""

        if not entities:
            return {}

        normalized_map = self._normalise_entities(entities)
        if not normalized_map:
            return {}

        loop = asyncio.get_running_loop()
        cached, missing = await self._collect_cache_hits(normalized_map.keys())
        if cached:
            _kg_lookup_counter.add(len(cached), {"outcome": "hit"})
        waiters, pending = await self._register_inflight_waiters(missing, loop)

        fetched_results: dict[str, bool] = {}
        if pending:
            _kg_lookup_counter.add(len(pending), {"outcome": "miss"})
            try:
                fetched_results = await self._query_kg_entities(pending)
            except asyncio.CancelledError:
                fetched_results = {normalized: False for normalized in pending}
                raise
            except Exception:  # pragma: no cover - defensive logging below
                logger.exception("curiosity.batch_lookup.unexpected_error")
                fetched_results = {normalized: False for normalized in pending}
            finally:
                await self._complete_inflight(pending, fetched_results)

        combined: dict[str, bool] = dict(cached)
        for normalized, future in waiters.items():
            try:
                combined[normalized] = await future
            except asyncio.CancelledError:
                raise
            except Exception:
                combined[normalized] = False

        entity_presence: dict[str, bool] = {}
        for normalized, originals in normalized_map.items():
            value = combined.get(normalized, False)
            for original in originals:
                entity_presence[original] = value
        return entity_presence

    async def _query_kg_entities(
        self, normalized_entities: Sequence[str]
    ) -> dict[str, bool]:
        """Resolve entity presence flags from the knowledge graph driver."""

        if not normalized_entities:
            return {}

        driver = getattr(self.embedding_system, "driver", None)
        session_factory = getattr(driver, "session", None)
        if not callable(session_factory):
            logger.debug(
                "Knowledge graph batch lookup skipped; driver session unavailable.",
            )
            return {normalized: False for normalized in normalized_entities}

        try:
            async with session_factory() as session:
                result = await session.run(
                    (
                        "UNWIND $entities AS entity "
                        "OPTIONAL MATCH (n) "
                        "WHERE toLower(n.name) CONTAINS entity "
                        "RETURN entity AS normalized, count(n) > 0 AS exists"
                    ),
                    entities=list(normalized_entities),
                )
        except Exception as exc:  # pragma: no cover - driver optional
            logger.debug(
                "Knowledge graph batch lookup failed for %s entities: %s",
                len(normalized_entities),
                exc,
            )
            return {normalized: False for normalized in normalized_entities}

        rows = await self._consume_result_rows(result)
        query_results: dict[str, bool] = {}
        for row in rows:
            normalized = str(row.get("normalized", "")).strip()
            if not normalized:
                continue
            exists = bool(row.get("exists"))
            query_results[normalized] = exists
        for normalized in normalized_entities:
            query_results.setdefault(normalized, False)
        return query_results

    async def _check_entity_in_kg(self, entity: str) -> bool:
        """Backward-compatible single-entity lookup that leverages batching."""

        cleaned = entity.strip() if isinstance(entity, str) else None
        if not cleaned:
            return False
        result = await self._check_entities_in_kg_batch([cleaned])
        return result.get(cleaned, False)

    async def _consume_result_rows(self, result: Any) -> list[dict[str, Any]]:
        """Normalise driver result objects into a list of dictionaries.

        Neo4j's async driver exposes several shapes depending on how results are
        consumed (``result.data()``, ``result.records()`` or async iteration).
        Tests also exercise lightweight stubs that mimic a subset of that
        interface. To keep the engine decoupled from a specific driver, this
        helper attempts the common access patterns in order and coerces each row
        into a mapping.
        """

        if result is None:
            return []

        rows = await self._call_result_method(result, "data")
        if rows is not None:
            return [await self._coerce_row(row) for row in rows]

        records = await self._call_result_method(result, "records")
        if records is not None:
            return [await self._coerce_row(record) for record in records]

        normalised_records: list[dict[str, Any]] = []
        aiter_method = getattr(result, "__aiter__", None)
        if callable(aiter_method):
            async for record in result:
                normalised_records.append(await self._coerce_row(record))
        return normalised_records

    def _normalise_entities(self, entities: Sequence[str]) -> dict[str, list[str]]:
        """Return a mapping of normalised entity keys to the original forms."""

        normalized_map: dict[str, list[str]] = {}
        for raw_entity in entities:
            if not isinstance(raw_entity, str):
                continue
            cleaned = raw_entity.strip()
            if not cleaned:
                continue
            normalized = cleaned.lower()
            normalized_map.setdefault(normalized, []).append(cleaned)
        return normalized_map

    async def _collect_cache_hits(
        self, normalized_keys: Iterable[str]
    ) -> tuple[dict[str, bool], list[str]]:
        """Return cached results and the keys missing from the cache."""

        cached: dict[str, bool] = {}
        missing: list[str] = []
        async with self._kg_cache_lock:
            for normalized in normalized_keys:
                try:
                    cached[normalized] = self._kg_cache[normalized]
                except KeyError:
                    missing.append(normalized)
        return cached, missing

    async def _register_inflight_waiters(
        self,
        normalized_keys: Iterable[str],
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[dict[str, asyncio.Future[bool]], list[str]]:
        """Register in-flight futures for ``normalized_keys`` and return waiters."""

        waiters: dict[str, asyncio.Future[bool]] = {}
        pending: list[str] = []
        async with self._kg_inflight_lock:
            for normalized in normalized_keys:
                future = self._kg_inflight.get(normalized)
                if future is None:
                    future = loop.create_future()
                    self._kg_inflight[normalized] = future
                    pending.append(normalized)
                waiters[normalized] = future
        return waiters, pending

    async def _complete_inflight(
        self, pending: Iterable[str], results: Mapping[str, bool]
    ) -> None:
        """Resolve pending futures and refresh the cache with ``results``."""

        resolved = {
            normalized: results.get(normalized, False) for normalized in pending
        }
        async with self._kg_inflight_lock:
            futures = {
                normalized: self._kg_inflight.pop(normalized, None)
                for normalized in pending
            }
        for normalized, future in futures.items():
            if future and not future.done():
                future.set_result(resolved[normalized])
        if resolved:
            async with self._kg_cache_lock:
                for normalized, value in resolved.items():
                    self._kg_cache[normalized] = value

    async def _call_result_method(self, result: Any, method_name: str) -> Any | None:
        """Invoke ``method_name`` on ``result`` if available and return the value."""

        method = getattr(result, method_name, None)
        if not callable(method):
            return None
        try:
            value = method()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "curiosity.result_method.error %s",
                exc,
                extra={"method": method_name, "error": str(exc)},
            )
            return None
        if inspect.isawaitable(value):
            try:
                value = await value
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "curiosity.result_method.await_error %s",
                    exc,
                    extra={"method": method_name, "error": str(exc)},
                )
                return None
        return value

    async def _coerce_row(self, row: Any) -> dict[str, Any]:
        """Convert *row* into a dictionary, awaiting ``data()`` if provided."""

        data_getter = getattr(row, "data", None)
        if callable(data_getter):
            try:
                value = data_getter()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "curiosity.result_row.data_error %s",
                    exc,
                    extra={"error": str(exc)},
                )
            else:
                if inspect.isawaitable(value):
                    try:
                        value = await value
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.debug(
                            "curiosity.result_row.await_error %s",
                            exc,
                            extra={"error": str(exc)},
                        )
                        value = None
                if isinstance(value, dict):
                    return dict(value)
                if value is not None:
                    try:
                        return dict(value)
                    except (TypeError, ValueError) as exc:
                        logger.debug(
                            "curiosity.result_row.coercion_error %s",
                            exc,
                            extra={"error": str(exc)},
                        )
                return {}
        if isinstance(row, dict):
            return dict(row)
        if hasattr(row, "_asdict"):
            return row._asdict()
        try:
            return dict(row)
        except (TypeError, ValueError) as exc:
            logger.debug(
                "curiosity.result_row.fallback_error %s",
                exc,
                extra={"error": str(exc)},
            )
            return {}

    def _formulate_research_query(
        self, missing_entities: list[str], original_query: str
    ) -> str:
        """Combine the original prompt and missing entities into a query."""

        terms = [original_query.strip(), *missing_entities]
        seen: set[str] = set()
        normalised: list[str] = []
        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalised.append(cleaned)
        return " ".join(normalised)

    async def _perform_research(self, query: str) -> str:
        """Fetch additional context from the document service or Iris."""

        normalised_query = query.strip()
        if not normalised_query:
            return "Aucun contexte supplémentaire trouvé."

        cached_response = await self._get_cached_research(normalised_query)
        if cached_response is not None:
            _research_cache_counter.add(1, {"event": "hit"})
            return cached_response

        _research_cache_counter.add(1, {"event": "miss"})
        _external_research_counter.add(1, {"channel": "document_service"})

        document_summary = await self._fetch_document_summary(normalised_query)
        if document_summary:
            enriched = f"Contexte supplémentaire: {document_summary}"
            await self._set_cached_research(normalised_query, enriched)
            return enriched

        logger.info(
            "curiosity.iris_fallback",
            extra={"query_len": len(normalised_query)},
        )
        _external_research_counter.add(1, {"channel": "iris"})
        result = await self.iris.search(normalised_query)
        if result:
            enriched = f"Contexte supplémentaire: {result}"
            await self._set_cached_research(normalised_query, enriched)
            return enriched
        fallback = "Aucun contexte supplémentaire trouvé."
        await self._set_cached_research(normalised_query, fallback)
        return fallback

    async def _fetch_document_summary(self, query: str) -> str:
        """Return a concatenated summary from the document retrieval service."""

        try:
            async with self._http_client_factory() as client:
                response = await client.post(
                    f"{settings.DOC_RETRIEVAL_URL}/api/search",
                    json={"query": query},
                    timeout=10,
                )
                response.raise_for_status()
                documents = response.json().get("documents", [])
        except httpx.HTTPStatusError as exc:  # pragma: no cover - depends on network
            logger.warning(
                "curiosity.document_service.status_error",
                extra={
                    "status_code": exc.response.status_code,
                    "query_len": len(query),
                },
            )
            return ""
        except Exception as exc:  # pragma: no cover - network optional
            logger.debug("Document retrieval error: %s", exc)
            return ""

        return self._summarise_documents(documents)

    async def _get_cached_research(self, query: str) -> str | None:
        """Return a cached research result for ``query`` if available."""

        async with self._research_cache_lock:
            return self._research_cache.get(query)

    async def _set_cached_research(self, query: str, value: str) -> None:
        """Store ``value`` in the research cache for ``query``."""

        async with self._research_cache_lock:
            self._research_cache[query] = value

    def _summarise_documents(self, documents: Sequence[Mapping[str, Any]]) -> str:
        """Combine document summaries into a compact string."""

        cleaned_summaries: list[str] = []
        for document in documents:
            if not isinstance(document, Mapping):
                continue
            summary = document.get("summary")
            if isinstance(summary, str) and (stripped := summary.strip()):
                cleaned_summaries.append(stripped)
        return " ".join(cleaned_summaries)
