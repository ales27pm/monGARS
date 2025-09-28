from __future__ import annotations

import asyncio
import inspect
import logging
import math
import types
from collections.abc import Iterable, Sequence

import httpx
import spacy

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


def _tokenize(text: str) -> set[str]:
    """Return a lower-cased token set without empty strings."""

    return {token for token in text.lower().split() if token}


class CuriosityEngine:
    """Detect knowledge gaps and trigger research fetches when required."""

    _MAX_HISTORY_CANDIDATES = 50
    _HISTORY_KEY_PRIORITY: tuple[str, ...] = ("query", "message", "prompt", "text")

    def __init__(self, iris: Iris | None = None) -> None:
        """Initialise the curiosity engine with NLP and embedding utilities."""

        self.embedding_system = EmbeddingSystem()
        self.knowledge_gap_threshold = settings.curiosity_similarity_threshold
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
            doc = self.nlp(last_query)
            entities = [ent.text for ent in getattr(doc, "ents", [])]
            missing_entities = [
                entity
                for entity in entities
                if not await self._check_entity_in_kg(entity)
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
            if similarity >= self.knowledge_gap_threshold:
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
            if similarity >= self.knowledge_gap_threshold:
                similar += 1
        return similar

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

    async def _check_entity_in_kg(self, entity: str) -> bool:
        """Return True if the entity is already known in the knowledge graph."""

        try:
            async with self.embedding_system.driver.session() as session:
                result = await session.run(
                    (
                        "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($entity) "
                        "RETURN count(n) > 0 AS exists"
                    ),
                    entity=entity,
                )
                record = await result.single()
                return bool(record.get("exists"))
        except Exception as exc:  # pragma: no cover - driver optional
            logger.debug("Knowledge graph lookup failed for %s: %s", entity, exc)
            return False

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

        _external_research_counter.add(1, {"channel": "document_service"})

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{settings.DOC_RETRIEVAL_URL}/api/search",
                    json={"query": query},
                    timeout=10,
                )
                response.raise_for_status()
                documents = response.json().get("documents", [])
                if documents:
                    summary = " ".join(
                        doc.get("summary", "")
                        for doc in documents
                        if doc.get("summary")
                    )
                    if summary:
                        return f"Contexte supplémentaire: {summary}"
            except Exception as exc:  # pragma: no cover - network optional
                logger.debug("Document retrieval error: %s", exc)

        logger.info(
            "curiosity.iris_fallback",
            extra={"query_len": len(query)},
        )
        _external_research_counter.add(1, {"channel": "iris"})
        result = await self.iris.search(query)
        if result:
            return f"Contexte supplémentaire: {result}"
        return "Aucun contexte supplémentaire trouvé."
