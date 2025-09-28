from __future__ import annotations

import logging
import types
from typing import Iterable

import httpx
import spacy
from sqlalchemy import select

from monGARS.config import get_settings
from monGARS.core.iris import Iris
from monGARS.core.neurones import EmbeddingSystem

from ...init_db import ConversationHistory, async_session_factory

logger = logging.getLogger(__name__)
settings = get_settings()


def _tokenize(text: str) -> set[str]:
    """Return a lower-cased token set without empty strings."""

    return {token for token in text.lower().split() if token}


class CuriosityEngine:
    """Detect knowledge gaps and trigger research fetches when required."""

    def __init__(self, iris: Iris | None = None) -> None:
        """Initialise the curiosity engine with NLP and embedding utilities."""

        self.embedding_system = EmbeddingSystem()
        self.knowledge_gap_threshold = 0.5
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except Exception:  # pragma: no cover - optional model
            logger.warning(
                "spaCy model unavailable; falling back to rule-based entity detection."
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
            similar_count = await self._vector_similarity_search(last_query)
            if similar_count >= 3:
                return {"status": "sufficient_knowledge"}
            doc = self.nlp(last_query)
            entities = [ent.text for ent in getattr(doc, "ents", [])]
            missing_entities = [
                entity
                for entity in entities
                if not await self._check_entity_in_kg(entity)
            ]
            if missing_entities:
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "curiosity.detect_gaps.error",
                exc_info=True,
                extra={
                    "has_context": bool(conversation_context),
                    "error": str(exc),
                },
            )
            return {"status": "sufficient_knowledge"}

    async def _vector_similarity_search(self, query_text: str) -> int:
        """Count previous queries whose tokens overlap with the new one."""

        query_terms = _tokenize(query_text)
        if not query_terms:
            return 0
        try:
            async with async_session_factory() as session:
                result = await session.execute(
                    select(ConversationHistory.query)
                    .order_by(ConversationHistory.timestamp.desc())
                    .limit(25)
                )
                history: Iterable[str] = [row[0] for row in result if row[0]]
        except Exception as exc:  # pragma: no cover - DB optional
            logger.debug("Vector similarity fallback due to DB error: %s", exc)
            return 0
        similar = 0
        for previous in history:
            previous_terms = _tokenize(previous)
            if not previous_terms:
                continue
            overlap = query_terms.intersection(previous_terms)
            similarity = len(overlap) / len(query_terms)
            if similarity >= self.knowledge_gap_threshold:
                similar += 1
        return similar

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
        return " ".join(term for term in terms if term)

    async def _perform_research(self, query: str) -> str:
        """Fetch additional context from the document service or Iris."""

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

        logger.info("Falling back to Iris for query: %s", query)
        result = await self.iris.search(query)
        if result:
            return f"Contexte supplémentaire: {result}"
        return "Aucun contexte supplémentaire trouvé."
