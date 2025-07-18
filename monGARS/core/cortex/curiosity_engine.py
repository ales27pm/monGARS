import asyncio
import logging

import httpx
import spacy
from sqlalchemy import text

from monGARS.config import get_settings
from monGARS.core.iris import Iris
from monGARS.core.neurones import EmbeddingSystem

from ...init_db import async_session_factory

logger = logging.getLogger(__name__)
settings = get_settings()


class CuriosityEngine:
    def __init__(self, iris: Iris | None = None):
        self.embedding_system = EmbeddingSystem()
        self.knowledge_gap_threshold = 0.65
        self.nlp = spacy.load("fr_core_news_sm")
        self.iris = iris or Iris()

    async def detect_gaps(self, conversation_context: dict) -> dict:
        last_query = conversation_context.get("last_query", "")
        context_embedding = await self.embedding_system.encode(last_query)
        similar_count = await self._vector_similarity_search(context_embedding)
        if similar_count >= 3:
            return {"status": "sufficient_knowledge"}
        doc = self.nlp(last_query)
        entities = [ent.text for ent in doc.ents]
        missing_entities = []
        for entity in entities:
            if not await self._check_entity_in_kg(entity):
                missing_entities.append(entity)
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

    async def _vector_similarity_search(self, embedding: list) -> int:
        async with async_session_factory() as session:
            query = text(
                "SELECT COUNT(*) FROM conversation_history WHERE vector <-> :embedding < 0.5"
            )
            result = await session.execute(query, {"embedding": embedding})
            count = result.scalar() or 0
            return count

    async def _check_entity_in_kg(self, entity: str) -> bool:
        try:
            async with self.embedding_system.driver.session() as session:
                result = await session.run(
                    "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($entity) RETURN count(n) > 0 AS exists",
                    entity=entity,
                )
                record = await result.single()
                return record["exists"]
        except Exception as e:
            logger.error(f"Error checking entity in KG: {e}")
            return False

    def _formulate_research_query(
        self, missing_entities: list, original_query: str
    ) -> str:
        return original_query + " " + " ".join(missing_entities)

    async def _perform_research(self, query: str) -> str:
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
                    summary = " ".join(doc["summary"] for doc in documents)
                    return f"Contexte supplémentaire: {summary}"
            except Exception as e:
                logger.error(f"Document retrieval error: {e}")

            logger.info("Falling back to Iris for query: %s", query)
            result = await self.iris.search(query)
            if result:
                return f"Contexte supplémentaire: {result}"
            return "Aucun contexte supplémentaire trouvé."
