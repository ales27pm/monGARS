import asyncio
import logging
from datetime import datetime

from sqlalchemy import desc, select, update

from monGARS.config import get_settings
from monGARS.core.bouche import Bouche
from monGARS.core.caching.tiered_cache import get_cached_data
from monGARS.core.cortex.curiosity_engine import CuriosityEngine
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.evolution_engine import EvolutionEngine
from monGARS.core.hippocampus import Hippocampus
from monGARS.core.init_db import ConversationHistory, Interaction, async_session_factory
from monGARS.core.llm_integration import LLMIntegration
from monGARS.core.mains_virtuelles import ImageCaptioning
from monGARS.core.mimicry import MimicryModule
from monGARS.core.neuro_symbolic.advanced_reasoner import AdvancedReasoner
from monGARS.core.neurones import EmbeddingSystem
from monGARS.core.personality import PersonalityEngine

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationalModule:
    def __init__(self):
        self.embedding_system = EmbeddingSystem()
        self.llm = LLMIntegration()
        self.reasoner = AdvancedReasoner()
        self.dynamic_response = AdaptiveResponseGenerator()
        self.evolution_engine = EvolutionEngine()
        self.mimicry_module = MimicryModule()
        self.curiosity = CuriosityEngine()
        self.personality_engine = PersonalityEngine()
        self.captioner = ImageCaptioning()
        self.hippocampus = Hippocampus()
        self.bouche = Bouche()

    async def _save_interaction(
        self,
        user_id: str,
        session_id: str,
        input_data: str,
        output_data: str,
        message: str,
        response: str,
        personality: dict,
        context: dict,
        meta_data: str,
        confidence: float,
        processing_time: float,
    ):
        async with async_session_factory() as session:
            try:
                new_interaction = Interaction(
                    user_id=user_id,
                    session_id=session_id,
                    input_data=input_data,
                    output_data=output_data,
                    message=message,
                    response=response,
                    personality=personality,
                    context=context,
                    meta_data=meta_data,
                    confidence=confidence,
                    processing_time=processing_time,
                )
                session.add(new_interaction)
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Error saving interaction: {e}")
                raise

    async def _get_conversation_history(self, user_id: str, limit: int = 10):
        async with async_session_factory() as session:
            try:
                result = await session.execute(
                    select(ConversationHistory)
                    .where(ConversationHistory.user_id == user_id)
                    .order_by(desc(ConversationHistory.timestamp))
                    .limit(limit)
                )
                history = result.scalars().all()
                return [
                    {
                        "query": entry.query,
                        "response": entry.response,
                        "timestamp": entry.timestamp,
                    }
                    for entry in history
                ]
            except Exception as e:
                logger.error(f"Error retrieving conversation history: {e}")
                return []

    async def generate_response(
        self, user_id: str, query: str, session_id: str = None, image_data: bytes = None
    ) -> dict:
        start_time = datetime.utcnow()
        if image_data:
            caption = await self.captioner.generate_caption(image_data)
            logger.info(f"Image caption generated: {caption}")
            query = f"{query} Description de l'image: {caption}"
        conversation_context = {"last_query": query}
        gap_info = await self.curiosity.detect_gaps(conversation_context)
        if gap_info.get("status") == "insufficient_knowledge":
            query += " " + gap_info.get("additional_context", "")
            logger.info(
                "Query augmented with additional context from curiosity engine."
            )
        reason_result = await self.reasoner.reason(query, user_id)
        if "result" in reason_result:
            refined_query = f"{query} {reason_result['result']}"
            logger.info(f"Query refined with neuro-symbolic reasoning: {refined_query}")
        else:
            refined_query = query
        llm_response = await self.llm.generate_response(refined_query)
        base_response = llm_response.get("text", "")
        user_personality = await self.personality_engine.analyze_personality(
            user_id, []
        )
        adapted_response = await self.dynamic_response.generate_adaptive_response(
            base_response, user_personality
        )
        interaction_data = {
            "feedback": 0.8,
            "response_time": (datetime.utcnow() - start_time).total_seconds(),
        }
        await self.mimicry_module.update_profile(user_id, interaction_data)
        final_response = await self.mimicry_module.adapt_response_style(
            adapted_response, user_id
        )
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await self.hippocampus.store(user_id, refined_query, final_response)
        await self.log_interaction(
            user_id,
            refined_query,
            final_response,
            llm_response.get("confidence", 0.9),
            processing_time,
        )
        spoken_response = await self.bouche.speak(final_response)
        return {
            "text": spoken_response,
            "confidence": llm_response.get("confidence", 0.9),
            "processing_time": processing_time,
        }

    async def log_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        confidence: float,
        processing_time: float,
    ):
        logger.info(
            f"User: {user_id} | Query: {query} | Response: {response} | Confidence: {confidence} | Processing Time: {processing_time:.2f}s"
        )

    async def run_loop(self):
        while True:
            await asyncio.sleep(60)
            logger.info("Orchestrator periodic task executed.")
