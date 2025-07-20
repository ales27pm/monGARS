from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from monGARS.config import get_settings
from monGARS.core.cortex.curiosity_engine import CuriosityEngine
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.llm_integration import LLMIntegration
from monGARS.core.mains_virtuelles import ImageCaptioning
from monGARS.core.mimicry import MimicryModule
from monGARS.core.neuro_symbolic.advanced_reasoner import AdvancedReasoner
from monGARS.core.personality import PersonalityEngine

logger = logging.getLogger(__name__)
settings = get_settings()


class Orchestrator:
    """Coordinate modules to handle user queries."""

    def __init__(self) -> None:
        self.llm = LLMIntegration()
        self.reasoner = AdvancedReasoner()
        self.dynamic_response = AdaptiveResponseGenerator()
        self.mimicry = MimicryModule()
        self.curiosity = CuriosityEngine()
        self.personality = PersonalityEngine()
        self.captioner = ImageCaptioning()

    async def process_query(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        image_data: bytes | None = None,
    ) -> dict:
        """Process a user query end-to-end."""
        start_time = datetime.utcnow()
        if image_data:
            caption = await self.captioner.generate_caption(image_data)
            if caption:
                logger.info("Image caption generated: %s", caption)
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
            logger.info(
                "Query refined with neuro-symbolic reasoning: %s", refined_query
            )
        else:
            refined_query = query

        llm_response = await self.llm.generate_response(refined_query)
        base_response = llm_response.get("text", "")
        user_personality = await self.personality.analyze_personality(user_id, [])
        adapted_response = await self.dynamic_response.generate_adaptive_response(
            base_response, user_personality
        )
        interaction_data = {
            "feedback": 0.8,
            "response_time": (datetime.utcnow() - start_time).total_seconds(),
        }
        await self.mimicry.update_profile(user_id, interaction_data)
        final_response = await self.mimicry.adapt_response_style(
            adapted_response, user_id
        )
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await self.log_interaction(
            user_id,
            refined_query,
            final_response,
            llm_response.get("confidence", 0.9),
            processing_time,
        )
        return {
            "text": final_response,
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
    ) -> None:
        logger.info(
            "User: %s | Query: %s | Response: %s | Confidence: %s | Processing Time: %.2fs",
            user_id,
            query,
            response,
            confidence,
            processing_time,
        )

    async def run_loop(self) -> None:
        """Simple periodic loop for background tasks."""
        while True:
            await asyncio.sleep(60)
            logger.info("Orchestrator periodic task executed.")
