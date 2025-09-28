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
        self.personality = PersonalityEngine()
        self.dynamic_response = AdaptiveResponseGenerator(self.personality)
        self.mimicry = MimicryModule()
        self.curiosity = CuriosityEngine()
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
        try:
            if image_data:
                try:
                    caption = await asyncio.wait_for(
                        self.captioner.generate_caption(image_data), timeout=5
                    )
                except Exception as exc:
                    logger.error("Captioning failed: %s", exc)
                    caption = ""
                if caption:
                    logger.info("Image caption generated: %s", caption)
                    query = f"{query} {settings.caption_prefix} {caption}"

            conversation_context = {"last_query": query}
            try:
                gap_info = await asyncio.wait_for(
                    self.curiosity.detect_gaps(conversation_context), timeout=5
                )
            except Exception as exc:
                logger.error("Curiosity engine failed: %s", exc)
                gap_info = {}
            if gap_info.get("status") == "insufficient_knowledge":
                query += " " + gap_info.get("additional_context", "")
                logger.info(
                    "Query augmented with additional context from curiosity engine."
                )

            try:
                reason_result = await asyncio.wait_for(
                    self.reasoner.reason(query, user_id), timeout=10
                )
            except Exception as exc:
                logger.error("Reasoner failed: %s", exc)
                reason_result = {}
            if "result" in reason_result:
                refined_query = f"{query} {reason_result['result']}"
                logger.info(
                    "Query refined with neuro-symbolic reasoning: %s", refined_query
                )
            else:
                refined_query = query

            try:
                llm_response = await asyncio.wait_for(
                    self.llm.generate_response(refined_query), timeout=30
                )
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                llm_response = {"text": "", "confidence": 0.0}
            base_response = llm_response.get("text", "")
            try:
                user_personality = await asyncio.wait_for(
                    self.dynamic_response.get_personality_traits(user_id, []),
                    timeout=5,
                )
            except Exception as exc:
                logger.error("Personality analysis failed: %s", exc)
                user_personality = {}
            try:
                adapted_response = self.dynamic_response.generate_adaptive_response(
                    base_response, user_personality
                )
            except Exception as exc:
                logger.error("Adaptive response generation failed: %s", exc)
                adapted_response = base_response
            interaction_data = {
                # Using a neutral baseline until real feedback collection is implemented
                "feedback": 0.8,
                "response_time": (datetime.utcnow() - start_time).total_seconds(),
            }
            try:
                await asyncio.wait_for(
                    self.mimicry.update_profile(user_id, interaction_data),
                    timeout=5,
                )
            except Exception as exc:
                logger.error("Mimicry update failed: %s", exc)
            try:
                final_response = await asyncio.wait_for(
                    self.mimicry.adapt_response_style(adapted_response, user_id),
                    timeout=5,
                )
            except Exception as exc:
                logger.error("Mimicry adaptation failed: %s", exc)
                final_response = adapted_response
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.log_interaction(
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
        except Exception as exc:  # pragma: no cover - unexpected failure
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error("Process query failed for %s: %s", user_id, exc, exc_info=True)
            return {
                "text": "An error occurred while processing your request.",
                "confidence": 0.0,
                "processing_time": processing_time,
                "error": str(exc),
            }

    def log_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        confidence: float,
        processing_time: float,
    ) -> None:
        safe_query = query[:100]
        safe_response = response[:100]
        logger.debug(
            "User: %s | Query: %s | Response: %s | Confidence: %s | Processing Time: %.2fs",
            user_id,
            safe_query,
            safe_response,
            confidence,
            processing_time,
        )

    async def run_loop(self) -> None:
        """Simple periodic loop for background tasks."""
        while True:
            await asyncio.sleep(60)
            logger.info("Orchestrator periodic task executed.")
