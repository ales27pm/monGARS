import logging
from datetime import datetime
from typing import Optional

from monGARS.config import get_settings
from monGARS.core.bouche import Bouche
from monGARS.core.cortex.curiosity_engine import CuriosityEngine
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.evolution_engine import EvolutionEngine
from monGARS.core.hippocampus import Hippocampus
from monGARS.core.llm_integration import LLMIntegration
from monGARS.core.mains_virtuelles import ImageCaptioning
from monGARS.core.mimicry import MimicryModule
from monGARS.core.neuro_symbolic.advanced_reasoner import AdvancedReasoner
from monGARS.core.neurones import EmbeddingSystem
from monGARS.core.persistence import PersistenceRepository
from monGARS.core.personality import PersonalityEngine
from monGARS.core.services import MemoryService, SpeakerService

from ..init_db import Interaction

logger = logging.getLogger(__name__)
settings = get_settings()


class ConversationalModule:
    def __init__(
        self,
        llm: Optional[LLMIntegration] = None,
        reasoner: Optional[AdvancedReasoner] = None,
        curiosity: Optional[CuriosityEngine] = None,
        dynamic: Optional[AdaptiveResponseGenerator] = None,
        mimicry: Optional[MimicryModule] = None,
        personality: Optional[PersonalityEngine] = None,
        captioner: Optional[ImageCaptioning] = None,
        memory: Optional[MemoryService] = None,
        speaker: Optional[SpeakerService] = None,
        persistence: Optional[PersistenceRepository] = None,
    ) -> None:
        self.embedding_system = EmbeddingSystem()
        self.llm = llm or LLMIntegration()
        self.reasoner = reasoner or AdvancedReasoner()
        self.curiosity = curiosity or CuriosityEngine()
        self.personality = personality or PersonalityEngine()
        self.dynamic = dynamic or AdaptiveResponseGenerator(self.personality)
        self.mimicry = mimicry or MimicryModule()
        self.captioner = captioner or ImageCaptioning()
        self.memory = memory or MemoryService(Hippocampus())
        self.speaker = speaker or SpeakerService(Bouche())
        self.persistence = persistence or PersistenceRepository()
        self.evolution_engine = EvolutionEngine()

    async def _handle_image(self, query: str, image_data: bytes | None) -> str:
        if not image_data:
            return query
        caption = await self.captioner.generate_caption(image_data)
        return f"{query} {settings.caption_prefix} {caption}"

    async def _augment_with_curiosity(
        self, query: str, history: list[tuple[str, str]]
    ) -> str:
        context = {"last_query": query, "history": history}
        gap = await self.curiosity.detect_gaps(context)
        if gap.get("status") == "insufficient_knowledge":
            return f"{query} {gap.get('additional_context', '')}"
        return query

    async def _refine_query(self, query: str, user_id: str) -> str:
        result = await self.reasoner.reason(query, user_id)
        return f"{query} {result['result']}" if "result" in result else query

    async def _adapt_response(
        self,
        text: str,
        user_id: str,
        interactions: list[dict[str, str]],
        user_message: str,
    ) -> tuple[str, dict]:
        personality = await self.dynamic.get_personality_traits(user_id, interactions)
        adaptive = self.dynamic.generate_adaptive_response(
            text, personality, user_id=user_id
        )
        await self.mimicry.update_profile(
            user_id,
            {"message": user_message, "response": text},
        )
        styled = await self.mimicry.adapt_response_style(adaptive, user_id)
        return styled, personality

    async def generate_response(
        self,
        user_id: str,
        query: str,
        session_id: str | None = None,
        image_data: bytes | None = None,
    ) -> dict:
        start = datetime.utcnow()
        history_items = await self.memory.history(user_id, limit=5)
        history_pairs = [(m.query, m.response) for m in history_items]

        original_query = query
        query_with_image = await self._handle_image(query, image_data)
        augmented_query = await self._augment_with_curiosity(
            query_with_image, history_pairs
        )
        refined = await self._refine_query(augmented_query, user_id)

        llm_out = await self.llm.generate_response(refined)
        recent_interactions = [
            {"message": query_text, "response": response_text}
            for query_text, response_text in history_pairs
        ]
        final, personality_traits = await self._adapt_response(
            llm_out.get("text", ""),
            user_id,
            recent_interactions,
            original_query,
        )

        processing_time = (datetime.utcnow() - start).total_seconds()

        speech_turn = await self.speaker.speak(final)

        await self.persistence.save_interaction(
            Interaction(
                user_id=user_id,
                session_id=session_id,
                input_data={
                    "original_query": original_query,
                    "with_image": query_with_image,
                    "augmented_query": augmented_query,
                    "refined_prompt": refined,
                },
                output_data={
                    "raw_llm": llm_out,
                    "adapted_text": final,
                    "speech_turn": speech_turn.to_payload(),
                },
                message=augmented_query,
                response=final,
                personality=personality_traits,
                context={"history": history_pairs},
                meta_data="",
                confidence=llm_out.get("confidence", 0.0),
                processing_time=processing_time,
            ),
            history_query=augmented_query,
            history_response=final,
        )
        await self.memory.store(user_id, augmented_query, final)
        return {
            "text": speech_turn.text,
            "confidence": llm_out.get("confidence", 0.0),
            "processing_time": processing_time,
            "speech_turn": speech_turn.to_payload(),
        }
