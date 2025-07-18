import logging
from datetime import datetime
from typing import Optional

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
        self.dynamic = dynamic or AdaptiveResponseGenerator()
        self.mimicry = mimicry or MimicryModule()
        self.personality = personality or PersonalityEngine()
        self.captioner = captioner or ImageCaptioning()
        self.memory = memory or MemoryService(Hippocampus())
        self.speaker = speaker or SpeakerService(Bouche())
        self.persistence = persistence or PersistenceRepository()
        self.evolution_engine = EvolutionEngine()

    async def _handle_image(self, query: str, image_data: bytes | None) -> str:
        if not image_data:
            return query
        caption = await self.captioner.generate_caption(image_data)
        return f"{query} Description de l'image: {caption}"

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

    async def _adapt_response(self, text: str, user_id: str) -> str:
        personality = await self.personality.analyze_personality(user_id, [])
        adaptive = await self.dynamic.generate_adaptive_response(text, personality)
        await self.mimicry.update_profile(
            user_id,
            {"feedback": 0.8},
        )
        return await self.mimicry.adapt_response_style(adaptive, user_id)

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

        query = await self._handle_image(query, image_data)
        query = await self._augment_with_curiosity(query, history_pairs)
        refined = await self._refine_query(query, user_id)

        llm_out = await self.llm.generate_response(refined)
        final = await self._adapt_response(llm_out.get("text", ""), user_id)

        await self.persistence.save_interaction(
            Interaction(
                user_id=user_id,
                session_id=session_id,
                input_data=refined,
                output_data=final,
                message=query,
                response=final,
                personality={},
                context={},
                meta_data="",
                confidence=llm_out.get("confidence", 0.0),
                processing_time=(datetime.utcnow() - start).total_seconds(),
            )
        )
        await self.memory.store(user_id, refined, final)

        spoken = await self.speaker.speak(final)
        elapsed = (datetime.utcnow() - start).total_seconds()
        return {
            "text": spoken,
            "confidence": llm_out.get("confidence", 0.0),
            "processing_time": elapsed,
        }
