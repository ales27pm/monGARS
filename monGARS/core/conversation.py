import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Optional

from monGARS.config import get_settings
from monGARS.core.cortex.curiosity_engine import CuriosityEngine
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.evolution_engine import EvolutionEngine
from monGARS.core.hippocampus import Hippocampus
from monGARS.core.llm_integration import LLMIntegration
from monGARS.core.mains_virtuelles import ImageCaptioning
from monGARS.core.mimicry import MimicryModule
from monGARS.core.neuro_symbolic.advanced_reasoner import AdvancedReasoner
from monGARS.core.neurones import EmbeddingSystem
from monGARS.core.persistence import PersistenceRepository, VectorMatch
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
        self.speaker = speaker or SpeakerService()
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

    async def _refine_query(
        self, query: str, user_id: str
    ) -> tuple[str, Mapping[str, Any]]:
        result = await self.reasoner.reason(query, user_id)
        if not isinstance(result, Mapping):
            from hashlib import blake2s

            redacted = blake2s(user_id.encode("utf-8"), digest_size=4).hexdigest()
            logger.warning(
                "conversation.reasoner.invalid_result",
                extra={
                    "user": f"u:{redacted}",
                    "result_type": type(result).__name__,
                },
            )
            result = {}
        refined = f"{query} {result['result']}" if "result" in result else query
        return refined, result

    def _build_response_hints(
        self, reasoning: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        """Translate reasoning metadata into LLM response hints."""

        if not isinstance(reasoning, Mapping):
            return None

        detail = reasoning.get("result")
        if isinstance(detail, str) and detail.strip():
            return {"reasoning": True}

        return None

    def _determine_task_type(self, original_query: str) -> str:
        """Route prompts to specialised model roles when appropriate."""

        infer = getattr(self.llm, "infer_task_type", None)
        if callable(infer):
            try:
                return infer(original_query)
            except Exception:  # pragma: no cover - defensive guardrail
                logger.exception("conversation.task_type_inference_failed")
        return "general"

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

    async def _semantic_context_matches(
        self,
        *,
        user_id: str,
        query: str,
        history_pairs: Sequence[tuple[str, str]],
    ) -> list[dict[str, object]]:
        """Retrieve semantically-related conversation history for prompt grounding."""

        limit = max(int(getattr(settings, "llm2vec_context_limit", 0)), 0)
        if limit == 0:
            return []

        trimmed_query = query.strip()
        if not trimmed_query:
            return []

        if not hasattr(self.persistence, "vector_search_history"):
            return []

        distance_cutoff_setting = getattr(
            settings, "llm2vec_context_max_distance", None
        )
        distance_cutoff: float | None
        if (
            isinstance(distance_cutoff_setting, (int, float))
            and distance_cutoff_setting > 0
        ):
            distance_cutoff = float(distance_cutoff_setting)
        else:
            distance_cutoff = None

        try:
            matches = await self.persistence.vector_search_history(
                user_id,
                trimmed_query,
                limit=limit,
                max_distance=distance_cutoff,
            )
        except Exception:  # pragma: no cover - persistence errors surfaced to logs
            from hashlib import blake2s

            redacted = blake2s(user_id.encode("utf-8"), digest_size=4).hexdigest()
            logger.exception(
                "conversation.semantic_context.lookup_failed",
                extra={"user": f"u:{redacted}"},
            )
            return []

        if not matches:
            return []

        history_lookup = {
            ((query_text or "").strip(), (response_text or "").strip())
            for query_text, response_text in history_pairs
        }

        semantic_results: list[dict[str, object]] = []
        seen_ids: set[object] = set()

        for match in matches:
            if not isinstance(match, VectorMatch):
                continue
            record = match.record
            record_id = getattr(record, "id", None)
            if record_id is not None:
                if record_id in seen_ids:
                    continue
                seen_ids.add(record_id)

            query_text = (getattr(record, "query", "") or "").strip()
            response_text = (getattr(record, "response", "") or "").strip()
            if not query_text and not response_text:
                continue

            if (query_text, response_text) in history_lookup:
                continue

            entry: dict[str, object] = {
                "id": record_id,
                "query": query_text,
                "response": response_text,
                "timestamp": getattr(record, "timestamp", None),
            }

            distance = getattr(match, "distance", None)
            if isinstance(distance, (int, float)):
                distance_value = float(distance)
                entry["distance"] = distance_value
                entry["similarity"] = max(0.0, min(1.0, 1.0 - distance_value))

            semantic_results.append(entry)
            if len(semantic_results) >= limit:
                break

        return semantic_results

    def _compose_prompt(
        self,
        refined_prompt: str,
        *,
        history_pairs: Sequence[tuple[str, str]],
        semantic_context: Sequence[dict[str, object]],
    ) -> str:
        """Render the final prompt combining history and semantic recall."""

        sections: list[str] = []

        if history_pairs:
            history_lines: list[str] = []
            for idx, (query_text, response_text) in enumerate(history_pairs, start=1):
                user_line = (query_text or "").strip()
                assistant_line = (response_text or "").strip()
                history_lines.append(
                    f"[{idx}] User: {user_line}\n    Assistant: {assistant_line}"
                )
            sections.append(
                "Recent conversation turns (most recent first):\n"
                + "\n".join(history_lines)
            )

        if semantic_context:
            semantic_lines: list[str] = []
            for idx, entry in enumerate(semantic_context, start=1):
                similarity = entry.get("similarity")
                similarity_text = (
                    f" (similarity {similarity:.3f})"
                    if isinstance(similarity, float)
                    else ""
                )
                query_text = (entry.get("query") or "").strip()
                response_text = (entry.get("response") or "").strip()
                semantic_lines.append(
                    f"[{idx}]{similarity_text} User: {query_text}\n    Assistant: {response_text}"
                )
            sections.append(
                "Archived interactions retrieved via semantic search:\n"
                + "\n".join(semantic_lines)
            )

        instructions = (
            "Leverage the provided context to craft an accurate and concise reply. "
            "If the context is unrelated, continue with your best effort response. "
            "Current user request:\n"
            f"{refined_prompt}"
        )
        sections.append(instructions)

        return "\n\n".join(section for section in sections if section.strip())

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
        refined_prompt, reasoning_metadata = await self._refine_query(
            augmented_query, user_id
        )

        semantic_context = await self._semantic_context_matches(
            user_id=user_id,
            query=augmented_query,
            history_pairs=history_pairs,
        )
        prompt = self._compose_prompt(
            refined_prompt,
            history_pairs=history_pairs,
            semantic_context=semantic_context,
        )
        response_hints = self._build_response_hints(reasoning_metadata)
        task_type = self._determine_task_type(original_query)

        llm_out = await self.llm.generate_response(
            prompt,
            task_type=task_type,
            response_hints=response_hints,
        )
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

        speech_session_id = session_id or user_id
        speech_turn = await self.speaker.speak(final, session_id=speech_session_id)

        await self.persistence.save_interaction(
            Interaction(
                user_id=user_id,
                session_id=session_id,
                input_data={
                    "original_query": original_query,
                    "with_image": query_with_image,
                    "augmented_query": augmented_query,
                    "refined_prompt": refined_prompt,
                    "reasoning_metadata": dict(reasoning_metadata),
                    "semantic_context": semantic_context,
                    "semantic_prompt": prompt,
                    "llm_task_type": task_type,
                    "llm_response_hints": response_hints,
                },
                output_data={
                    "raw_llm": llm_out,
                    "adapted_text": final,
                    "speech_turn": speech_turn.to_payload(),
                    "llm_source": llm_out.get("source"),
                    "llm_adapter_version": llm_out.get("adapter_version"),
                },
                message=augmented_query,
                response=final,
                personality=personality_traits,
                context={
                    "history": history_pairs,
                    "semantic_matches": semantic_context,
                },
                meta_data="",
                confidence=llm_out.get("confidence", 0.0),
                processing_time=processing_time,
            ),
            history_query=augmented_query,
            history_response=final,
        )
        memory_item = await self.memory.store(user_id, augmented_query, final)
        if memory_item is not None:
            await self.evolution_engine.record_memory_sample(
                user_id=memory_item.user_id,
                query=memory_item.query,
                response=memory_item.response,
                timestamp=memory_item.timestamp,
                expires_at=memory_item.expires_at,
            )
        return {
            "text": speech_turn.text,
            "confidence": llm_out.get("confidence", 0.0),
            "processing_time": processing_time,
            "speech_turn": speech_turn.to_payload(),
        }
