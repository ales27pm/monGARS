import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import blake2s
from typing import Any, Optional

from monGARS.config import get_settings
from monGARS.core.cortex.curiosity_engine import CuriosityEngine
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.evolution_engine import EvolutionEngine
from monGARS.core.hippocampus import Hippocampus
from monGARS.core.inference_utils import (
    ChatPrompt,
    build_converged_chat_prompt,
    estimate_token_count,
)
from monGARS.core.llm_integration import (
    LLMIntegration,
    LLMRuntimeError,
    UnifiedLLMRuntime,
)
from monGARS.core.mains_virtuelles import ImageCaptioning
from monGARS.core.mimicry import MimicryModule
from monGARS.core.neuro_symbolic.advanced_reasoner import AdvancedReasoner
from monGARS.core.persistence import PersistenceRepository, VectorMatch
from monGARS.core.personality import PersonalityEngine
from monGARS.core.services import MemoryService, SpeakerService

from ..init_db import Interaction

logger = logging.getLogger(__name__)
settings = get_settings()
UTC = timezone.utc


class _FakeLLM:
    """Lightweight LLM double used in legacy unit tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        prompt: str,
        *,
        task_type: str = "general",
        response_hints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        call = {
            "prompt": prompt,
            "task_type": task_type,
            "response_hints": response_hints,
        }
        self.calls.append(call)
        return {
            "text": "sample-response",
            "confidence": 0.0,
            "source": "sample",
            "adapter_version": "test",
        }

    async def generate_response(
        self,
        prompt: str,
        *,
        task_type: str = "general",
        response_hints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.__call__(
            prompt, task_type=task_type, response_hints=response_hints
        )

    def prompt_token_limit(self, task_type: str = "general") -> int | None:
        return None

    def generation_token_target(self, task_type: str = "general") -> int | None:
        return None


class PromptTooLargeError(RuntimeError):
    """Raised when a composed prompt exceeds the configured token budget."""

    def __init__(self, prompt_tokens: int, limit: int) -> None:
        super().__init__(
            f"Prompt requires {prompt_tokens} tokens which exceeds the limit of {limit}"
        )
        self.prompt_tokens = prompt_tokens
        self.limit = limit


class _FakePersistence:
    """Persistence façade that records interactions for inspection in tests."""

    def __init__(self) -> None:
        self.vector_queries: list[dict[str, Any]] = []
        self.saved_interactions: list[dict[str, Any]] = []

    async def vector_search_history(
        self,
        user_id: str,
        query: str,
        *,
        limit: int,
        max_distance: float | None = None,
    ) -> list[VectorMatch]:
        self.vector_queries.append(
            {
                "user_id": user_id,
                "query": query,
                "limit": limit,
                "max_distance": max_distance,
            }
        )
        return []

    async def save_interaction(
        self,
        interaction: Interaction,
        *,
        history_query: str | None = None,
        history_response: str | None = None,
    ) -> None:
        self.saved_interactions.append(
            {
                "interaction": interaction,
                "history_query": history_query,
                "history_response": history_response,
            }
        )


class _FakeMimicry:
    """Trivial mimicry adapter that echoes inputs for backwards compatibility tests."""

    def __init__(self) -> None:
        self.profile_updates: list[tuple[str, Mapping[str, Any]]] = []
        self.style_requests: list[tuple[str, str]] = []

    async def update_profile(
        self, user_id: str, interaction: dict
    ) -> dict:  # pragma: no cover - simple recorder
        self.profile_updates.append((user_id, interaction))
        return interaction

    async def adapt_response_style(self, text: str, user_id: str) -> str:
        self.style_requests.append((text, user_id))
        return text


async def generate_response(
    *,
    prompt: str,
    llm: LLMIntegration | _FakeLLM | None = None,
    persistence: PersistenceRepository | _FakePersistence | None = None,
    mimicry: MimicryModule | _FakeMimicry | None = None,
) -> Mapping[str, Any]:
    """Legacy coroutine kept for backwards compatibility in older tests.

    The newer orchestration flow uses :class:`ConversationalModule`. Some
    integration tests still import this module-level helper directly, so we
    preserve a lightweight version that validates the provided doubles and
    funnels the call to the injected LLM implementation.
    """

    llm_impl = llm or _FakeLLM()
    persistence_impl = persistence or _FakePersistence()
    mimicry_impl = mimicry or _FakeMimicry()

    required_hooks = ("update_profile", "adapt_response_style")
    missing_hooks = [hook for hook in required_hooks if not hasattr(mimicry_impl, hook)]
    if missing_hooks:
        raise TypeError("mimicry component missing required hooks")

    response = await llm_impl.generate_response(
        prompt, task_type="general", response_hints=None
    )
    if not isinstance(response, Mapping):
        raise TypeError("LLM must return a mapping")

    text = response.get("text", "")
    try:
        await mimicry_impl.update_profile(
            "legacy-user", {"message": prompt, "response": text}
        )
        styled_text = await mimicry_impl.adapt_response_style(text, "legacy-user")
    except Exception as exc:  # pragma: no cover - defensive; surfaced to caller
        raise ValueError("mimicry post-processing failed") from exc

    if not isinstance(styled_text, str):
        raise TypeError("mimicry must return str")

    enriched_response = dict(response)
    enriched_response["text"] = styled_text

    if hasattr(persistence_impl, "save_interaction"):
        try:
            confidence_value = enriched_response.get("confidence")
            try:
                confidence = (
                    float(confidence_value) if confidence_value is not None else 0.0
                )
            except (TypeError, ValueError):
                confidence = 0.0

            interaction = Interaction(
                user_id="legacy-user",
                session_id=None,
                input_data={"prompt": prompt},
                output_data={"response": enriched_response},
                message=prompt,
                response=styled_text,
                personality={},
                context={},
                meta_data=None,
                confidence=confidence,
                processing_time=enriched_response.get("processing_time"),
            )
            await persistence_impl.save_interaction(
                interaction,
                history_query=prompt,
                history_response=styled_text,
            )
        except Exception:  # pragma: no cover - diagnostics for legacy hook
            prompt_hash = blake2s(prompt.encode("utf-8"), digest_size=4).hexdigest()
            logger.exception(
                "conversation.legacy_persistence_failed",
                extra={"prompt_hash": f"p:{prompt_hash}"},
            )

    return enriched_response


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
    ) -> ChatPrompt:
        """Render the final prompt combining history and semantic recall."""

        system_prompt = getattr(
            settings,
            "llm_system_prompt",
            "You are Dolphin, a helpful assistant.",
        )
        return build_converged_chat_prompt(
            refined_prompt,
            history_pairs=[
                ((query or "").strip(), (response or "").strip())
                for query, response in history_pairs
            ],
            semantic_context=semantic_context,
            system_prompt=system_prompt,
        )

    async def generate_response(
        self,
        user_id: str,
        query: str,
        session_id: str | None = None,
        image_data: bytes | None = None,
    ) -> dict:
        start = datetime.now(UTC)
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

        task_type = self._determine_task_type(original_query)

        semantic_context = await self._semantic_context_matches(
            user_id=user_id,
            query=augmented_query,
            history_pairs=history_pairs,
        )
        trimmed_history = list(history_pairs)
        trimmed_semantic = list(semantic_context)
        prompt_bundle = self._compose_prompt(
            refined_prompt,
            history_pairs=trimmed_history,
            semantic_context=trimmed_semantic,
        )
        response_hints = self._build_response_hints(reasoning_metadata)

        configured_limit = self.llm.prompt_token_limit(task_type)
        if isinstance(configured_limit, int) and configured_limit <= 0:
            configured_limit = None

        reserved_generation_tokens = 0
        generation_target = self.llm.generation_token_target(task_type)
        if isinstance(generation_target, int) and generation_target > 0:
            reserved_generation_tokens = generation_target

        budget = configured_limit
        if configured_limit is not None and reserved_generation_tokens:
            budget = max(configured_limit - reserved_generation_tokens, 1)

        prompt_tokens = estimate_token_count(prompt_bundle.chatml)
        trimmed = False
        while (
            budget and prompt_tokens > budget and (trimmed_history or trimmed_semantic)
        ):
            if trimmed_history:
                trimmed_history.pop(0)
            else:
                trimmed_semantic.pop()
            prompt_bundle = self._compose_prompt(
                refined_prompt,
                history_pairs=trimmed_history,
                semantic_context=trimmed_semantic,
            )
            prompt_tokens = estimate_token_count(prompt_bundle.chatml)
            trimmed = True

        if budget and prompt_tokens > budget:
            redacted = blake2s(user_id.encode("utf-8"), digest_size=4).hexdigest()
            logger.warning(
                "conversation.prompt.exceeds_limit",
                extra={
                    "user": f"u:{redacted}",
                    "prompt_tokens": prompt_tokens,
                    "token_limit": budget,
                    "configured_limit": configured_limit,
                    "reserved_generation_tokens": reserved_generation_tokens,
                },
            )
            raise PromptTooLargeError(prompt_tokens, budget)

        if trimmed:
            redacted = blake2s(user_id.encode("utf-8"), digest_size=4).hexdigest()
            logger.info(
                "conversation.prompt.trimmed",
                extra={
                    "user": f"u:{redacted}",
                    "prompt_tokens": prompt_tokens,
                    "token_limit": budget,
                    "configured_limit": configured_limit,
                    "reserved_generation_tokens": reserved_generation_tokens,
                    "history_entries": len(trimmed_history),
                    "semantic_entries": len(trimmed_semantic),
                },
            )
            history_pairs = list(trimmed_history)
            semantic_context = list(trimmed_semantic)
        else:
            history_pairs = trimmed_history
            semantic_context = trimmed_semantic

        llm_out: dict[str, Any]
        llm_adapter = getattr(self, "llm", None)
        use_adapter = bool(llm_adapter and hasattr(llm_adapter, "generate_response"))
        if use_adapter:
            try:
                response_mapping = await llm_adapter.generate_response(  # type: ignore[attr-defined]
                    prompt_bundle.text,
                    task_type=task_type,
                    response_hints=response_hints,
                    formatted_prompt=prompt_bundle.chatml,
                )
            except LLMRuntimeError:
                logger.exception("conversation.runtime.generate_failed")
                raise
            if not isinstance(response_mapping, Mapping):
                raise TypeError("LLM integration must return a mapping")
            llm_out = dict(response_mapping)
        else:
            runtime = UnifiedLLMRuntime.instance(settings)
            runtime_kwargs: dict[str, Any] = {}
            if generation_target:
                runtime_kwargs["max_new_tokens"] = generation_target
            try:
                response_text = await asyncio.to_thread(
                    runtime.generate, prompt_bundle.text, **runtime_kwargs
                )
            except LLMRuntimeError:
                logger.exception("conversation.runtime.generate_failed")
                raise
            llm_out = {
                "text": response_text,
                "confidence": 0.0,
                "source": "unified-runtime",
                "adapter_version": "unified",
            }
        llm_out.setdefault("prompt_tokens", prompt_tokens)
        if configured_limit is not None:
            llm_out.setdefault("prompt_token_limit", configured_limit)
        if budget is not None:
            llm_out.setdefault("prompt_token_budget", budget)
        if reserved_generation_tokens:
            llm_out.setdefault("prompt_token_reservation", reserved_generation_tokens)
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

        processing_time = (datetime.now(UTC) - start).total_seconds()

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
                    "semantic_prompt": prompt_bundle.text,
                    "chatml_prompt": prompt_bundle.chatml,
                    "prompt_tokens": prompt_tokens,
                    "prompt_token_limit": configured_limit,
                    "prompt_token_budget": budget,
                    "prompt_token_reservation": reserved_generation_tokens,
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
