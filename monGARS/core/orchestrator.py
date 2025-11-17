from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import threading
import uuid
from concurrent.futures import Future
from datetime import datetime
from typing import Any, ClassVar, Coroutine, Optional, TypeVar

from monGARS.config import get_settings
from monGARS.core.cortex.curiosity_engine import CuriosityEngine
from monGARS.core.dynamic_response import AdaptiveResponseGenerator
from monGARS.core.llm_integration import (
    LLMIntegration,
    LLMRuntimeError,
    UnifiedLLMRuntime,
)
from monGARS.core.mains_virtuelles import ImageCaptioning
from monGARS.core.mimicry import MimicryModule
from monGARS.core.neuro_symbolic.advanced_reasoner import AdvancedReasoner
from monGARS.core.personality import PersonalityEngine
from monGARS.core.rag import (
    RagContextEnricher,
    RagDisabledError,
    RagServiceError,
)

logger = logging.getLogger(__name__)
settings = get_settings()
T = TypeVar("T")


class ToolReasoningMonitor:
    """Record trace lifecycle events for tool executions."""

    def start_trace(self, trace_id: str, metadata: dict[str, Any]) -> None:
        logger.info(
            "react.trace.start",
            extra={"trace_id": trace_id, **metadata},
        )

    def log_trace(self, trace_id: str, reasoning: str) -> None:
        logger.info(
            "react.trace.log",
            extra={"trace_id": trace_id, "reasoning": reasoning},
        )

    def log_error(
        self, trace_id: str, error_code: str, payload: dict[str, Any]
    ) -> None:
        logger.error(
            "react.trace.error",
            extra={"trace_id": trace_id, "code": error_code, **payload},
        )

    def complete_trace(self, trace_id: str, result: dict[str, Any]) -> None:
        logger.info(
            "react.trace.complete",
            extra={"trace_id": trace_id, "result": result},
        )


monitor = ToolReasoningMonitor()


def generate_uuid() -> str:
    """Generate a short random trace identifier for tool executions."""

    return str(uuid.uuid4())[:8]


class Orchestrator:
    """Coordinate modules to handle user queries."""

    def __init__(
        self,
        *,
        llm: LLMIntegration | None = None,
        reasoner: AdvancedReasoner | None = None,
        personality: PersonalityEngine | None = None,
        dynamic_response: AdaptiveResponseGenerator | None = None,
        mimicry: MimicryModule | None = None,
        curiosity: CuriosityEngine | None = None,
        captioner: ImageCaptioning | None = None,
    ) -> None:
        self.llm = llm or LLMIntegration()
        self.reasoner = reasoner or AdvancedReasoner()
        self.personality = personality or PersonalityEngine()
        self.dynamic_response = dynamic_response or AdaptiveResponseGenerator(
            self.personality
        )
        self.mimicry = mimicry or MimicryModule()
        self.curiosity = curiosity or CuriosityEngine()
        self.captioner = captioner or ImageCaptioning()

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

            runtime = UnifiedLLMRuntime.instance()
            try:
                response_text = await asyncio.wait_for(
                    asyncio.to_thread(runtime.generate, refined_query), timeout=30
                )
                llm_response = {"text": response_text, "confidence": 0.9}
            except (LLMRuntimeError, Exception) as exc:
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
                    base_response, user_personality, user_id=user_id
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


class ReActOrchestrator:
    """Execute tools with deterministic tracing and ReAct reasoning."""

    _SAFE_BUILTINS: ClassVar[dict[str, Any]] = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "sorted": sorted,
        "map": map,
        "filter": filter,
        "all": all,
        "any": any,
        "round": round,
        "print": print,
    }

    def __init__(self, rag_enricher: RagContextEnricher | None = None) -> None:
        self._rag_enricher = rag_enricher or RagContextEnricher()

    def execute_tool(self, tool_name: str, arguments: dict, context: dict) -> dict:
        """Execute tool after LLM generates reasoning trace"""
        trace_id = generate_uuid()
        monitor.start_trace(
            trace_id,
            {
                "tool": tool_name,
                "arguments": arguments,
                "context_user": context.get("user_id", "anonymous"),
            },
        )

        reasoning_prompt = f"""
        You are Dolphin, an AI assistant. Use ReAct format to solve this:
        [THOUGHT] Your step-by-step reasoning about which tool to use and why
        [ACTION] {{"tool": "{tool_name}", "arguments": {json.dumps(arguments)}}}

        Current context: {json.dumps(context)}
        """

        response = LLMIntegration.instance().generate(reasoning_prompt)

        if "[THOUGHT]" not in response:
            monitor.log_error(
                trace_id,
                "INVALID_REASONING_FORMAT",
                {"response": response[:200], "expected": "[THOUGHT] pattern missing"},
            )
            return {
                "error": "invalid_reasoning",
                "message": "LLM failed to generate proper reasoning trace",
            }

        monitor.log_trace(trace_id, response)

        if tool_name == "rag_search":
            result = self._rag_tool(arguments, context)
        elif tool_name == "code_execution":
            result = self._execute_code(arguments, context)
        else:
            result = {"error": "unknown_tool", "message": f"Tool {tool_name} not found"}

        monitor.complete_trace(trace_id, result)
        return result

    def _rag_tool(
        self, arguments: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str):
            fallback_keys = ("query", "prompt", "last_query", "message")
            query = next(
                (
                    str(context[key])
                    for key in fallback_keys
                    if isinstance(context.get(key), str)
                ),
                None,
            )
        if not query:
            return {
                "error": "missing_query",
                "message": "rag_search requires a query string",
            }

        repositories = self._parse_repositories(arguments.get("repositories"))
        max_results = self._coerce_int(arguments.get("max_results"))
        settings_max = getattr(settings, "rag_max_results", 8)
        limit = min(max_results, settings_max) if max_results else settings_max
        try:
            enrichment = self._run_async(
                self._rag_enricher.enrich(
                    query,
                    repositories=repositories,
                    max_results=limit,
                )
            )
        except RagDisabledError:
            return {
                "error": "rag_disabled",
                "message": "RAG enrichment is disabled in configuration",
            }
        except RagServiceError as exc:
            return {
                "error": "rag_service_error",
                "message": str(exc),
            }
        except Exception as exc:  # pragma: no cover - unexpected runtime failures
            logger.exception("rag_search.tool.failed", exc_info=exc)
            return {"error": "rag_failure", "message": str(exc)}

        return {
            "tool": "rag_search",
            "query": query,
            "focus_areas": enrichment.focus_areas,
            "references": [
                {
                    "repository": ref.repository,
                    "file_path": ref.file_path,
                    "summary": ref.summary,
                    "score": ref.score,
                    "url": ref.url,
                }
                for ref in enrichment.references
            ],
        }

    def _execute_code(
        self, arguments: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute trusted Python snippets inside a constrained sandbox."""

        # This helper is intended for operator-approved code only; untrusted
        # callers must never gain access to the ``code_execution`` tool.
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return {
                "error": "missing_code",
                "message": "code_execution requires non-empty code",
            }
        language = str(arguments.get("language", "python")).lower()
        if language not in {"python", "py"}:
            return {
                "error": "unsupported_language",
                "message": f"Language {language} is not supported",
            }

        safe_globals: dict[str, Any] = {"__builtins__": self._SAFE_BUILTINS.copy()}
        injected_locals: dict[str, Any] = {}
        context_locals = context.get("variables")
        if isinstance(context_locals, dict):
            for key, value in context_locals.items():
                if isinstance(key, str):
                    injected_locals[key] = value

        stdout_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(  # noqa: S102 - sandbox intentionally executes supplied code
                    compile(code, "<tool-exec>", "exec"), safe_globals, injected_locals
                )
        except Exception as exc:  # noqa: BLE001
            # Sandbox must report arbitrary execution failures back to the caller.
            return {
                "error": "execution_error",
                "message": str(exc),
                "stdout": stdout_buffer.getvalue(),
            }

        serialised_locals: dict[str, Any] = {}
        for key, value in injected_locals.items():
            if key.startswith("__"):
                continue
            try:
                json.dumps(value)
                serialised_locals[key] = value
            except TypeError:
                serialised_locals[key] = repr(value)

        return {
            "tool": "code_execution",
            "stdout": stdout_buffer.getvalue(),
            "variables": serialised_locals,
        }

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        future: Future[T] = Future()

        def runner() -> None:
            try:
                future.set_result(asyncio.run(coro))
            except Exception as exc:  # pragma: no cover - propagate failure
                future.set_exception(exc)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        return future.result()

    def _parse_repositories(self, repositories: Any) -> list[str] | None:
        if repositories is None:
            return None
        if isinstance(repositories, str):
            cleaned = repositories.strip()
            return [cleaned] if cleaned else None
        if isinstance(repositories, (list, tuple, set)):
            cleaned_list = [
                str(item).strip()
                for item in repositories
                if isinstance(item, str) and item.strip()
            ]
            return cleaned_list or None
        return None

    def _coerce_int(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None
