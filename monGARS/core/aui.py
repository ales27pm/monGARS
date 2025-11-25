from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from .llm_integration import GuardRejectionError, LLMIntegration

logger = logging.getLogger(__name__)

# Default actions (keys MUST match the front-end data-action)
DEFAULT_ACTIONS: list[tuple[str, str]] = [
    ("code", "Write, refactor or generate source code and tests."),
    ("summarize", "Summarize long passages, chats, or documents succinctly."),
    ("explain", "Explain a concept in simpler terms with examples."),
]


class LLMActionSuggester:
    def __init__(self, context_window: int = 1024):
        self.context_window = context_window
        try:
            self.llm = LLMIntegration.instance()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning(
                "llm_action_suggester_init_failed", extra={"error": repr(exc)}
            )
            self.llm = None

    @property
    def model_name(self) -> str:
        return "dolphin3-reasoning-baseline" if self.llm else "heuristic-keyword"

    def suggest(self, prompt: str, actions: List[str], context: dict) -> List[str]:
        """Generate ranked action suggestions using LLM reasoning"""
        if not actions:
            return []

        trimmed_context = self._trim_context_to_tokens(context, max_tokens=256)
        serialized_context = json.dumps(trimmed_context, ensure_ascii=False)
        fallback_used = False

        if not self.llm:
            fallback_used = True
            result = self._heuristic_fallback(prompt, actions)
        else:
            try:
                response = self.llm.generate(
                    f"""
        You are Dolphin. Rank these actions by relevance to user intent:
        USER INTENT: '{prompt}'
        ACTIONS: {json.dumps(actions)}
        CONTEXT: {serialized_context}
        Respond ONLY with JSON array of action names:
        """,
                    max_new_tokens=100,
                    context=context,
                )
                try:
                    ranked = json.loads(self._extract_json(response))
                except Exception as exc:  # pragma: no cover - defensive log
                    logger.error(
                        "llm_action_suggester_json_parse_failed",
                        extra={
                            "error": repr(exc),
                            "response_excerpt": response[:200],
                        },
                        exc_info=True,
                    )
                    raise
                if not isinstance(ranked, list):
                    logger.error(
                        "llm_action_suggester_non_list_response",
                        extra={"parsed_type": type(ranked).__name__},
                    )
                    raise TypeError("LLM response must be a list")
                filtered = [a for a in ranked if a in actions]
                deduped: list[str] = []
                seen: set[str] = set()
                for action in filtered:
                    if isinstance(action, str) and action not in seen:
                        deduped.append(action)
                        seen.add(action)
                result = deduped or self._heuristic_fallback(prompt, actions)
                fallback_used = not bool(deduped)
            except GuardRejectionError:
                fallback_used = True
                result = self._heuristic_fallback(prompt, actions)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                fallback_used = True
                logger.error(
                    "llm_action_suggester_invalid_response",
                    extra={"error": repr(exc)},
                    exc_info=True,
                )
                result = self._heuristic_fallback(prompt, actions)
            except Exception as exc:  # pragma: no cover - runtime fallback
                fallback_used = True
                logger.exception(
                    "llm_action_suggester_runtime_failure",
                    extra={"error": repr(exc)},
                )
                result = self._heuristic_fallback(prompt, actions)

        logger.info(
            "suggestion_generated",
            extra={
                "event": "suggestion_generated",
                "model": self.model_name,
                "user_intent": prompt[:50],
                "action_count": len(actions),
                "suggested_count": len(result),
                "context_size": len(serialized_context),
                "fallback_used": fallback_used,
            },
        )
        return result

    def _trim_context_to_tokens(
        self, context: dict | None, *, max_tokens: int
    ) -> dict[str, Any]:
        if not isinstance(context, dict) or max_tokens <= 0:
            return {}

        token_limit = max(1, max_tokens)

        tokenizer: Any | None = None
        if self.llm is not None:
            try:
                tokenizer = self.llm.tokenizer
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug(
                    "llm_action_suggester_tokenizer_unavailable",
                    extra={"error": repr(exc)},
                )
                tokenizer = None

        def estimate_tokens(serialized: str) -> int:
            if tokenizer is not None:
                try:
                    return len(tokenizer.encode(serialized, add_special_tokens=False))
                except Exception:  # pragma: no cover - tokenizer errors fallback
                    logger.debug(
                        "llm_action_suggester_tokenizer_failed",
                        exc_info=True,
                    )
            return max(1, len(serialized) // 4)

        def normalize(value: Any) -> Any:
            if isinstance(value, str):
                return value[:token_limit]
            if isinstance(value, (int, float, bool)) or value is None:
                return value
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                normalized_dict: dict[str, Any] = {}
                for key, val in value.items():
                    normalized_dict[str(key)] = normalize(val)
                return normalized_dict
            return str(value)

        trimmed: dict[str, Any] = {}
        for key, value in context.items():
            normalized_value = normalize(value)
            candidate = dict(trimmed)
            candidate[str(key)] = normalized_value
            try:
                serialized = json.dumps(candidate, ensure_ascii=False)
            except TypeError:
                normalized_value = str(value)
                candidate[str(key)] = normalized_value
                serialized = json.dumps(candidate, ensure_ascii=False)
            if estimate_tokens(serialized) > token_limit:
                break
            trimmed[str(key)] = normalized_value
        return trimmed

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response"""
        if not text:
            return "[]"
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped

        if code_block := re.search(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", stripped, re.IGNORECASE
        ):
            return code_block[1]

        start_index = -1
        depth = 0
        for index, char in enumerate(stripped):
            if char == "[":
                if depth == 0:
                    start_index = index
                depth += 1
            elif char == "]" and depth > 0:
                depth -= 1
                if depth == 0 and start_index != -1:
                    return stripped[start_index:index + 1]

        if inline := re.search(r"\[(?:[^\]\[]|\[[^\]]*\])*\]", stripped):
            return inline[0]

        return "[]"

    def _heuristic_fallback(self, prompt: str, actions: List[str]) -> List[str]:
        """Fallback to simple heuristic when LLM fails"""
        prompt_lower = prompt.lower()
        return sorted(
            actions,
            key=lambda x: prompt_lower.count(x.lower()),
            reverse=True,
        )
