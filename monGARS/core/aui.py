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
        CONTEXT: {json.dumps(trimmed_context)}
        Respond ONLY with JSON array of action names:
        """,
                    max_new_tokens=100,
                )
                ranked = json.loads(self._extract_json(response))
                if not isinstance(ranked, list):
                    raise ValueError("LLM response must be a list")
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
            except (json.JSONDecodeError, ValueError):
                fallback_used = True
                result = self._heuristic_fallback(prompt, actions)
            except Exception:  # pragma: no cover - runtime fallback
                fallback_used = True
                result = self._heuristic_fallback(prompt, actions)

        logger.info(
            {
                "event": "suggestion_generated",
                "model": "dolphin3-reasoning-baseline",
                "user_intent": prompt[:50],
                "action_count": len(actions),
                "context_size": len(str(trimmed_context)),
                "fallback_used": fallback_used,
            }
        )
        return result

    def _trim_context_to_tokens(
        self, context: dict | None, *, max_tokens: int
    ) -> dict[str, Any]:
        if not context or not isinstance(context, dict):
            return {}

        token_limit = max(1, max_tokens)

        tokenizer: Any | None = None
        if self.llm is not None:
            try:
                tokenizer = self.llm.tokenizer
            except Exception:  # pragma: no cover - defensive guard
                tokenizer = None

        def token_count(value: Any) -> int:
            serialized = json.dumps(value, ensure_ascii=False)
            if tokenizer is not None:
                try:
                    encoded = tokenizer.encode(serialized, add_special_tokens=False)
                    return len(encoded)
                except Exception:  # pragma: no cover - tokenizer errors fallback
                    pass
            approx = max(1, len(serialized) // 4)
            return approx

        def trim_string(value: str, budget: int) -> str | None:
            if budget <= 0:
                return None
            if token_count(value) <= budget:
                return value
            low, high = 1, len(value)
            best: str | None = None
            while low <= high:
                mid = (low + high) // 2
                candidate = value[:mid]
                if token_count(candidate) <= budget:
                    best = candidate
                    low = mid + 1
                else:
                    high = mid - 1
            return best

        def trim_value(value: Any, budget: int) -> Any | None:
            if budget <= 0:
                return None
            if isinstance(value, str):
                return trim_string(value, budget)
            if isinstance(value, (int, float, bool)) or value is None:
                return value
            if isinstance(value, list):
                trimmed_list: list[Any] = []
                for item in value:
                    remaining = budget - token_count(trimmed_list)
                    if remaining <= 0:
                        break
                    trimmed_item = trim_value(item, remaining)
                    if trimmed_item is None:
                        break
                    candidate = trimmed_list + [trimmed_item]
                    if token_count(candidate) > budget:
                        break
                    trimmed_list.append(trimmed_item)
                return trimmed_list
            if isinstance(value, dict):
                trimmed_dict: dict[str, Any] = {}
                for key, val in value.items():
                    remaining = budget - token_count(trimmed_dict)
                    if remaining <= 0:
                        break
                    trimmed_val = trim_value(val, remaining)
                    if trimmed_val is None:
                        break
                    candidate = dict(trimmed_dict)
                    candidate[key] = trimmed_val
                    if token_count(candidate) > budget:
                        break
                    trimmed_dict[key] = trimmed_val
                return trimmed_dict
            as_string = str(value)
            return trim_string(as_string, budget)

        trimmed: dict[str, Any] = {}
        for key, value in context.items():
            remaining_budget = token_limit - token_count(trimmed)
            if remaining_budget <= 0:
                break
            trimmed_value = trim_value(value, remaining_budget)
            if trimmed_value is None:
                break
            candidate = dict(trimmed)
            candidate[key] = trimmed_value
            if token_count(candidate) > token_limit:
                break
            trimmed[key] = trimmed_value
        return trimmed

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response"""
        if not text:
            return "[]"
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped

        code_block = re.search(
            r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", stripped, re.IGNORECASE
        )
        if code_block:
            return code_block.group(1)

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
                    return stripped[start_index : index + 1]

        inline = re.search(r"\[(?:[^\]\[]|\[[^\]]*\])*\]", stripped)
        if inline:
            return inline.group(0)

        return "[]"

    def _heuristic_fallback(self, prompt: str, actions: List[str]) -> List[str]:
        """Fallback to simple heuristic when LLM fails"""
        prompt_lower = prompt.lower()
        return sorted(
            actions,
            key=lambda x: prompt_lower.count(x.lower()),
            reverse=True,
        )
