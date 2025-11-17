from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from .llm_integration import LLMIntegration

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

        if not self.llm:
            return self._heuristic_fallback(prompt, actions)

        # Trim context to fit within token limits
        trimmed_context = self._trim_context(context)

        suggestion_prompt = (
            "You are Dolphin, an AI assistant that helps users by suggesting relevant actions.\n"
            "Given the user's current intent and available actions, rank them from most to least relevant.\n\n"
            f"USER INTENT: '{prompt}'\n"
            f"AVAILABLE ACTIONS: {json.dumps(actions)}\n"
            f"CURRENT CONTEXT: {json.dumps(trimmed_context)}\n\n"
            "Respond ONLY with a JSON array of action names in order of relevance:"
        )

        try:
            response = self.llm.generate(suggestion_prompt, max_new_tokens=150)
        except Exception as exc:  # pragma: no cover - runtime fallback
            logger.warning("Suggestion generation failed: %s", str(exc))
            return self._heuristic_fallback(prompt, actions)

        try:
            ranked_actions = json.loads(self._extract_json(response))
            if not isinstance(ranked_actions, list):
                raise ValueError("LLM response must be a list")
            logger.info(
                {
                    "event": "suggestion_generated",
                    "model": "dolphin3-reasoning-baseline",
                    "user_intent": prompt[:50],
                    "suggested_count": len(ranked_actions),
                    "context_size": len(str(trimmed_context)),
                }
            )
            filtered_actions = [
                action
                for action in ranked_actions
                if isinstance(action, str) and action in actions
            ]
            return filtered_actions or self._heuristic_fallback(prompt, actions)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Suggestion parsing failed: {str(e)}")
            return self._heuristic_fallback(prompt, actions)

    def _trim_context(self, context: dict) -> dict:
        """Reduce context size to fit within token limits"""
        if not context or not isinstance(context, dict):
            return {}

        limit = max(self.context_window, 256)
        serialized = json.dumps(context, ensure_ascii=False)
        if len(serialized) <= limit:
            return context

        def trim_value(value: Any, budget: int) -> Any | None:
            if budget <= 0:
                return None
            if isinstance(value, str):
                return value[:budget]
            if isinstance(value, (int, float, bool)) or value is None:
                return value
            if isinstance(value, list):
                trimmed_list = []
                for item in value:
                    remaining_budget = budget - len(
                        json.dumps(trimmed_list, ensure_ascii=False)
                    )
                    if remaining_budget <= 0:
                        break
                    trimmed_item = trim_value(item, remaining_budget)
                    if trimmed_item is None:
                        break
                    trimmed_list.append(trimmed_item)
                return trimmed_list
            if isinstance(value, dict):
                trimmed_dict: dict[str, Any] = {}
                for key, val in value.items():
                    remaining_budget = budget - len(
                        json.dumps(trimmed_dict, ensure_ascii=False)
                    )
                    if remaining_budget <= 0:
                        break
                    trimmed_val = trim_value(val, remaining_budget)
                    if trimmed_val is None:
                        break
                    trimmed_dict[key] = trimmed_val
                return trimmed_dict
            return str(value)[:budget]

        trimmed: dict[str, Any] = {}
        for key, value in context.items():
            current_length = len(json.dumps(trimmed, ensure_ascii=False))
            if current_length >= limit:
                break
            trimmed_value = trim_value(value, limit - current_length)
            if trimmed_value is None:
                break
            trimmed[key] = trimmed_value
            if len(json.dumps(trimmed, ensure_ascii=False)) > limit:
                trimmed.pop(key, None)
                break

        return trimmed

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response"""
        match = re.search(r"\[.*\]", text, re.DOTALL)
        return match.group(0) if match else "[]"

    def _heuristic_fallback(self, prompt: str, actions: List[str]) -> List[str]:
        """Fallback to simple heuristic when LLM fails"""
        prompt_lower = prompt.lower()
        return sorted(
            actions,
            key=lambda x: prompt_lower.count(x.lower()),
            reverse=True,
        )
