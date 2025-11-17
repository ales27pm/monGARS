from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Sequence

from .llm_integration import LLMIntegration

logger = logging.getLogger(__name__)

# Default actions (keys MUST match the front-end data-action)
DEFAULT_ACTIONS: list[tuple[str, str]] = [
    ("code", "Write, refactor or generate source code and tests."),
    ("summarize", "Summarize long passages, chats, or documents succinctly."),
    ("explain", "Explain a concept in simpler terms with examples."),
]

_FALLBACK_KEYWORD_WEIGHT = 0.2


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _keyword_score(action_key: str, prompt: str, action_desc: str) -> float:
    p = prompt.lower()
    d = action_desc.lower()
    hints: dict[str, tuple[str, ...]] = {
        "code": (
            "code",
            "function",
            "bug",
            "refactor",
            "compile",
            "typescript",
            "python",
            "class",
        ),
        "summarize": ("tl;dr", "summary", "summarize", "condense", "short version"),
        "explain": ("explain", "why", "how", "teach", "beginner", "simple"),
    }
    score = 0.0
    for word in hints.get(action_key, ()):
        if word in p:
            score += 1.0
        if word in d:
            score += 0.25
    return score


class AUISuggester:
    """Produces ordered suggestions for action-oriented UI shortcuts."""

    def __init__(self) -> None:
        self._llm: LLMIntegration | None = None
        try:
            self._llm = LLMIntegration.instance()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("aui_llm_init_failed", extra={"error": repr(exc)})
            self._llm = None

    @property
    def model_name(self) -> str:
        return "llm" if self._llm else "keyword"

    async def suggest(
        self,
        prompt: str,
        actions: Iterable[tuple[str, str]] = DEFAULT_ACTIONS,
    ) -> dict[str, float]:
        """
        Return a mapping of action key to relevance score for the provided prompt.

        Falls back to the keyword heuristic when embeddings are unavailable.
        """

        items = list(actions)
        if not items:
            return {}

        fallback_scores = {
            key: _keyword_score(key, prompt, description) for key, description in items
        }

        if self._llm:
            payloads = [prompt, *(description for _, description in items)]
            try:
                vectors = await asyncio.to_thread(
                    self._llm.embed_batch, payloads
                )
            except Exception as exc:  # pragma: no cover - runtime fallback
                logger.warning(
                    "aui_embedding_inference_failed", extra={"error": repr(exc)}
                )
            else:
                if len(vectors) != len(payloads):
                    logger.debug("aui_embedding_vector_mismatch")
                    return fallback_scores
                prompt_vector = vectors[0]
                action_vectors = vectors[1:]
                embedding_scores = {
                    key: _cosine(prompt_vector, vector)
                    for (key, _), vector in zip(items, action_vectors, strict=False)
                }
                return {
                    key: embedding_scores[key]
                    + (fallback_scores.get(key, 0.0) * _FALLBACK_KEYWORD_WEIGHT)
                    for key in embedding_scores
                }

        return fallback_scores

    async def order(
        self,
        prompt: str,
        actions: Iterable[tuple[str, str]] = DEFAULT_ACTIONS,
    ) -> tuple[list[str], dict[str, float]]:
        scores = await self.suggest(prompt, actions)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [key for key, _ in ordered], scores
