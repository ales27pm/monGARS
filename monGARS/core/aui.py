from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Sequence

# Prefer the existing embedding system if present
try:  # pragma: no cover - import guard
    from .neurones import EmbeddingSystem  # type: ignore[attr-defined]

    _HAS_NEURONES = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_NEURONES = False

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
        self._embed: EmbeddingSystem | None = None
        if _HAS_NEURONES:
            try:
                self._embed = EmbeddingSystem()
            except Exception as exc:  # pragma: no cover - instantiation failure
                logger.warning("aui_embed_init_failed", extra={"error": repr(exc)})
                self._embed = None

    @property
    def model_name(self) -> str:
        return "neurones" if self._embed else "keyword"

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

        if self._embed:
            try:
                prompt_vector, prompt_used_fallback = await self._embed.encode(prompt)
                action_vectors = await asyncio.gather(
                    *(self._embed.encode(description) for _, description in items)
                )
                if prompt_used_fallback or any(
                    action_used_fallback for _, action_used_fallback in action_vectors
                ):
                    return fallback_scores
                embedding_scores = {
                    key: _cosine(prompt_vector, vector)
                    for (key, _), (vector, _) in zip(
                        items, action_vectors, strict=False
                    )
                }
                return {
                    key: embedding_scores[key]
                    + (fallback_scores.get(key, 0.0) * _FALLBACK_KEYWORD_WEIGHT)
                    for key in embedding_scores
                }
            except Exception as exc:  # pragma: no cover - runtime fallback
                logger.warning(
                    "aui_embedding_inference_failed", extra={"error": repr(exc)}
                )

        return fallback_scores

    async def order(
        self,
        prompt: str,
        actions: Iterable[tuple[str, str]] = DEFAULT_ACTIONS,
    ) -> tuple[list[str], dict[str, float]]:
        scores = await self.suggest(prompt, actions)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [key for key, _ in ordered], scores
