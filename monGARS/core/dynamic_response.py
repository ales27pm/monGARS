from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from monGARS.core.personality import PersonalityEngine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CachedPersonality:
    """Container used to keep track of cached personality traits."""

    traits: dict[str, float]
    signature: str
    expires_at: float


class AdaptiveResponseGenerator:
    """Adaptive response generator with personality caching support."""

    def __init__(
        self,
        personality_engine: PersonalityEngine | None = None,
        *,
        cache_ttl_seconds: int = 300,
        time_provider: Callable[[], float] | None = None,
    ) -> None:
        self._personality_engine = personality_engine or PersonalityEngine()
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, _CachedPersonality] = {}
        self._lock = asyncio.Lock()
        self._time_provider = time_provider or time.monotonic

    async def get_personality_traits(
        self,
        user_id: str,
        interactions: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Return cached personality traits or refresh them if stale.

        A cached entry is reused when the interaction fingerprint has not changed
        and the configured TTL has not expired. This avoids redundant calls to the
        potentially expensive :class:`PersonalityEngine` analysis.
        """

        fingerprint = self._fingerprint_interactions(interactions)
        now = self._time_provider()
        async with self._lock:
            cached = self._cache.get(user_id)
            if cached and cached.signature == fingerprint:
                if self._cache_ttl < 0:
                    logger.debug(
                        "Using indefinitely cached personality traits for user %s",
                        user_id,
                    )
                    return dict(cached.traits)
                if self._cache_ttl > 0 and cached.expires_at > now:
                    logger.debug("Using cached personality traits for user %s", user_id)
                    return dict(cached.traits)

        traits = await self._personality_engine.analyze_personality(
            user_id,
            list(interactions or []),
        )
        normalized_traits = dict(traits)
        async with self._lock:
            expires_at = now + self._cache_ttl if self._cache_ttl > 0 else now
            self._cache[user_id] = _CachedPersonality(
                traits=dict(normalized_traits),
                signature=fingerprint,
                expires_at=expires_at,
            )
            logger.debug(
                "Cached personality traits for user %s (signature=%s, ttl=%s)",
                user_id,
                fingerprint,
                self._cache_ttl,
            )
        return dict(normalized_traits)

    def generate_adaptive_response(
        self, text: str, personality: Mapping[str, float] | None
    ) -> str:
        """Return an adapted response based on user personality."""

        personality = personality or {}
        formality = self._safe_float(personality.get("formality"), 0.5)
        humor = self._safe_float(personality.get("humor"), 0.5)
        enthusiasm = self._safe_float(personality.get("enthusiasm"), 0.5)

        adapted = text

        # Handle formality with more comprehensive pronoun replacement
        if formality > 0.7:
            adapted = re.sub(r"\btu\b", "vous", adapted, flags=re.IGNORECASE)
        elif formality < 0.3:
            adapted = re.sub(r"\bvous\b", "tu", adapted, flags=re.IGNORECASE)

        if enthusiasm > 0.7 and not adapted.endswith("!"):
            adapted += "!"

        smiling_emoji = "\U0001f603"
        if humor > 0.7 and smiling_emoji not in adapted:
            adapted += f" {smiling_emoji}"

        return adapted

    def _fingerprint_interactions(
        self, interactions: Sequence[Mapping[str, Any]] | None
    ) -> str:
        if not interactions:
            return "no-interactions"
        normalized: list[dict[str, Any]] = []
        for item in interactions:
            normalized.append({key: item.get(key) for key in sorted(item)})
        payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
