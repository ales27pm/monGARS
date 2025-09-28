from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from cachetools import TTLCache

from monGARS.core.personality import PersonalityEngine

logger = logging.getLogger(__name__)

_CACHE_MAXSIZE = 1024


def _fingerprint_interactions(
    interactions: Sequence[Mapping[str, Any]] | None,
) -> str:
    if not interactions:
        return "no-interactions"
    normalized = [{key: item.get(key) for key in sorted(item)} for item in interactions]
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        self._time_provider = time_provider or time.monotonic
        self._lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Task[dict[str, float]]] = {}
        self._permanent_cache: dict[str, dict[str, float]] | None = (
            {} if cache_ttl_seconds < 0 else None
        )
        self._ttl_cache: TTLCache[str, dict[str, float]] | None = (
            TTLCache(
                maxsize=_CACHE_MAXSIZE,
                ttl=cache_ttl_seconds,
                timer=self._time_provider,
            )
            if cache_ttl_seconds > 0
            else None
        )

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

        interactions_list = list(interactions or [])
        fingerprint = _fingerprint_interactions(interactions_list)
        cache_key = self._cache_key(user_id, fingerprint)

        async with self._lock:
            cached = self._get_cached_locked(cache_key)
            if cached is not None:
                logger.debug("Using cached personality traits for user %s", user_id)
                return dict(cached)

            pending = self._pending.get(cache_key)
            if pending is None:
                pending = asyncio.create_task(
                    self._personality_engine.analyze_personality(
                        user_id, interactions_list
                    )
                )
                self._pending[cache_key] = pending
            else:
                logger.debug(
                    "Awaiting in-flight personality analysis for user %s", user_id
                )

        try:
            traits = await pending
        finally:
            async with self._lock:
                if self._pending.get(cache_key) is pending:
                    self._pending.pop(cache_key, None)

        normalized_traits = dict(traits)

        async with self._lock:
            self._store_cached_locked(cache_key, normalized_traits)
            logger.debug(
                "Cached personality traits for user %s (ttl=%s)",
                user_id,
                "infinite" if self._cache_ttl < 0 else self._cache_ttl,
            )

        return dict(normalized_traits)

    def generate_adaptive_response(
        self, text: str, personality: Mapping[str, float] | None
    ) -> str:
        """Return an adapted response based on user personality."""

        personality = personality or {}

        def _coerce(value: Any, default: float) -> float:
            if value is None:
                return default
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        formality = _coerce(personality.get("formality"), 0.5)
        humor = _coerce(personality.get("humor"), 0.5)
        enthusiasm = _coerce(personality.get("enthusiasm"), 0.5)

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

    def _cache_key(self, user_id: str, fingerprint: str) -> str:
        if self._cache_ttl < 0:
            return user_id
        if self._cache_ttl == 0:
            return f"{user_id}:no-cache"
        return f"{user_id}:{fingerprint}"

    def _get_cached_locked(self, cache_key: str) -> dict[str, float] | None:
        if self._cache_ttl == 0:
            return None
        if self._cache_ttl < 0 and self._permanent_cache is not None:
            return self._permanent_cache.get(cache_key)
        if self._ttl_cache is None:
            return None
        return self._ttl_cache.get(cache_key)

    def _store_cached_locked(self, cache_key: str, traits: dict[str, float]) -> None:
        if self._cache_ttl == 0:
            return
        if self._cache_ttl < 0 and self._permanent_cache is not None:
            self._permanent_cache[cache_key] = dict(traits)
            return
        if self._ttl_cache is not None:
            self._ttl_cache[cache_key] = dict(traits)
