from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from cachetools import TTLCache

from monGARS.core.personality import PersonalityEngine
from monGARS.core.style_finetuning import StyleFineTuner

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
        style_tuner: StyleFineTuner | None = None,
    ) -> None:
        shared_tuner = style_tuner
        if personality_engine is None:
            if shared_tuner is None:
                shared_tuner = StyleFineTuner()
            if isinstance(shared_tuner, StyleFineTuner):
                personality_engine = PersonalityEngine(style_tuner=shared_tuner)
            else:
                personality_engine = PersonalityEngine()
        else:
            if shared_tuner is None and hasattr(personality_engine, "style_tuner"):
                shared_tuner = personality_engine.style_tuner  # type: ignore[attr-defined]
            elif isinstance(shared_tuner, StyleFineTuner) and hasattr(
                personality_engine, "set_style_tuner"
            ):
                personality_engine.set_style_tuner(shared_tuner)  # type: ignore[attr-defined]
            elif shared_tuner is None:
                shared_tuner = StyleFineTuner()
                if hasattr(personality_engine, "set_style_tuner"):
                    personality_engine.set_style_tuner(shared_tuner)  # type: ignore[attr-defined]

        self._personality_engine = personality_engine
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
        self._style_tuner = shared_tuner or StyleFineTuner()
        self._recent_histories: dict[str, list[Mapping[str, Any]]] = {}

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
            self._recent_histories[user_id] = interactions_list

        return dict(normalized_traits)

    async def estimate_personality(
        self,
        user_id: str,
        interactions: Sequence[Mapping[str, Any]] | None = None,
        *,
        force_refresh: bool = False,
    ) -> dict[str, float]:
        """Return the latest personality traits for ``user_id``.

        Parameters
        ----------
        user_id:
            Identifier of the user whose personality needs to be estimated.
        interactions:
            Optional interaction history to analyse. When omitted, the most
            recent history observed for ``user_id`` is reused so callers can
            request another estimate without rehydrating chat transcripts.
        force_refresh:
            When ``True`` the cached entry (if any) is invalidated prior to
            running a fresh analysis. This allows callers to bypass the
            time-based cache when they know that the interaction history has
            materially changed.

        Returns
        -------
        dict[str, float]
            The normalised trait mapping provided by
            :class:`PersonalityEngine`.
        """

        interactions_list: list[Mapping[str, Any]]
        if interactions is None:
            interactions_list = list(self._recent_histories.get(user_id, []))
        else:
            interactions_list = list(interactions)

        fingerprint = _fingerprint_interactions(interactions_list)
        cache_key = self._cache_key(user_id, fingerprint)

        if force_refresh:
            async with self._lock:
                self._pending.pop(cache_key, None)
                if self._cache_ttl < 0 and self._permanent_cache is not None:
                    self._permanent_cache.pop(cache_key, None)
                if self._ttl_cache is not None:
                    self._ttl_cache.pop(cache_key, None)

        try:
            traits = await self.get_personality_traits(user_id, interactions_list)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to estimate personality for %s: %s", user_id, exc
            )
            return {}

        return traits

    def generate_adaptive_response(
        self,
        text: str,
        personality: Mapping[str, float] | None,
        *,
        user_id: str,
    ) -> str:
        """Return an adapted response based on user personality."""

        history = self._recent_histories.get(user_id, [])
        if not history:
            logger.debug(
                "No cached history for %s; attempting personality estimation cache reuse.",
                user_id,
            )

        try:
            adapted = self._style_tuner.apply_style(
                user_id,
                text,
                dict(personality or {}),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception(
                "Failed to apply style adaptation for %s: %s", user_id, exc
            )
            return text

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
