from __future__ import annotations

import asyncio
from time import monotonic
from typing import Callable

from fastapi import HTTPException, status


class InMemoryRateLimiter:
    """Simple in-memory rate limiter with pruning for stale entries."""

    def __init__(
        self,
        *,
        interval_seconds: float,
        prune_after_seconds: float,
        on_reject: Callable[[str], None] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if prune_after_seconds <= 0:
            raise ValueError("prune_after_seconds must be positive")
        self._interval = interval_seconds
        self._prune_after = max(prune_after_seconds, interval_seconds)
        self._on_reject = on_reject
        self._lock = asyncio.Lock()
        self._last_seen: dict[str, float] = {}

    async def ensure_permitted(self, user_id: str) -> None:
        """Raise an HTTP 429 if the user sends requests too quickly."""

        now = monotonic()
        async with self._lock:
            self._prune(now)
            last = self._last_seen.get(user_id)
            if last is not None and now - last < self._interval:
                if self._on_reject is not None:
                    self._on_reject(user_id)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests: please wait before sending another message.",
                )
            self._last_seen[user_id] = now

    async def reset(self) -> None:
        """Clear limiter state in a concurrency-safe manner."""

        async with self._lock:
            self._last_seen.clear()

    def _prune(self, now: float) -> None:
        """Remove stale entries to prevent unbounded memory usage."""

        cutoff = now - self._prune_after
        if cutoff <= 0 or not self._last_seen:
            return
        stale_users = [
            user for user, timestamp in self._last_seen.items() if timestamp < cutoff
        ]
        if not stale_users:
            return
        for user in stale_users:
            self._last_seen.pop(user, None)
