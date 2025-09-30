"""Event model and bus for UI streaming."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any, Optional

from monGARS.config import get_settings

try:  # pragma: no cover - optional dependency
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover - redis is optional in many deployments
    aioredis = None


settings = get_settings()


@dataclass(frozen=True, slots=True)
class Event:
    """Typed envelope pushed to the UI."""

    id: str
    type: str
    ts: float
    user: str | None
    data: dict[str, Any]

    def to_json(self) -> str:
        """Serialise the event payload into a compact JSON string."""

        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)


class EventBus:
    """Pluggable pub/sub. Starts in-memory, auto-upgrades to Redis if configured."""

    def __init__(self) -> None:
        self._memory_queue: "asyncio.Queue[Event]" = asyncio.Queue()
        self._redis = None
        self._channel = "mongars:events"
        if settings.REDIS_URL and aioredis:
            self._redis = aioredis.from_url(
                str(settings.REDIS_URL), encoding="utf-8", decode_responses=True
            )

    async def publish(self, ev: Event) -> None:
        """Publish an event to subscribers."""

        if self._redis:
            await self._redis.publish(self._channel, ev.to_json())
            return
        await self._memory_queue.put(ev)

    async def subscribe(self) -> AsyncIterator[Event]:
        """Yield events as they are received."""

        if self._redis:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(self._channel)
            try:
                async for message in pubsub.listen():
                    if message.get("type") != "message":
                        continue
                    data = message.get("data")
                    if not data:
                        continue
                    payload = json.loads(data)
                    yield Event(**payload)
            finally:
                await pubsub.unsubscribe(self._channel)
                await pubsub.close()
        else:
            while True:
                event = await self._memory_queue.get()
                yield event


_event_bus: Optional[EventBus] = None


def event_bus() -> EventBus:
    """Return the process-wide event bus instance."""

    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def make_event(ev_type: str, user: str | None, data: dict[str, Any]) -> Event:
    """Create a standardised event payload."""

    return Event(
        id=str(uuid.uuid4()), type=ev_type, ts=time.time(), user=user, data=data
    )
