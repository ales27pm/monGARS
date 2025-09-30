"""Event model and bus for UI streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any, Optional

from monGARS.config import get_settings

try:  # pragma: no cover - optional dependency
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover - redis is optional in many deployments
    aioredis = None


settings = get_settings()
log = logging.getLogger(__name__)


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


class EventBackend(ABC):
    """Abstract backend for publishing and subscribing to events."""

    @abstractmethod
    async def publish(self, ev: Event) -> None:
        """Publish an event to interested subscribers."""

    @abstractmethod
    def subscribe(self) -> AsyncIterator[Event]:
        """Return an async iterator yielding new events."""


class MemoryEventBackend(EventBackend):
    """Broadcast events to per-subscriber queues backed by asyncio."""

    def __init__(self, *, max_queue_size: int) -> None:
        self._max_queue_size = max_queue_size
        self._subscribers: set[asyncio.Queue[Event]] = set()

    async def publish(self, ev: Event) -> None:
        for queue in tuple(self._subscribers):
            await queue.put(ev)

    def subscribe(self) -> AsyncIterator[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers.add(queue)

        async def iterator() -> AsyncIterator[Event]:
            try:
                while True:
                    yield await queue.get()
            finally:
                self._subscribers.discard(queue)

        return iterator()


class RedisEventBackend(EventBackend):
    """Publish events via Redis pub/sub."""

    def __init__(self, *, redis_url: str, channel: str) -> None:
        if not aioredis:  # pragma: no cover - guard for optional dependency
            raise RuntimeError("redis backend requested without redis client")
        self._redis = aioredis.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )
        self._channel = channel

    async def publish(self, ev: Event) -> None:
        await self._redis.publish(self._channel, ev.to_json())

    def subscribe(self) -> AsyncIterator[Event]:
        async def iterator() -> AsyncIterator[Event]:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(self._channel)
            try:
                async for message in pubsub.listen():
                    if message.get("type") != "message":
                        continue
                    data = message.get("data")
                    if not data:
                        continue
                    if not isinstance(data, str):
                        log.warning(
                            "redis_event_bus.invalid_payload_type",
                            extra={"type": type(data).__name__},
                        )
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        log.warning("redis_event_bus.invalid_json")
                        continue
                    if not isinstance(payload, dict):
                        log.warning(
                            "redis_event_bus.non_mapping_payload",
                            extra={"payload_type": type(payload).__name__},
                        )
                        continue
                    try:
                        yield Event(**payload)
                    except TypeError:
                        log.warning(
                            "redis_event_bus.invalid_event_payload",
                            extra={"payload_keys": sorted(payload.keys())},
                        )
                        continue
            finally:
                await pubsub.unsubscribe(self._channel)
                await pubsub.close()

        return iterator()


class EventBus:
    """Pluggable pub/sub. Starts in-memory, auto-upgrades to Redis if configured."""

    def __init__(self) -> None:
        if settings.REDIS_URL and aioredis:
            self._backend: EventBackend = RedisEventBackend(
                redis_url=str(settings.REDIS_URL),
                channel="mongars:events",
            )
        else:
            maxsize = getattr(settings, "EVENTBUS_MEMORY_QUEUE_MAXSIZE", 1000)
            self._backend = MemoryEventBackend(max_queue_size=maxsize)

    async def publish(self, ev: Event) -> None:
        """Publish an event to subscribers."""

        await self._backend.publish(ev)

    def subscribe(self) -> AsyncIterator[Event]:
        """Yield events as they are received."""

        return self._backend.subscribe()


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
