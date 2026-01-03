"""Event model and bus for UI streaming."""

from __future__ import annotations

import asyncio
import contextlib
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
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - redis is optional in many deployments
    aioredis = None
    RedisError = Exception  # type: ignore[misc,assignment]


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


class BackendUnavailable(RuntimeError):
    """Raised when the configured backend cannot service requests."""


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
        try:
            await self._redis.publish(self._channel, ev.to_json())
        except asyncio.CancelledError:  # pragma: no cover - cancellation passthrough
            raise
        except (RedisError, OSError) as exc:
            raise BackendUnavailable("redis publish failed") from exc

    def subscribe(self) -> AsyncIterator[Event]:
        async def iterator() -> AsyncIterator[Event]:
            pubsub = self._redis.pubsub()
            try:
                await pubsub.subscribe(self._channel)
            except (
                asyncio.CancelledError
            ):  # pragma: no cover - cancellation passthrough
                raise
            except (RedisError, OSError) as exc:
                await pubsub.close()
                raise BackendUnavailable("redis subscribe failed") from exc

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
            except (
                asyncio.CancelledError
            ):  # pragma: no cover - cancellation passthrough
                raise
            except (RedisError, OSError) as exc:
                raise BackendUnavailable("redis listen failed") from exc
            finally:
                with contextlib.suppress(Exception):
                    await pubsub.unsubscribe(self._channel)
                    await pubsub.close()

        return iterator()


class EventBus:
    """Pluggable pub/sub. Starts in-memory, auto-upgrades to Redis if configured."""

    def __init__(self) -> None:
        maxsize = getattr(settings, "EVENTBUS_MEMORY_QUEUE_MAXSIZE", 1000)
        self._memory_backend = MemoryEventBackend(max_queue_size=maxsize)
        self._backend: EventBackend = self._select_backend()

    def _select_backend(self) -> EventBackend:
        if settings.EVENTBUS_USE_REDIS and settings.REDIS_URL and aioredis:
            try:
                return RedisEventBackend(
                    redis_url=str(settings.REDIS_URL),
                    channel="mongars:events",
                )
            except RuntimeError as exc:  # pragma: no cover - defensive guard
                log.warning(
                    "event_bus.redis_initialisation_failed",
                    extra={"reason": str(exc)},
                )
        return self._memory_backend

    def _fallback_to_memory(self, exc: Exception | None = None) -> None:
        if isinstance(self._backend, MemoryEventBackend):
            return
        reason = str(exc) if exc else "unavailable"
        log.warning(
            "event_bus.falling_back_to_memory",
            extra={"reason": reason},
        )
        self._backend = self._memory_backend

    def _wrap_iterator(self, iterator: AsyncIterator[Event]) -> AsyncIterator[Event]:
        async def generator() -> AsyncIterator[Event]:
            nonlocal iterator
            while True:
                try:
                    yield await iterator.__anext__()
                except BackendUnavailable as exc:
                    self._fallback_to_memory(exc)
                    with contextlib.suppress(Exception):
                        await iterator.aclose()  # type: ignore[attr-defined]
                    iterator = self._backend.subscribe()
                except asyncio.CancelledError:
                    raise
                except StopAsyncIteration:
                    return

        return generator()

    async def publish(self, ev: Event) -> None:
        """Publish an event to subscribers."""

        try:
            await self._backend.publish(ev)
        except BackendUnavailable as exc:
            self._fallback_to_memory(exc)
            await self._backend.publish(ev)

    def subscribe(self) -> AsyncIterator[Event]:
        """Yield events as they are received."""

        try:
            iterator = self._backend.subscribe()
        except BackendUnavailable as exc:
            self._fallback_to_memory(exc)
            iterator = self._backend.subscribe()
        return self._wrap_iterator(iterator)


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
