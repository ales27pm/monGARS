import asyncio
from collections.abc import AsyncIterator

import pytest

from monGARS.core.ui_events import Event, EventBus, event_bus, make_event


@pytest.mark.asyncio
async def test_in_memory_bus_publish_and_receive() -> None:
    bus = EventBus()
    ev = make_event("ai_model.response_chunk", "user-1", {"text": "hello"})

    subscriber: AsyncIterator[Event] = bus.subscribe()
    await bus.publish(ev)

    received = await asyncio.wait_for(subscriber.__anext__(), timeout=1)
    await subscriber.aclose()

    assert received == ev


@pytest.mark.asyncio
async def test_in_memory_bus_broadcasts_to_all_subscribers() -> None:
    bus = EventBus()
    ev = make_event("ai_model.response_chunk", "user-2", {"text": "bonjour"})

    first = bus.subscribe()
    second = bus.subscribe()

    await bus.publish(ev)

    first_result = await asyncio.wait_for(first.__anext__(), timeout=1)
    second_result = await asyncio.wait_for(second.__anext__(), timeout=1)

    await first.aclose()
    await second.aclose()

    assert first_result == ev
    assert second_result == ev


@pytest.mark.asyncio
async def test_in_memory_bus_applies_backpressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from monGARS.core import ui_events

    custom_settings = ui_events.settings.model_copy(
        update={"EVENTBUS_MEMORY_QUEUE_MAXSIZE": 1}
    )
    monkeypatch.setattr(ui_events, "settings", custom_settings)

    bus = EventBus()
    subscriber = bus.subscribe()

    first = make_event("demo.first", None, {})
    second = make_event("demo.second", None, {})

    await bus.publish(first)

    publish_task = asyncio.create_task(bus.publish(second))
    await asyncio.sleep(0)

    assert not publish_task.done()

    await asyncio.wait_for(subscriber.__anext__(), timeout=1)
    await asyncio.sleep(0)

    assert publish_task.done()
    await publish_task

    await subscriber.aclose()


def test_make_event_populates_fields() -> None:
    ev = make_event("system.notice", None, {"message": "ready"})

    assert ev.type == "system.notice"
    assert ev.user is None
    assert ev.data == {"message": "ready"}
    uuid_parts = ev.id.split("-")
    assert len(uuid_parts) == 5
    assert ev.ts > 0


@pytest.mark.asyncio
async def test_event_bus_singleton_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    from monGARS.core import ui_events

    monkeypatch.setattr(ui_events, "_event_bus", None)

    first = event_bus()
    second = event_bus()

    assert first is second


@pytest.mark.asyncio
async def test_event_bus_falls_back_when_redis_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from monGARS.core import ui_events

    class DummyRedisBackend(ui_events.EventBackend):
        def __init__(self) -> None:
            self.publish_attempts = 0

        async def publish(self, ev: ui_events.Event) -> None:  # noqa: ARG002
            self.publish_attempts += 1
            raise ui_events.BackendUnavailable("redis down")

        def subscribe(self) -> AsyncIterator[ui_events.Event]:
            raise ui_events.BackendUnavailable("redis down")

    dummy_backend = DummyRedisBackend()

    monkeypatch.setattr(ui_events, "aioredis", object())
    monkeypatch.setattr(
        ui_events,
        "settings",
        ui_events.settings.model_copy(update={"REDIS_URL": "redis://redis:6379/0"}),
    )
    monkeypatch.setattr(ui_events, "_event_bus", None)
    monkeypatch.setattr(
        ui_events,
        "RedisEventBackend",
        lambda **_: dummy_backend,
    )

    bus = ui_events.EventBus()

    initial = make_event("demo.initial", "user", {})
    await bus.publish(initial)
    assert isinstance(bus._backend, ui_events.MemoryEventBackend)

    subscriber = bus.subscribe()
    follow_up = make_event("demo.follow_up", "user", {})
    await bus.publish(follow_up)

    received = await asyncio.wait_for(subscriber.__anext__(), timeout=1)
    await subscriber.aclose()

    assert received == follow_up
