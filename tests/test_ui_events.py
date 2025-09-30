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

    assert received == ev


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
