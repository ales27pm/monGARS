from datetime import timedelta

import pytest

import monGARS.core.hippocampus as hippocampus_module
from monGARS.core.hippocampus import Hippocampus
from monGARS.core.persistence import PersistenceRepository
from monGARS.init_db import reset_database


async def populate(h: Hippocampus, user: str, count: int):
    for i in range(count):
        await h.store(user, f"q{i}", f"r{i}")


@pytest.mark.asyncio
async def test_store_and_history_order():
    h = Hippocampus()
    await populate(h, "u1", 3)
    history = await h.history("u1")
    assert [item.query for item in history] == ["q2", "q1", "q0"]


@pytest.mark.asyncio
async def test_multiple_users_isolated():
    h = Hippocampus()
    await populate(h, "u1", 1)
    await populate(h, "u2", 2)
    hist1 = await h.history("u1")
    hist2 = await h.history("u2")
    assert len(hist1) == 1
    assert len(hist2) == 2
    assert hist2[0].query == "q1"


@pytest.mark.asyncio
async def test_history_limit():
    h = Hippocampus()
    await populate(h, "u1", h.MAX_HISTORY + 5)
    history = await h.history("u1", limit=h.MAX_HISTORY)
    # Should return the most recent entries up to MAX_HISTORY
    assert len(history) == h.MAX_HISTORY
    assert history[0].query == f"q{h.MAX_HISTORY + 4}"


@pytest.mark.asyncio
async def test_empty_history():
    h = Hippocampus()
    history = await h.history("nouser")
    assert history == []


@pytest.mark.asyncio
async def test_persistent_history_across_instances():
    await reset_database()
    repo = PersistenceRepository()
    h1 = Hippocampus(persistence=repo, persist_on_store=True)
    await h1.store("u-persist", "q0", "r0")

    # New instance should hydrate from the shared persistence layer.
    h2 = Hippocampus(persistence=repo)
    history = await h2.history("u-persist", limit=1)
    assert history
    assert history[0].query == "q0"
    assert history[0].response == "r0"


@pytest.mark.asyncio
async def test_flush_now_removes_expired_entries(monkeypatch):
    await reset_database()
    hippocampus = Hippocampus()
    await hippocampus.store("u1", "fresh", "stay", ttl=timedelta(hours=1))
    expiring = await hippocampus.store("u1", "old", "gone", ttl=timedelta(seconds=30))

    future = expiring.expires_at + timedelta(seconds=5)
    original_now = hippocampus_module._utcnow
    monkeypatch.setattr(hippocampus_module, "_utcnow", lambda: future)
    removed = await hippocampus.flush_now()
    monkeypatch.setattr(hippocampus_module, "_utcnow", original_now)

    assert removed >= 1
    history = await hippocampus.history("u1", limit=5)
    queries = [item.query for item in history]
    assert "old" not in queries
    assert "fresh" in queries
