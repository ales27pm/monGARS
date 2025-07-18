import pytest

from monGARS.core.hippocampus import Hippocampus


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
