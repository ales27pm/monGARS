import asyncio

import pytest

from monGARS.core.caching.tiered_cache import SimpleDiskCache, TieredCache


@pytest.mark.asyncio
async def test_tiered_cache_set_and_get(tmp_path):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()

    await cache.set("k", "v")
    assert await cache.get("k") == "v"

    # force fallback by clearing fast layers
    await cache.memory.clear()
    try:
        await cache.redis.clear()
    except Exception:
        pass
    assert await cache.get("k") == "v"


@pytest.mark.asyncio
async def test_tiered_cache_ttl(tmp_path):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()

    await cache.set("expire", "v", ttl=1)
    await asyncio.sleep(1.5)
    assert await cache.get("expire") is None


@pytest.mark.asyncio
async def test_simple_disk_cache_negative_ttl(tmp_path):
    cache = SimpleDiskCache(str(tmp_path))

    await cache.set("persist", "value", ttl=-5)
    await asyncio.sleep(0.1)

    assert await cache.get("persist") == "value"


@pytest.mark.asyncio
async def test_tiered_cache_special_and_none(tmp_path):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()

    key = 'spécial/\\:*?"<>|'
    await cache.set(key, None)
    assert await cache.get(key) is None


@pytest.mark.asyncio
async def test_tiered_cache_concurrent(tmp_path):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()

    async def worker(i: int):
        await cache.set(f"k{i}", i)
        return await cache.get(f"k{i}")

    results = await asyncio.gather(*(worker(i) for i in range(10)))
    assert results == list(range(10))


@pytest.mark.asyncio
async def test_tiered_cache_set_failure(tmp_path, monkeypatch):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()

    async def broken_set(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(cache.redis, "set", broken_set)
    await cache.set("fail", "ok")
    assert await cache.get("fail") == "ok"


@pytest.mark.asyncio
async def test_tiered_cache_metrics(tmp_path):
    cache = TieredCache(directory=str(tmp_path))
    await cache.clear_all()

    await cache.set("a", 1)
    await cache.get("a")
    await cache.get("missing")

    metrics = cache.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
