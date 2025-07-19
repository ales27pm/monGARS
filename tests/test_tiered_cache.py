import asyncio
import os

import pytest

os.environ.setdefault("DEBUG", "True")
os.environ.setdefault("SECRET_KEY", "test-secret")

from monGARS.core.caching.tiered_cache import TieredCache


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
