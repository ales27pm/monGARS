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
