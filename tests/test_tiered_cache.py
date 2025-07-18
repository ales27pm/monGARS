import pytest

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
