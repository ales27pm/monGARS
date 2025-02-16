import asyncio
import redis.asyncio as redis
import aiocache
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer
from monGARS.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

aiocache.settings.set_defaults(
    {
        "default": {
            "cache": "aiocache.SimpleMemoryCache",
            "serializer": {"class": "aiocache.serializers.PickleSerializer"},
            "size": settings.in_memory_cache_size
        },
        "redis_cache": {
            "cache": "aiocache.RedisCache",
            "host": settings.redis_url.host,
            "port": settings.redis_url.port,
            "db": int(settings.redis_url.path.strip('/')),
            "serializer": {"class": "aiocache.serializers.PickleSerializer"},
            "timeout": 1
         },
        "disk_cache": {
            "cache": "aiocache.FileCache",
            "serializer": {"class": "aiocache.serializers.PickleSerializer"},
            "dir": settings.disk_cache_path
        }
    }
)

async def clear_cache(cache_alias="default"):
    cache = aiocache.caches.get(cache_alias)
    if cache:
        await cache.clear()
        logger.info(f"Cache '{cache_alias}' cleared.")

@cached(
    ttl=60, cache=Cache.REDIS, key_builder=lambda f, *args, **kw: f"{f.__name__}:{args}:{kw}", alias="redis_cache"
)
@cached(
    ttl=30, cache=Cache.MEMORY, key_builder=lambda f, *args, **kw: f"{f.__name__}:{args}:{kw}"
)
async def get_cached_data(key: str):
    logger.info(f"Fetching data for key: {key} (not from cache)")
    await asyncio.sleep(1)
    return f"Data for {key}"

@cached(ttl=3600, cache=Cache.FILE, key_builder=lambda f, *args, **kw: f"{f.__name__}:{args}:{kw}", alias="disk_cache")
async def get_data_from_disk_cache(param1: str, param2: int):
    logger.info(f"Loading data from source with params: {param1}, {param2}")
    await asyncio.sleep(5)
    return {"param1": param1, "param2": param2, "source": "disk"}