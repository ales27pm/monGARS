import logging
import pickle
from pathlib import Path
from typing import Any

from aiocache import Cache, caches
from aiocache.serializers import PickleSerializer

from monGARS.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

caches.set_config(
    {
        "default": {
            "cache": "aiocache.SimpleMemoryCache",
            "serializer": {"class": "aiocache.serializers.PickleSerializer"},
        },
        "redis": {
            "cache": "aiocache.RedisCache",
            "endpoint": settings.redis_url.host,
            "port": settings.redis_url.port,
            "db": int(settings.redis_url.path.strip("/")),
            "serializer": {"class": "aiocache.serializers.PickleSerializer"},
            "timeout": 1,
        },
    }
)


class SimpleDiskCache:
    """Very small file-based cache used when aiocache FileCache is unavailable."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    async def get(self, key: str) -> Any:
        path = self.directory / f"{key}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                return pickle.load(fh)
        return None

    async def set(
        self, key: str, value: Any, ttl: int | None = None
    ) -> None:  # noqa: ARG002
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f"{key}.pkl").open("wb") as fh:
            pickle.dump(value, fh)

    async def clear(self) -> None:
        for file in self.directory.glob("*.pkl"):
            file.unlink(missing_ok=True)


class TieredCache:
    """Memory, Redis and disk-backed cache with graceful fallbacks."""

    def __init__(self, directory: str | None = None) -> None:
        self.memory = caches.get("default")
        self.redis = caches.get("redis")
        self.disk = SimpleDiskCache(directory or settings.DISK_CACHE_PATH)

    async def _safe(self, cache: Cache, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            func = getattr(cache, method)
            return await func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on external services
            logger.debug("Cache %s error: %s", method, exc)
            return None

    async def get(self, key: str) -> Any:
        value = await self._safe(self.memory, "get", key)
        if value is not None:
            logger.debug("Memory hit for %s", key)
            return value

        value = await self._safe(self.redis, "get", key)
        if value is not None:
            logger.debug("Redis hit for %s", key)
            await self._safe(self.memory, "set", key, value)
            return value

        value = await self._safe(self.disk, "get", key)
        if value is not None:
            logger.debug("Disk hit for %s", key)
            await self._safe(self.redis, "set", key, value)
            await self._safe(self.memory, "set", key, value)
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self._safe(self.memory, "set", key, value, ttl=ttl)
        await self._safe(self.redis, "set", key, value, ttl=ttl)
        await self._safe(self.disk, "set", key, value, ttl=ttl)

    async def clear_all(self) -> None:
        for cache in (self.memory, self.redis, self.disk):
            await self._safe(cache, "clear")
