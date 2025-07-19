import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from aiocache import Cache, caches

from monGARS.config import get_settings

logger = logging.getLogger(__name__)
settings: Any | None = None


class SimpleDiskCache:
    """Very small file-based cache used when aiocache FileCache is unavailable."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()

    def _path(self, key: str) -> Path:
        name = hashlib.sha256(key.encode()).hexdigest()
        return self.directory / f"{name}.json"

    async def get(self, key: str) -> Any:
        async with self.lock:
            path = self._path(key)
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            expires = data.get("expires")
            if expires and expires <= time.time():
                path.unlink(missing_ok=True)
                return None
            return data.get("value")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        async with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._path(key)
            data = {
                "value": value,
                "expires": time.time() + ttl if ttl else None,
            }
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh)

    async def clear(self) -> None:
        async with self.lock:
            for file in self.directory.glob("*.json"):
                file.unlink(missing_ok=True)


class TieredCache:
    """Memory, Redis and disk-backed cache with graceful fallbacks."""

    def __init__(self, directory: str | None = None) -> None:
        global settings
        if settings is None:
            settings = get_settings()
            caches.set_config(
                {
                    "default": {
                        "cache": "aiocache.SimpleMemoryCache",
                        "serializer": {
                            "class": "aiocache.serializers.PickleSerializer"
                        },
                    },
                    "redis": {
                        "cache": "aiocache.RedisCache",
                        "endpoint": settings.redis_url.host,
                        "port": settings.redis_url.port,
                        "db": (
                            int(db)
                            if (db := settings.redis_url.path.lstrip("/")).isdigit()
                            else 0
                        ),
                        "serializer": {
                            "class": "aiocache.serializers.PickleSerializer"
                        },
                        "timeout": 1,
                    },
                }
            )
        self.memory = caches.get("default")
        if hasattr(__import__("aiocache"), "RedisCache"):
            self.redis = caches.get("redis")
        else:
            self.redis = caches.get("default")
        self.disk = SimpleDiskCache(directory or settings.DISK_CACHE_PATH)
        self.caches = [self.memory, self.redis, self.disk]

    async def _safe(self, cache: Cache, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            func = getattr(cache, method)
            return await func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # pragma: no cover - depends on external services
            logger.warning("Cache %s error: %s", method, exc, exc_info=True)
            return None

    async def get(self, key: str) -> Any:
        for idx, cache in enumerate(self.caches):
            value = await self._safe(cache, "get", key)
            if value is not None:
                logger.debug("%s hit for %s", cache.__class__.__name__, key)
                for prev in self.caches[:idx]:
                    await self._safe(prev, "set", key, value)
                return value
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        for cache in self.caches:
            try:
                await getattr(cache, "set")(key, value, ttl=ttl)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:  # pragma: no cover - individual logging
                logger.error(
                    "Failed to set key '%s' in %s cache: %s",
                    key,
                    cache.__class__.__name__,
                    exc,
                    exc_info=True,
                )

    async def clear_all(self) -> None:
        for cache in self.caches:
            await self._safe(cache, "clear")
