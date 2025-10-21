from __future__ import annotations

from typing import Any, Protocol

class Cache(Protocol):
    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any, ttl: int | None = ...) -> None: ...

    async def clear(self) -> None: ...

class _Caches:
    def set_config(self, config: Any) -> None: ...

    def get(self, name: str) -> Cache: ...

caches: _Caches

__all__ = ["Cache", "caches"]
