from __future__ import annotations

from typing import Any, Protocol

class _AsyncDriver(Protocol):
    async def close(self) -> None: ...

class AsyncGraphDatabase:
    @staticmethod
    def driver(*args: Any, **kwargs: Any) -> _AsyncDriver: ...

__all__ = ["AsyncGraphDatabase"]
