from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")

class TTLCache(Generic[_K, _V]):
    def __init__(
        self,
        maxsize: int,
        ttl: float,
        *,
        timer: Callable[[], float] | None = ...,
        **kwargs: Any,
    ) -> None: ...

    def __contains__(self, key: _K) -> bool: ...

    def __getitem__(self, key: _K) -> _V: ...

    def __setitem__(self, key: _K, value: _V) -> None: ...

    def get(self, key: _K, default: _V | None = ...) -> _V | None: ...

    def pop(self, key: _K, default: _V | None = ...) -> _V | None: ...

__all__ = ["TTLCache"]
