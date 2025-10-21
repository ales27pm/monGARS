from __future__ import annotations

from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

def flow(
    *args: Any, **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

__all__ = ["flow"]
