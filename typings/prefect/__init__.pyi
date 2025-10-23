from __future__ import annotations

from typing import Any, Callable, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., Any])


@overload
def flow(__fn: _F, /) -> _F: ...


@overload
def flow(
    *,
    name: str | None = ...,
    **kwargs: Any,
) -> Callable[[_F], _F]: ...


__all__ = ["flow"]
