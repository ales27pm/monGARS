from __future__ import annotations

from typing import Any, Callable, TypeVar

_T = TypeVar("_T", bound=Callable[..., Any])

class Celery:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def task(
        self, func: _T | None = ..., *args: Any, **kwargs: Any
    ) -> Callable[..., Any]: ...
    def send_task(
        self,
        name: str,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ) -> Any: ...

__all__ = ["Celery"]
