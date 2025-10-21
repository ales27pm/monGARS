from __future__ import annotations

from typing import Any, Iterable

def setup(*args: Any, **kwargs: Any) -> None: ...
def find_packages(*args: Any, **kwargs: Any) -> list[str]: ...

__all__ = ["setup", "find_packages"]
