from __future__ import annotations

from typing import Any, Mapping

def safe_load(stream: str | bytes | bytearray) -> Any: ...

def safe_dump(data: Any, *args: Any, **kwargs: Any) -> str: ...

__all__ = ["safe_load", "safe_dump"]
