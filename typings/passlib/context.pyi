from __future__ import annotations

from typing import Any

class CryptContext:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def hash(self, password: str) -> str: ...

    def verify(self, password: str, hash: str) -> bool: ...

__all__ = ["CryptContext"]
