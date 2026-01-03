from __future__ import annotations

from typing import Any

class EmissionsTracker:
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> Any: ...

__all__ = ["EmissionsTracker"]
