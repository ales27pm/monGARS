from __future__ import annotations

from typing import Any, Mapping

class Deployment:
    @classmethod
    def build_from_flow(
        cls,
        *,
        flow: Any,
        name: str,
        schedule: Any,
        parameters: Mapping[str, Any] | None = ...,
        tags: list[str] | None = ...,
    ) -> Deployment: ...
    def apply(self) -> None: ...

__all__ = ["Deployment"]
