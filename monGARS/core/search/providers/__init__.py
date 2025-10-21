"""Built-in search providers for the orchestrator."""

from __future__ import annotations

from .ddg import DDGProvider
from .wikipedia import WikipediaProvider

__all__ = ["DDGProvider", "WikipediaProvider"]

try:  # pragma: no cover - optional dependency
    from .gnews import GNewsProvider
except ModuleNotFoundError:  # feedparser not installed
    GNewsProvider = None  # type: ignore[assignment]
else:
    __all__.append("GNewsProvider")
