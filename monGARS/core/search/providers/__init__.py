"""Built-in search providers for the orchestrator."""

from __future__ import annotations

from .ddg import DDGProvider
from .gnews import GNewsProvider
from .wikipedia import WikipediaProvider

__all__ = ["DDGProvider", "GNewsProvider", "WikipediaProvider"]
