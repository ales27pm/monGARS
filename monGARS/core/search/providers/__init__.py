"""Built-in search providers for the orchestrator."""

from __future__ import annotations

from .arxiv import ArxivProvider
from .crossref import CrossrefProvider
from .ddg import DDGProvider
from .factcheckers import PolitiFactProvider, SnopesProvider
from .pubmed import PubMedProvider
from .wikipedia import WikipediaProvider

__all__ = [
    "ArxivProvider",
    "CrossrefProvider",
    "DDGProvider",
    "PolitiFactProvider",
    "PubMedProvider",
    "SnopesProvider",
    "WikipediaProvider",
]

try:  # pragma: no cover - optional dependency
    from .gnews import GNewsProvider
except ModuleNotFoundError:  # feedparser not installed
    GNewsProvider = None  # type: ignore[assignment]
else:
    __all__.append("GNewsProvider")
