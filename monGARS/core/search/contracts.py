"""Contracts for search providers and verification bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Sequence
from urllib.parse import urlparse


@dataclass(slots=True)
class NormalizedHit:
    """Representation of a single search result across providers."""

    provider: str
    title: str
    url: str
    snippet: str
    published_at: Optional[datetime]
    event_date: Optional[datetime]
    source_domain: str
    lang: Optional[str]
    raw: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.source_domain:
            self.source_domain = urlparse(self.url).netloc
        else:
            self.source_domain = self.source_domain.lower()

    def is_trustworthy(self) -> bool:
        domain = self.source_domain
        return (
            domain.endswith(".gov")
            or domain.endswith(".gouv")
            or domain.endswith(".gouv.fr")
            or domain.endswith(".edu")
            or domain.endswith(".ac.uk")
            or domain.endswith(".eu")
        )


@dataclass(slots=True)
class VerifiedBundle:
    """Aggregated verification output for a search query."""

    query: str
    hits: List[NormalizedHit]
    agreed_facts: Dict[str, str]
    disagreements: Dict[str, List[str]]
    confidence: float
    primary_citation: Optional[str]
    citations: List[str]


class SearchProvider(Protocol):
    """Protocol for pluggable search providers."""

    async def search(
        self,
        query: str,
        *,
        lang: Optional[str] = None,
        max_results: int = 8,
    ) -> Sequence[NormalizedHit]: ...
