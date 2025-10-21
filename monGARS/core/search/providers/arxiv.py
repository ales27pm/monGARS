"""arXiv provider returning the latest preprints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List
from urllib.parse import urlencode, urlparse

try:  # pragma: no cover - optional dependency
    import feedparser
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    feedparser = None  # type: ignore[assignment]

from ..contracts import NormalizedHit


class ArxivProvider:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: float = 6.0) -> None:
        self._timeout = timeout

    async def search(self, query: str, max_results: int = 8) -> List[NormalizedHit]:
        if feedparser is None:
            return []
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "lastUpdatedDate",
        }
        feed = await asyncio.to_thread(
            feedparser.parse, f"{self.BASE_URL}?{urlencode(params)}"
        )
        hits: list[NormalizedHit] = []
        for entry in feed.entries[:max_results]:
            url = entry.link
            domain = urlparse(url).netloc
            timestamp = getattr(entry, "updated_parsed", None) or getattr(
                entry, "published_parsed", None
            )
            published_at = (
                datetime(*timestamp[:6], tzinfo=timezone.utc) if timestamp else None
            )
            summary = getattr(entry, "summary", "") or ""
            authors = [author.name for author in getattr(entry, "authors", [])]
            hits.append(
                NormalizedHit(
                    provider="arxiv",
                    title=entry.title,
                    url=url,
                    snippet=summary.strip()[:500],
                    published_at=published_at,
                    event_date=None,
                    source_domain=domain,
                    lang="en",
                    raw={"authors": authors},
                )
            )
        return hits


__all__ = ["ArxivProvider"]
