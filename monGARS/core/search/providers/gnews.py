"""Google News RSS provider."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus, urlparse

import feedparser

from ..contracts import NormalizedHit

logger = logging.getLogger(__name__)


class GNewsProvider:
    """Fetch headlines from the public Google News RSS endpoint."""

    BASE_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    def __init__(self, *, timeout: float = 6.0) -> None:
        self._timeout = timeout

    async def search(
        self, query: str, *, lang: Optional[str] = None, max_results: int = 8
    ) -> list[NormalizedHit]:
        if not query.strip():
            return []

        feed_url = self.BASE_URL.format(query=quote_plus(query))
        entries = await asyncio.to_thread(feedparser.parse, feed_url)
        hits: list[NormalizedHit] = []
        for entry in entries.entries[:max_results]:
            published = None
            if getattr(entry, "published_parsed", None):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            link = getattr(entry, "link", "")
            if not link:
                continue
            snippet = getattr(entry, "summary", "") or ""
            hits.append(
                NormalizedHit(
                    provider="gnews",
                    title=getattr(entry, "title", ""),
                    url=link,
                    snippet=snippet,
                    published_at=published,
                    event_date=None,
                    source_domain=urlparse(link).netloc,
                    lang=lang or "en",
                    raw={"entry": entry},
                )
            )

        logger.debug(
            "gnews.search.completed",
            extra={"result_count": len(hits), "query_len": len(query)},
        )
        return hits
