"""Fact-checking providers using public RSS feeds."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import List
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency
    import feedparser
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    feedparser = None  # type: ignore[assignment]

from ..contracts import NormalizedHit

_VERDICT_RE = re.compile(
    r"(true|mostly true|half true|mostly false|false|pants on fire|mixture|unproven|miscaptioned)",
    re.IGNORECASE,
)


class _RSSFactChecker:
    BASE_URL: str
    PROVIDER: str

    async def search(self, query: str, max_results: int = 8) -> List[NormalizedHit]:
        if feedparser is None:
            return []
        feed = await asyncio.to_thread(feedparser.parse, self.BASE_URL)
        query_lower = query.lower()
        hits: list[NormalizedHit] = []
        for entry in feed.entries:
            searchable = (entry.title + " " + getattr(entry, "summary", "")).lower()
            if query_lower not in searchable:
                continue
            verdict = ""
            match = _VERDICT_RE.search(searchable)
            if match:
                verdict = match.group(1)
            timestamp = getattr(entry, "published_parsed", None)
            published_at = (
                datetime(*timestamp[:6], tzinfo=timezone.utc) if timestamp else None
            )
            url = entry.link
            domain = urlparse(url).netloc
            snippet = (verdict + " — " + getattr(entry, "summary", "")).strip()
            hits.append(
                NormalizedHit(
                    provider=self.PROVIDER,
                    title=entry.title,
                    url=url,
                    snippet=snippet[:500],
                    published_at=published_at,
                    event_date=None,
                    source_domain=domain,
                    lang="en",
                    raw={"verdict": verdict},
                )
            )
            if len(hits) >= max_results:
                break
        return hits


class PolitiFactProvider(_RSSFactChecker):
    BASE_URL = "https://www.politifact.com/rss/factchecks/"
    PROVIDER = "politifact"


class SnopesProvider(_RSSFactChecker):
    BASE_URL = "https://www.snopes.com/feed/"
    PROVIDER = "snopes"


__all__ = ["PolitiFactProvider", "SnopesProvider"]
