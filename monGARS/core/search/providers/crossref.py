"""Crossref search provider for academic literature."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from urllib.parse import urlparse

import httpx

from ..contracts import NormalizedHit


class CrossrefProvider:
    """Query the Crossref works API for scholarly articles."""

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, client: httpx.AsyncClient, timeout: float = 6.0) -> None:
        self._client = client
        self._timeout = timeout

    @staticmethod
    def _parse_date(parts: list[int] | list[str] | None) -> datetime | None:
        if not parts:
            return None
        try:
            year, month, day = (list(parts) + [1, 1, 1])[:3]
            return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
        except Exception:  # pragma: no cover - defensive
            return None

    async def search(self, query: str, max_results: int = 8) -> List[NormalizedHit]:
        params = {
            "query": query,
            "rows": max_results,
            "select": "title,URL,issued,abstract,language",
        }
        response = await self._client.get(
            self.BASE_URL,
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json().get("message", {})
        items = payload.get("items", [])
        hits: list[NormalizedHit] = []
        for item in items:
            url = item.get("URL") or ""
            title = " ".join(item.get("title") or []) or url
            abstract = item.get("abstract") or ""
            issued = item.get("issued", {}).get("date-parts", [])
            published_at = self._parse_date(issued[0]) if issued else None
            language = item.get("language")
            domain = urlparse(url).netloc
            hits.append(
                NormalizedHit(
                    provider="crossref",
                    title=title.strip(),
                    url=url,
                    snippet=abstract.strip()[:500],
                    published_at=published_at,
                    event_date=None,
                    source_domain=domain,
                    lang=language,
                    raw=item,
                )
            )
        return hits


__all__ = ["CrossrefProvider"]
