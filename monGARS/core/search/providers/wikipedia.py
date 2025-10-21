"""Wikipedia search provider."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from ..contracts import NormalizedHit

logger = logging.getLogger(__name__)


class WikipediaProvider:
    """Query the Wikipedia API for summary information."""

    def __init__(self, client: httpx.AsyncClient, *, timeout: float = 6.0) -> None:
        self._client = client
        self._timeout = timeout

    async def search(
        self, query: str, *, lang: Optional[str] = "en", max_results: int = 3
    ) -> list[NormalizedHit]:
        if not query.strip():
            return []

        params = {
            "action": "opensearch",
            "search": query,
            "limit": max_results,
            "namespace": 0,
            "format": "json",
        }
        base_url = f"https://{lang or 'en'}.wikipedia.org/w/api.php"
        response = await self._client.get(
            base_url, params=params, timeout=self._timeout
        )
        response.raise_for_status()

        payload = response.json()
        titles = payload[1] if len(payload) > 1 else []
        urls = payload[3] if len(payload) > 3 else []
        descriptions = payload[2] if len(payload) > 2 else []

        results: list[NormalizedHit] = []
        for title, url, description in zip(titles, urls, descriptions):
            results.append(
                NormalizedHit(
                    provider="wikipedia",
                    title=title,
                    url=url,
                    snippet=description or "",
                    published_at=None,
                    event_date=None,
                    source_domain="wikipedia.org",
                    lang=lang,
                    raw={"title": title, "description": description},
                )
            )

        logger.debug(
            "wikipedia.search.completed",
            extra={"result_count": len(results), "query_len": len(query)},
        )
        return results
