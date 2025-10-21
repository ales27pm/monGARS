"""DuckDuckGo HTML provider."""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlsplit

import httpx
from bs4 import BeautifulSoup

from ..contracts import NormalizedHit

logger = logging.getLogger(__name__)


class DDGProvider:
    """Scrape DuckDuckGo HTML results without requiring API keys."""

    BASE_URL = "https://duckduckgo.com/html/"

    def __init__(self, client: httpx.AsyncClient, *, timeout: float = 6.0) -> None:
        self._client = client
        self._timeout = timeout

    async def search(
        self, query: str, *, lang: Optional[str] = "en", max_results: int = 8
    ) -> list[NormalizedHit]:
        if not query.strip():
            return []

        params = {"q": query}
        if lang:
            params["kl"] = f"{lang}-en" if lang != "en" else "us-en"

        response = await self._client.get(
            f"{self.BASE_URL}?{urlencode(params)}", timeout=self._timeout
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results: list[NormalizedHit] = []
        for result in soup.select(".result"):
            anchor = result.select_one("a.result__a")
            snippet_node = result.select_one(".result__snippet")
            if anchor is None:
                continue
            href = anchor.get("href")
            if not href:
                continue
            resolved_href = href
            try:
                parts = urlsplit(href)
                if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith(
                    "/l/"
                ):
                    target = parse_qs(parts.query).get("uddg", [None])[0]
                    if target:
                        resolved_href = unquote(target)
            except Exception:  # pragma: no cover - defensive parsing
                resolved_href = href
            title = anchor.get_text(" ", strip=True)
            snippet = (
                snippet_node.get_text(" ", strip=True)
                if snippet_node is not None
                else ""
            )
            domain = urlparse(resolved_href).netloc
            results.append(
                NormalizedHit(
                    provider="ddg",
                    title=title,
                    url=resolved_href,
                    snippet=snippet,
                    published_at=None,
                    event_date=None,
                    source_domain=domain,
                    lang=lang,
                    raw={"title": title, "snippet": snippet, "href": href},
                )
            )
            if len(results) >= max_results:
                break

        logger.debug(
            "ddg.search.completed",
            extra={"result_count": len(results), "query_len": len(query)},
        )

        return results
