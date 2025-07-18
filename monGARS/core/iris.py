import asyncio
import logging
from typing import Optional
from urllib.parse import quote_plus, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class Iris:
    """Simple web scraper for retrieving page summaries."""

    _semaphore = asyncio.Semaphore(5)

    async def fetch_text(self, url: str) -> Optional[str]:
        """Fetch page text after validating the URL."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            logger.error("Invalid URL: %s", url)
            return None
        async with self._semaphore, httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=10)
                response.raise_for_status()
                text = trafilatura.extract(response.text)
                return text.strip() if text else None
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
            ) as e:  # pragma: no cover - network issues
                logger.error("Iris fetch error for %s: %s", url, e)
                return None

    async def search(self, query: str) -> Optional[str]:
        """Return snippet from the first DuckDuckGo result."""
        query = query.strip()
        if not query:
            return None
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        async with self._semaphore, httpx.AsyncClient() as client:
            try:
                response = await client.get(search_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                snippet = soup.select_one("div.result__snippet") or soup.select_one(
                    "a.result__a"
                )
                return snippet.get_text(strip=True) if snippet else None
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
            ) as e:  # pragma: no cover - network issues
                logger.error("Iris search error for %s: %s", query, e)
                return None
