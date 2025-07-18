import logging
from typing import Optional

import httpx
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class Iris:
    """Simple web scraper for retrieving page summaries."""

    async def fetch_text(self, url: str) -> Optional[str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=10)
                response.raise_for_status()
                text = trafilatura.extract(response.text)
                return text.strip() if text else None
            except Exception as e:  # pragma: no cover - network issues
                logger.error("Iris fetch error for %s: %s", url, e)
                return None

    async def search(self, query: str) -> Optional[str]:
        """Return snippet from the first search result."""
        search_url = f"https://duckduckgo.com/html/?q={httpx.utils.quote(query)}"
        text = await self.fetch_text(search_url)
        if not text:
            return None
        soup = BeautifulSoup(text, "html.parser")
        snippet = soup.find(class_="result__snippet")
        return snippet.get_text(strip=True) if snippet else None
