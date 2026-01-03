"""PubMed provider powered by NCBI e-utils."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import List
from urllib.parse import urlparse

import httpx

from ..contracts import NormalizedHit


class PubMedProvider:
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(self, client: httpx.AsyncClient, timeout: float = 6.0) -> None:
        self._client = client
        self._timeout = timeout

    @staticmethod
    def _parse_pub_date(value: str) -> datetime | None:
        try:
            candidate = datetime.fromisoformat(value.replace(" ", "-") + "T00:00:00")
        except ValueError:
            candidate = None
            for pattern in ("%Y %b %d", "%Y %b"):
                try:
                    parsed = datetime.strptime(value, pattern)
                except ValueError:
                    continue
                if pattern == "%Y %b":
                    parsed = parsed.replace(day=1)
                candidate = parsed
                break
        if candidate is None:
            return None
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=timezone.utc)
        return candidate

    async def search(self, query: str, max_results: int = 8) -> List[NormalizedHit]:
        esearch = await self._client.get(
            self.ESEARCH_URL,
            params={
                "db": "pubmed",
                "retmode": "json",
                "term": query,
                "retmax": max_results,
            },
            timeout=self._timeout,
        )
        esearch.raise_for_status()
        ids = esearch.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        esummary = await self._client.get(
            self.ESUMMARY_URL,
            params={"db": "pubmed", "retmode": "json", "id": ",".join(ids)},
            timeout=self._timeout,
        )
        esummary.raise_for_status()
        payload = esummary.json().get("result", {})
        hits: list[NormalizedHit] = []
        for pubmed_id in ids:
            item = payload.get(pubmed_id, {})
            title = html.unescape(item.get("title", "")).strip()
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
            domain = urlparse(url).netloc
            pub_date = item.get("pubdate") or item.get("sortpubdate")
            published_at = None
            if isinstance(pub_date, str) and pub_date:
                published_at = self._parse_pub_date(pub_date)
            snippet = (
                f"{item.get('sortfirstauthor', '')}. {item.get('source', '')}".strip()
            )
            hits.append(
                NormalizedHit(
                    provider="pubmed",
                    title=title or url,
                    url=url,
                    snippet=snippet[:500],
                    published_at=published_at,
                    event_date=None,
                    source_domain=domain,
                    lang="en",
                    raw=item,
                )
            )
        return hits


__all__ = ["PubMedProvider"]
