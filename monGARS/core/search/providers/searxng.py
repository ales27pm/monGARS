"""SearxNG JSON API provider."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx

from ..contracts import NormalizedHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearxNGConfig:
    """Configuration options for the SearxNG provider."""

    base_url: str
    api_key: str | None = None
    categories: Sequence[str] | None = None
    safesearch: int | None = None
    default_language: str | None = "en"
    max_results: int | None = 10
    engines: Sequence[str] | None = None
    time_range: str | None = None
    sitelimit: str | None = None
    page_size: int | None = None
    max_pages: int = 1
    language_strict: bool = False


class SearxNGProvider:
    """Query a SearxNG instance via its JSON API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        config: SearxNGConfig,
        timeout: float = 6.0,
    ) -> None:
        if not config.base_url:
            msg = "SearxNG base_url must be provided"
            raise ValueError(msg)
        self._client = client
        self._config = config
        self._timeout = timeout
        self._result_cap = (
            max(1, config.max_results)
            if isinstance(config.max_results, int) and config.max_results > 0
            else None
        )
        self._page_size = (
            max(1, config.page_size)
            if isinstance(config.page_size, int) and config.page_size > 0
            else None
        )
        self._max_pages = max(1, config.max_pages)
        self._search_endpoint = self._normalise_base_url(config.base_url)

    @staticmethod
    def _normalise_base_url(base_url: str) -> str:
        trimmed = base_url.rstrip("/")
        return f"{trimmed}/search"

    async def search(
        self,
        query: str,
        *,
        lang: Optional[str] = None,
        max_results: int = 8,
    ) -> list[NormalizedHit]:
        """Return normalised results for *query* from SearxNG."""

        if not query.strip():
            return []

        target_lang = lang or self._config.default_language
        headers: dict[str, str] = {
            "Accept": "application/json",
        }
        if target_lang and self._config.language_strict:
            headers["Accept-Language"] = target_lang
        if self._config.api_key:
            headers["Authorization"] = self._config.api_key

        limit = max_results
        if self._result_cap is not None:
            limit = min(limit, self._result_cap)
        limit = max(1, limit)
        per_page = min(limit, self._page_size or limit)

        normalized: list[NormalizedHit] = []
        pages_visited = 0
        for page in range(1, self._max_pages + 1):
            if len(normalized) >= limit:
                break
            params = self._build_params(
                query,
                target_lang=target_lang,
                page=page,
                per_page=per_page,
            )
            response = await self._client.get(
                self._search_endpoint,
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
            page_hits = self._normalise_payload(
                payload, target_lang=target_lang, remaining=limit - len(normalized)
            )
            normalized.extend(page_hits)
            pages_visited += 1
            if not self._should_continue(payload, expected_count=per_page):
                break

        logger.debug(
            "searxng.search.completed",
            extra={
                "result_count": len(normalized),
                "query_len": len(query),
                "pages": pages_visited,
            },
        )
        return normalized

    def _build_params(
        self,
        query: str,
        *,
        target_lang: Optional[str],
        page: int,
        per_page: int,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "q": query,
            "format": "json",
            "pageno": page,
            "count": per_page,
        }
        if target_lang:
            params["language"] = target_lang
        if self._config.categories:
            params["categories"] = ",".join(self._config.categories)
        if self._config.safesearch is not None:
            params["safesearch"] = self._config.safesearch
        if self._config.engines:
            params["engines"] = ",".join(self._config.engines)
        if self._config.time_range:
            params["time_range"] = self._config.time_range
        if self._config.sitelimit:
            params["sitelimit"] = self._config.sitelimit
        return params

    def _normalise_payload(
        self,
        payload: object,
        *,
        target_lang: Optional[str],
        remaining: int,
    ) -> list[NormalizedHit]:
        if not isinstance(payload, dict):
            return []
        hits: list[NormalizedHit] = []
        results = payload.get("results", [])
        if isinstance(results, list):
            for raw in results:
                if len(hits) >= remaining:
                    break
                hit = self._normalise_result(raw, target_lang)
                if hit is None:
                    continue
                hits.append(hit)
        if len(hits) < remaining:
            infobox_hits = self._normalise_infoboxes(
                payload.get("infoboxes"), target_lang, remaining - len(hits)
            )
            hits.extend(infobox_hits)
        return hits

    def _normalise_result(
        self, raw: object, target_lang: Optional[str]
    ) -> NormalizedHit | None:
        if not isinstance(raw, dict):
            return None
        url = raw.get("url") or raw.get("link")
        if not isinstance(url, str) or not url:
            return None
        title = raw.get("title")
        if not isinstance(title, str) or not title:
            title = url
        snippet_fields: Sequence[str] = (
            raw.get("content"),
            raw.get("description"),
            raw.get("summary"),
        )
        snippet = next((s for s in snippet_fields if isinstance(s, str) and s), "")
        published_at = self._parse_datetime(raw.get("publishedDate"))
        if published_at is None:
            published_at = self._parse_datetime(raw.get("published"))
        event_date = self._parse_datetime(raw.get("eventDate"))
        domain = urlparse(url).netloc
        language = raw.get("language")
        if not isinstance(language, str) or not language:
            language = target_lang
        return NormalizedHit(
            provider="searxng",
            title=title,
            url=url,
            snippet=snippet,
            published_at=published_at,
            event_date=event_date,
            source_domain=domain,
            lang=language,
            raw=raw,
        )

    def _normalise_infoboxes(
        self,
        infoboxes: object,
        target_lang: Optional[str],
        limit: int,
    ) -> list[NormalizedHit]:
        if not isinstance(infoboxes, Sequence):
            return []
        hits: list[NormalizedHit] = []
        for entry in infoboxes:
            if len(hits) >= limit:
                break
            if not isinstance(entry, Mapping):
                continue
            entry_dict = dict(entry)
            url = self._extract_infobox_url(entry_dict.get("urls"))
            if url is None:
                continue
            title = entry_dict.get("infobox") or entry_dict.get("label") or url
            if not isinstance(title, str) or not title:
                title = url
            content = entry_dict.get("content") or entry_dict.get("description") or ""
            snippet = content if isinstance(content, str) else ""
            published_at = self._parse_datetime(entry_dict.get("date"))
            domain = urlparse(url).netloc
            raw_entry = dict(entry_dict)
            if "infobox" in raw_entry:
                raw_entry["infobox_label"] = raw_entry["infobox"]
            raw_entry["infobox"] = True
            hits.append(
                NormalizedHit(
                    provider="searxng",
                    title=title,
                    url=url,
                    snippet=snippet,
                    published_at=published_at,
                    event_date=None,
                    source_domain=domain,
                    lang=target_lang,
                    raw=raw_entry,
                )
            )
        return hits

    @staticmethod
    def _extract_infobox_url(urls: object) -> str | None:
        if isinstance(urls, str) and urls:
            return urls
        if isinstance(urls, Sequence):
            for candidate in urls:
                if isinstance(candidate, str) and candidate:
                    return candidate
                if isinstance(candidate, dict):
                    href = candidate.get("url") or candidate.get("href")
                    if isinstance(href, str) and href:
                        return href
        return None

    @staticmethod
    def _should_continue(payload: object, *, expected_count: int) -> bool:
        if not isinstance(payload, dict):
            return False
        results = payload.get("results")
        if not isinstance(results, list):
            return False
        return len(results) >= expected_count

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        text = value.strip()
        if not text:
            return None
        for candidate in SearxNGProvider._iter_datetime_candidates(text):
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _iter_datetime_candidates(text: str) -> Iterable[datetime | None]:
        yield SearxNGProvider._parse_isoformat(text)
        yield SearxNGProvider._parse_email_datetime(text)

    @staticmethod
    def _parse_isoformat(text: str) -> datetime | None:
        try:
            formatted = text.replace("Z", "+00:00") if text.endswith("Z") else text
            dt = datetime.fromisoformat(formatted)
        except ValueError:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _parse_email_datetime(text: str) -> datetime | None:
        try:
            dt = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


__all__ = ["SearxNGConfig", "SearxNGProvider"]
