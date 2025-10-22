"""Utilities for lightweight web retrieval used in the curiosity engine."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse, urlunparse

import httpx
import trafilatura
from trafilatura import metadata as trafilatura_metadata
from bs4 import BeautifulSoup

from monGARS.core.search import NormalizedHit

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from monGARS.core.search.orchestrator import SearchOrchestrator

from .search.metadata import parse_date_from_text
from .search.schema_org import parse_schema_org

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS: Mapping[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_AUTHOR_SPLIT_RE = re.compile(r"[,;]|\band\b", re.IGNORECASE)
_TOPIC_SPLIT_RE = re.compile(r"[,;/]")
MAX_SNIPPET_LENGTH = 500


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True)
class IrisDocument:
    """Structured representation of extracted web content."""

    url: str
    text: str | None
    title: str | None = None
    summary: str | None = None
    language: str | None = None
    event_date: datetime | None = None
    published_at: datetime | None = None
    modified_at: datetime | None = None
    event_start: datetime | None = None
    event_end: datetime | None = None
    authors: list[str] | None = None
    publisher: str | None = None
    organization: str | None = None
    location_name: str | None = None
    comments: str | None = None
    tags: list[str] | None = None
    categories: list[str] | None = None
    image_url: str | None = None
    page_type: str | None = None
    fingerprint: str | None = None


class Iris:
    """Retrieve lightweight snippets from the public web with resiliency."""

    def __init__(
        self,
        *,
        max_concurrency: int = 5,
        request_timeout: float = 10.0,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        max_content_length: int = 1_500_000,
        headers: Mapping[str, str] | None = None,
        search_cache_ttl: float | None = 300.0,
        search_cache_size: int = 128,
        document_cache_ttl: float | None = 900.0,
        document_cache_size: int = 128,
        client_factory: Callable[..., httpx.AsyncClient] | None = None,
        search_orchestrator: "SearchOrchestrator | None" = None,
        search_language: str | None = "en",
        orchestrator_result_limit: int = 5,
    ) -> None:
        if max_concurrency <= 0:
            msg = "max_concurrency must be a positive integer"
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries cannot be negative"
            raise ValueError(msg)
        if max_content_length <= 0:
            msg = "max_content_length must be positive"
            raise ValueError(msg)
        if search_cache_ttl is not None and search_cache_ttl <= 0:
            msg = "search_cache_ttl must be positive when provided"
            raise ValueError(msg)
        if search_cache_size <= 0:
            msg = "search_cache_size must be a positive integer"
            raise ValueError(msg)
        if document_cache_ttl is not None and document_cache_ttl <= 0:
            msg = "document_cache_ttl must be positive when provided"
            raise ValueError(msg)
        if document_cache_size <= 0:
            msg = "document_cache_size must be a positive integer"
            raise ValueError(msg)
        if orchestrator_result_limit <= 0:
            msg = "orchestrator_result_limit must be a positive integer"
            raise ValueError(msg)

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout = httpx.Timeout(request_timeout)
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._max_content_length = max_content_length
        self._search_cache_ttl = search_cache_ttl
        self._search_cache_size = search_cache_size
        self._document_cache_ttl = document_cache_ttl
        self._document_cache_size = document_cache_size
        self._search_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._search_cache_lock = asyncio.Lock()
        self._document_cache: OrderedDict[str, tuple[float, IrisDocument]] = (
            OrderedDict()
        )
        self._document_cache_lock = asyncio.Lock()
        self._inflight_documents: dict[str, asyncio.Future[IrisDocument | None]] = {}
        self._inflight_lock = asyncio.Lock()
        merged_headers = dict(_DEFAULT_HEADERS)
        if headers:
            merged_headers.update(headers)
        self._client_options = {
            "headers": merged_headers,
            "timeout": self._timeout,
            "follow_redirects": True,
            "limits": httpx.Limits(
                max_connections=max_concurrency * 2,
                max_keepalive_connections=max_concurrency,
            ),
        }
        self._client_factory = client_factory or httpx.AsyncClient
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._search_orchestrator: "SearchOrchestrator | None" = search_orchestrator
        self._search_language = search_language
        self._orchestrator_result_limit = orchestrator_result_limit

    async def __aenter__(self) -> "Iris":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        async with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def fetch_document(self, url: str) -> IrisDocument | None:
        """Fetch structured content for ``url`` using resilient extraction."""

        canonical_url = self._canonicalise_url(url)
        cached = await self._get_cached_document(canonical_url)
        if cached is not None:
            return cached

        future = await self._get_or_create_inflight(canonical_url, url)
        try:
            document = await future
        finally:
            await self._clear_inflight(canonical_url, future)

        if document is None:
            return None

        await self._store_cached_document(canonical_url, document)
        final_canonical = self._canonicalise_url(document.url)
        if final_canonical != canonical_url:
            await self._store_cached_document(final_canonical, document)
        return document

    async def fetch_text(self, url: str) -> Optional[str]:
        """Fetch the main textual content for a given HTTP(S) URL."""

        document = await self.fetch_document(url)
        return document.text if document and document.text else None

    async def search(self, query: str) -> Optional[str]:
        """Return contextual snippet using the first DuckDuckGo result for *query*."""

        query = query.strip()
        if not query:
            logger.debug("iris.search.empty_query")
            return None

        cache_key = query.casefold()

        cached = await self._get_cached_snippet(cache_key)
        if cached is not None:
            return cached

        orchestrated = await self._search_with_orchestrator(query, cache_key)
        if orchestrated is not None:
            return orchestrated

        return await self._search_with_duckduckgo(query, cache_key)

    def attach_search_orchestrator(
        self, orchestrator: "SearchOrchestrator | None"
    ) -> None:
        """Attach or replace the search orchestrator used for snippets."""

        self._search_orchestrator = orchestrator

    async def _search_with_orchestrator(self, query: str, cache_key: str) -> str | None:
        orchestrator = self._search_orchestrator
        if orchestrator is None:
            return None
        lang = self._search_language or "en"
        try:
            hits: Sequence[NormalizedHit] = await orchestrator.search(
                query,
                lang=lang,
                max_results=self._orchestrator_result_limit,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - network dependent
            logger.debug(
                "iris.search.orchestrator_error",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return None

        for hit in hits:
            snippet_candidate = self._truncate_snippet(hit.snippet)
            if snippet_candidate:
                await self._store_cached_snippet(cache_key, snippet_candidate)
                return snippet_candidate

            url = hit.url
            if not url:
                continue
            try:
                document = await self.fetch_document(url)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network dependent
                logger.debug(
                    "iris.search.orchestrator_document_error",
                    extra={"url": url, "error": str(exc)},
                )
                continue
            if document is None:
                continue
            selected = self._select_snippet(document, hit.snippet)
            if selected:
                await self._store_cached_snippet(cache_key, selected)
                return selected

        return None

    async def _search_with_duckduckgo(self, query: str, cache_key: str) -> str | None:
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"

        async with self._semaphore:
            response = await self._request_with_retries("GET", search_url)

        if response is None or not self._is_textual_response(response):
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        primary_result = soup.select_one("div.result")
        result_link = None
        snippet_element = None
        if primary_result:
            result_link = primary_result.select_one("a.result__a")
            snippet_element = primary_result.select_one("div.result__snippet")
        if snippet_element is None:
            snippet_element = soup.select_one("div.result__snippet")
        if result_link is None:
            result_link = soup.select_one("a.result__a")

        snippet_text = (
            snippet_element.get_text(" ", strip=True) if snippet_element else None
        )

        resolved_url = None
        if result_link and result_link.has_attr("href"):
            resolved_url = self._resolve_result_url(result_link["href"])

        if resolved_url:
            document = await self.fetch_document(resolved_url)
            if document:
                best_candidate = self._select_snippet(document, snippet_text)
                if best_candidate:
                    await self._store_cached_snippet(cache_key, best_candidate)
                    return best_candidate

        if snippet_text:
            truncated = self._truncate_snippet(snippet_text)
            if truncated:
                await self._store_cached_snippet(cache_key, truncated)
                return truncated

        return None

    async def _get_or_create_inflight(
        self, canonical_url: str, original_url: str
    ) -> asyncio.Future[IrisDocument | None]:
        async with self._inflight_lock:
            future = self._inflight_documents.get(canonical_url)
            if future is not None:
                return future
            loop = asyncio.get_running_loop()
            future = loop.create_task(self._fetch_document_uncached(original_url))
            self._inflight_documents[canonical_url] = future
            return future

    async def _clear_inflight(
        self, canonical_url: str, future: asyncio.Future[IrisDocument | None]
    ) -> None:
        async with self._inflight_lock:
            existing = self._inflight_documents.get(canonical_url)
            if existing is future:
                self._inflight_documents.pop(canonical_url, None)

    async def _fetch_document_uncached(self, url: str) -> IrisDocument | None:
        response = await self._get_response(url)
        if response is None:
            return None
        return await self._extract_document(response)

    async def _request_with_retries(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response | None:
        """Execute an HTTP request with bounded retries."""

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network timing
                last_error = exc
                status_code = exc.response.status_code if exc.response else None
                should_retry = status_code is not None and 500 <= status_code < 600
                if should_retry and attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_time(attempt))
                    continue
                break
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.RequestError,
            ) as exc:  # pragma: no cover - network timing
                last_error = exc
                if attempt < self._max_retries:
                    await self._reset_client()
                    await asyncio.sleep(self._backoff_time(attempt))
                    continue
                break

        if last_error is not None:
            logger.error(
                "iris.request.failed",
                extra={
                    "url": url,
                    "error": str(last_error),
                    "attempts": self._max_retries + 1,
                },
            )
        return None

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None:
                self._client = self._client_factory(**self._client_options)
            return self._client

    async def _reset_client(self) -> None:
        async with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    def _backoff_time(self, attempt: int) -> float:
        return self._backoff_factor * (2**attempt)

    async def _get_response(self, url: str) -> httpx.Response | None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            logger.error(
                "iris.fetch_text.invalid_scheme",
                extra={"url": url, "scheme": parsed.scheme or ""},
            )
            return None

        async with self._semaphore:
            response = await self._request_with_retries("GET", url)

        if response is None:
            return None

        if not self._is_textual_response(response):
            logger.info(
                "iris.fetch_text.non_textual_response",
                extra={
                    "url": url,
                    "content_type": response.headers.get("Content-Type"),
                },
            )
            return None

        if self._content_too_large(response):
            logger.info(
                "iris.fetch_text.payload_too_large",
                extra={
                    "url": url,
                    "content_length": response.headers.get("Content-Length"),
                    "text_length": len(response.text),
                },
            )
            return None

        return response

    def _is_textual_response(self, response: httpx.Response) -> bool:
        content_type = response.headers.get("Content-Type", "").lower()
        if not content_type:
            return True
        textual_indicators = ("text", "json", "xml", "javascript")
        return any(token in content_type for token in textual_indicators)

    def _content_too_large(self, response: httpx.Response) -> bool:
        content_length = response.headers.get("Content-Length")
        if content_length and content_length.isdigit():
            if int(content_length) > self._max_content_length:
                return True
        return len(response.text) > self._max_content_length

    async def _extract_document(self, response: httpx.Response) -> IrisDocument | None:
        extracted_json: str | None = None
        try:
            html_text = response.text
            metadata_payload = await self._extract_trafilatura_metadata(
                html_text, str(response.request.url)
            )
            extracted_json = await asyncio.to_thread(
                trafilatura.extract,
                html_text,
                include_comments=True,
                include_tables=False,
                favor_precision=True,
                output_format="json",
                url=str(response.request.url),
                with_metadata=True,
            )
        except Exception as exc:  # pragma: no cover - library edge case
            logger.debug(
                "iris.fetch_text.trafilatura_failed",
                extra={"error": str(exc)},
            )
            metadata_payload = None

        document_data: Mapping[str, Any] | None = None
        if extracted_json:
            try:
                document_data = json.loads(extracted_json)
            except json.JSONDecodeError as exc:  # pragma: no cover - unexpected format
                logger.debug(
                    "iris.fetch_text.invalid_trafilatura_payload",
                    extra={"error": str(exc)},
                )

        fallback_text = None
        document: IrisDocument | None = None
        if document_data:
            document = self._document_from_trafilatura_json(
                document_data, str(response.request.url)
            )
        if document is None and metadata_payload:
            document = self._document_from_metadata(
                metadata_payload, str(response.request.url)
            )
        if document is None:
            fallback_text = self._fallback_text(response)
            if fallback_text:
                document = IrisDocument(
                    url=str(response.request.url),
                    text=fallback_text,
                    summary=None,
                    title=None,
                    language=None,
                )

        if document is None:
            return None

        if document_data:
            self._enrich_document_from_trafilatura_json(
                document, document_data, str(response.request.url)
            )
        if metadata_payload:
            self._enrich_document_from_metadata(document, metadata_payload)

        schema_meta = parse_schema_org(response.text)
        if schema_meta:
            if schema_meta.date_published and document.published_at is None:
                document.published_at = schema_meta.date_published
            if schema_meta.date_modified and document.modified_at is None:
                document.modified_at = schema_meta.date_modified
            if schema_meta.event_start and document.event_start is None:
                document.event_start = schema_meta.event_start
            if schema_meta.event_end and document.event_end is None:
                document.event_end = schema_meta.event_end
            if schema_meta.authors:
                if document.authors:
                    existing = {author.lower() for author in document.authors}
                    for author in schema_meta.authors:
                        lowered = author.lower()
                        if lowered not in existing:
                            document.authors.append(author)
                            existing.add(lowered)
                else:
                    document.authors = list(schema_meta.authors)
            if schema_meta.publisher and document.publisher is None:
                document.publisher = schema_meta.publisher
            if schema_meta.organization and document.organization is None:
                document.organization = schema_meta.organization
            if schema_meta.location_name and document.location_name is None:
                document.location_name = schema_meta.location_name

        fallback_source = document.summary or (document.text or "")[:800]
        if document.published_at is None:
            fallback_document_date = parse_date_from_text(fallback_source)
            if fallback_document_date is not None:
                document.published_at = _ensure_utc(fallback_document_date)
        if document.event_date is None:
            derived_event = document.event_start or parse_date_from_text(
                fallback_source
            )
            if derived_event is not None:
                document.event_date = _ensure_utc(derived_event)
        return document

    async def _extract_trafilatura_metadata(
        self, html_text: str, url: str
    ) -> Mapping[str, Any] | None:
        try:
            metadata_document = await asyncio.to_thread(
                trafilatura_metadata.extract_metadata,
                html_text,
                default_url=url,
            )
        except Exception as exc:  # pragma: no cover - library edge case
            logger.debug(
                "iris.fetch_text.metadata_failed",
                extra={"error": str(exc)},
            )
            return None
        if metadata_document is None:
            return None
        try:
            raw_metadata = metadata_document.as_dict()
        except Exception as exc:  # pragma: no cover - unexpected payload
            logger.debug(
                "iris.fetch_text.metadata_invalid",
                extra={"error": str(exc)},
            )
            return None
        filtered: dict[str, Any] = {}
        for key in (
            "title",
            "author",
            "url",
            "hostname",
            "sitename",
            "date",
            "filedate",
            "language",
            "description",
            "text",
            "raw_text",
            "excerpt",
            "source",
            "source-hostname",
            "image",
            "pagetype",
            "categories",
            "tags",
            "fingerprint",
            "comments",
        ):
            if key in raw_metadata:
                filtered[key] = raw_metadata[key]
        return filtered

    def _document_from_trafilatura_json(
        self, data: Mapping[str, Any], fallback_url: str
    ) -> IrisDocument | None:
        text = self._string_from_metadata_field(
            data.get("text") or data.get("raw_text")
        )
        summary = self._string_from_metadata_field(
            data.get("summary") or data.get("excerpt") or data.get("description")
        )
        title = self._string_from_metadata_field(data.get("title"))
        language = self._string_from_metadata_field(data.get("language"))
        canonical_url = (
            self._string_from_metadata_field(data.get("source"))
            or self._string_from_metadata_field(data.get("url"))
            or fallback_url
        )
        if not (text or summary or title):
            return None
        document = IrisDocument(
            url=canonical_url,
            text=text,
            summary=summary,
            title=title,
            language=language,
        )
        self._enrich_document_from_trafilatura_json(document, data, fallback_url)
        return document

    def _enrich_document_from_trafilatura_json(
        self,
        document: IrisDocument,
        data: Mapping[str, Any],
        fallback_url: str,
    ) -> None:
        canonical_url = self._string_from_metadata_field(
            data.get("source")
        ) or self._string_from_metadata_field(data.get("url"))
        if canonical_url:
            document.url = canonical_url
        else:
            document.url = document.url or fallback_url

        if not document.title:
            title = self._string_from_metadata_field(data.get("title"))
            if title:
                document.title = title

        if not document.summary:
            summary = self._string_from_metadata_field(
                data.get("summary") or data.get("excerpt") or data.get("description")
            )
            if summary:
                document.summary = summary

        if not document.text:
            text = self._string_from_metadata_field(
                data.get("text") or data.get("raw_text")
            )
            if text:
                document.text = text

        if not document.language:
            language = self._string_from_metadata_field(data.get("language"))
            if language:
                document.language = language

        authors = self._parse_authors(data.get("author"))
        if authors:
            if document.authors:
                existing = {author.casefold() for author in document.authors}
                for author in authors:
                    lowered = author.casefold()
                    if lowered not in existing:
                        document.authors.append(author)
                        existing.add(lowered)
            else:
                document.authors = authors

        publisher = self._string_from_metadata_field(
            data.get("source-hostname") or data.get("sitename")
        )
        if publisher and not document.publisher:
            document.publisher = publisher

        organization = self._string_from_metadata_field(data.get("hostname"))
        if organization and not document.organization:
            document.organization = organization

        if document.published_at is None:
            published = data.get("date")
            if isinstance(published, str):
                parsed = parse_date_from_text(published)
                if parsed is not None:
                    document.published_at = _ensure_utc(parsed)

        if document.modified_at is None:
            modified = data.get("filedate")
            if isinstance(modified, str):
                parsed = parse_date_from_text(modified)
                if parsed is not None:
                    document.modified_at = _ensure_utc(parsed)

        comments = self._string_from_metadata_field(data.get("comments"))
        if comments and not document.comments:
            document.comments = comments

        categories = self._parse_topics(data.get("categories"))
        if categories:
            if document.categories:
                self._extend_unique(document.categories, categories)
            else:
                document.categories = categories

        tags = self._parse_topics(data.get("tags"))
        if tags:
            if document.tags:
                self._extend_unique(document.tags, tags)
            else:
                document.tags = tags

        image_url = self._string_from_metadata_field(data.get("image"))
        if image_url and not document.image_url:
            document.image_url = image_url

        page_type = self._string_from_metadata_field(data.get("pagetype"))
        if page_type and not document.page_type:
            document.page_type = page_type

        fingerprint = self._string_from_metadata_field(data.get("fingerprint"))
        if fingerprint and not document.fingerprint:
            document.fingerprint = fingerprint

    def _document_from_metadata(
        self, metadata: Mapping[str, Any], fallback_url: str
    ) -> IrisDocument | None:
        text = self._string_from_metadata_field(
            metadata.get("text") or metadata.get("raw_text")
        )
        summary = self._string_from_metadata_field(metadata.get("description"))
        title = self._string_from_metadata_field(metadata.get("title"))
        language = self._string_from_metadata_field(metadata.get("language"))
        canonical_url = (
            self._string_from_metadata_field(metadata.get("url")) or fallback_url
        )
        comments = self._string_from_metadata_field(metadata.get("comments"))
        image_url = self._string_from_metadata_field(metadata.get("image"))
        page_type = self._string_from_metadata_field(metadata.get("pagetype"))
        fingerprint = self._string_from_metadata_field(metadata.get("fingerprint"))
        tags = self._parse_topics(metadata.get("tags"))
        categories = self._parse_topics(metadata.get("categories"))
        if text or summary or title:
            document = IrisDocument(
                url=canonical_url,
                text=text,
                summary=summary,
                title=title,
                language=language,
                comments=comments,
                image_url=image_url,
                page_type=page_type,
                fingerprint=fingerprint,
            )
            if tags:
                document.tags = tags
            if categories:
                document.categories = categories
            return document
        return None

    def _enrich_document_from_metadata(
        self, document: IrisDocument, metadata: Mapping[str, Any]
    ) -> None:
        canonical_url = self._string_from_metadata_field(metadata.get("url"))
        if canonical_url:
            document.url = canonical_url
        else:
            canonical_source = self._string_from_metadata_field(metadata.get("source"))
            if canonical_source:
                document.url = canonical_source

        if not document.title:
            title = self._string_from_metadata_field(metadata.get("title"))
            if title:
                document.title = title

        if not document.summary:
            summary = self._string_from_metadata_field(metadata.get("description"))
            if summary:
                document.summary = summary

        if not document.text:
            text = self._string_from_metadata_field(
                metadata.get("text") or metadata.get("raw_text")
            )
            if text:
                document.text = text

        if not document.language:
            language = self._string_from_metadata_field(metadata.get("language"))
            if language:
                document.language = language

        authors = self._parse_authors(metadata.get("author"))
        if authors:
            if document.authors:
                self._extend_unique(document.authors, authors)
            else:
                document.authors = authors

        publisher = self._string_from_metadata_field(
            metadata.get("sitename") or metadata.get("source-hostname")
        )
        if publisher and not document.publisher:
            document.publisher = publisher

        organization = self._string_from_metadata_field(metadata.get("hostname"))
        if organization and not document.organization:
            document.organization = organization

        comments = self._string_from_metadata_field(metadata.get("comments"))
        if comments and not document.comments:
            document.comments = comments

        categories = self._parse_topics(metadata.get("categories"))
        if categories:
            if document.categories:
                self._extend_unique(document.categories, categories)
            else:
                document.categories = categories

        tags = self._parse_topics(metadata.get("tags"))
        if tags:
            if document.tags:
                self._extend_unique(document.tags, tags)
            else:
                document.tags = tags

        image_url = self._string_from_metadata_field(metadata.get("image"))
        if image_url and not document.image_url:
            document.image_url = image_url

        page_type = self._string_from_metadata_field(metadata.get("pagetype"))
        if page_type and not document.page_type:
            document.page_type = page_type

        fingerprint = self._string_from_metadata_field(metadata.get("fingerprint"))
        if fingerprint and not document.fingerprint:
            document.fingerprint = fingerprint

        if document.published_at is None:
            published = metadata.get("date")
            if isinstance(published, str):
                parsed = parse_date_from_text(published)
                if parsed is not None:
                    document.published_at = _ensure_utc(parsed)

        if document.modified_at is None:
            modified = metadata.get("filedate")
            if isinstance(modified, str):
                parsed = parse_date_from_text(modified)
                if parsed is not None:
                    document.modified_at = _ensure_utc(parsed)

    def _string_from_metadata_field(self, value: Any) -> str | None:
        if isinstance(value, str):
            return self._normalise_whitespace(value)
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for item in value:
                if isinstance(item, str):
                    normalised = self._normalise_whitespace(item)
                    if normalised:
                        return normalised
        return None

    def _parse_authors(self, value: Any) -> list[str] | None:
        return self._parse_string_list(value, _AUTHOR_SPLIT_RE)

    def _parse_topics(self, value: Any) -> list[str] | None:
        return self._parse_string_list(value, _TOPIC_SPLIT_RE)

    def _parse_string_list(
        self, value: Any, splitter: re.Pattern[str]
    ) -> list[str] | None:
        if not value:
            return None
        candidates: list[str] = []
        if isinstance(value, str):
            candidates = splitter.split(value)
        elif isinstance(value, Sequence) and not isinstance(
            value, (bytes, bytearray, str)
        ):
            for entry in value:
                if isinstance(entry, str):
                    candidates.extend(splitter.split(entry))
        else:
            return None

        results: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalised = self._normalise_whitespace(candidate)
            if not normalised:
                continue
            lowered = normalised.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            results.append(normalised)
        return results or None

    def _extend_unique(self, target: list[str], additions: Sequence[str]) -> None:
        seen = {entry.casefold() for entry in target}
        for addition in additions:
            lowered = addition.casefold()
            if lowered not in seen:
                target.append(addition)
                seen.add(lowered)

    def _fallback_text(self, response: httpx.Response) -> str | None:
        soup = BeautifulSoup(response.text, "html.parser")
        extracted = soup.get_text(" ", strip=True)
        if not extracted:
            return None
        return self._normalise_whitespace(extracted)

    def _normalise_whitespace(self, value: str | None) -> str | None:
        if not value:
            return None
        return " ".join(value.split())

    def _resolve_result_url(self, href: str) -> str | None:
        if not href:
            return None
        parsed = urlparse(href)
        if not parsed.scheme:
            href = f"https://duckduckgo.com{href}" if href.startswith("/") else href
            parsed = urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            query_params = parse_qs(parsed.query)
            uddg_values = query_params.get("uddg")
            if uddg_values:
                return unquote(uddg_values[0])
        if parsed.scheme in {"http", "https"}:
            return href
        return None

    def _select_snippet(
        self, document: IrisDocument, fallback: str | None
    ) -> str | None:
        if document.summary:
            truncated = self._truncate_snippet(document.summary)
            if truncated:
                return truncated
        if document.text:
            for sentence in self._split_sentences(document.text):
                truncated = self._truncate_snippet(sentence)
                if truncated:
                    return truncated
            truncated_text = self._truncate_snippet(document.text)
            if truncated_text:
                return truncated_text
        return self._truncate_snippet(fallback)

    def _split_sentences(self, text: str) -> list[str]:
        normalised = self._normalise_whitespace(text)
        if not normalised:
            return []
        sentences = _SENTENCE_SPLIT_RE.split(normalised)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _truncate_snippet(self, value: str | None) -> str | None:
        normalised = self._normalise_whitespace(value)
        if not normalised:
            return None
        if len(normalised) <= MAX_SNIPPET_LENGTH:
            return normalised
        truncated = normalised[:MAX_SNIPPET_LENGTH]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated.rstrip() + "…"

    async def _get_cached_snippet(self, key: str) -> str | None:
        ttl = self._search_cache_ttl
        if ttl is None:
            return None
        async with self._search_cache_lock:
            cached = self._search_cache.get(key)
            if cached is None:
                return None
            timestamp, snippet = cached
            if monotonic() - timestamp > ttl:
                self._search_cache.pop(key, None)
                return None
            self._search_cache.move_to_end(key)
            return snippet

    async def _store_cached_snippet(self, key: str, snippet: str) -> None:
        ttl = self._search_cache_ttl
        if ttl is None:
            return
        truncated = self._truncate_snippet(snippet)
        if truncated is None:
            return
        async with self._search_cache_lock:
            self._search_cache[key] = (monotonic(), truncated)
            self._search_cache.move_to_end(key)
            while len(self._search_cache) > self._search_cache_size:
                self._search_cache.popitem(last=False)

    async def _get_cached_document(self, key: str) -> IrisDocument | None:
        ttl = self._document_cache_ttl
        if ttl is None:
            return None
        async with self._document_cache_lock:
            cached = self._document_cache.get(key)
            if cached is None:
                return None
            timestamp, document = cached
            if monotonic() - timestamp > ttl:
                self._document_cache.pop(key, None)
                return None
            self._document_cache.move_to_end(key)
            return document

    async def _store_cached_document(self, key: str, document: IrisDocument) -> None:
        ttl = self._document_cache_ttl
        if ttl is None:
            return
        async with self._document_cache_lock:
            self._document_cache[key] = (monotonic(), document)
            self._document_cache.move_to_end(key)
            while len(self._document_cache) > self._document_cache_size:
                self._document_cache.popitem(last=False)

    def _canonicalise_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
        except ValueError:
            return url
        if not parsed.scheme or not parsed.netloc:
            return url
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        query = parsed.query
        return urlunparse((scheme, netloc, path, "", query, ""))
