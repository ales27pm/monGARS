"""Utilities for lightweight web retrieval used in the curiosity engine."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS: Mapping[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(slots=True)
class IrisDocument:
    """Structured representation of extracted web content."""

    url: str
    text: str | None
    title: str | None = None
    summary: str | None = None
    language: str | None = None


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

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout = httpx.Timeout(request_timeout)
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._max_content_length = max_content_length
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

    async def fetch_document(self, url: str) -> IrisDocument | None:
        """Fetch structured content for ``url`` using trafilatura extraction."""

        response = await self._get_response(url)
        if response is None:
            return None
        document = self._extract_document(response)
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
                    return best_candidate

        return snippet_text or None

    async def _request_with_retries(
        self, method: str, url: str, **kwargs: object
    ) -> httpx.Response | None:
        """Execute an HTTP request with bounded retries."""

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(**self._client_options) as client:
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

    def _extract_document(self, response: httpx.Response) -> IrisDocument | None:
        extracted_json: str | None = None
        try:
            extracted_json = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                output_format="json",
                url=str(response.request.url),
            )
        except Exception as exc:  # pragma: no cover - library edge case
            logger.debug(
                "iris.fetch_text.trafilatura_failed",
                extra={"error": str(exc)},
            )

        document_data: Mapping[str, object] | None = None
        if extracted_json:
            try:
                document_data = json.loads(extracted_json)
            except json.JSONDecodeError as exc:  # pragma: no cover - unexpected format
                logger.debug(
                    "iris.fetch_text.invalid_trafilatura_payload",
                    extra={"error": str(exc)},
                )

        fallback_text = None
        if document_data:
            text = document_data.get("text")
            summary = document_data.get("summary")
            title = document_data.get("title")
            language = document_data.get("language")
            cleaned_text = (
                self._normalise_whitespace(text) if isinstance(text, str) else None
            )
            cleaned_summary = (
                self._normalise_whitespace(summary)
                if isinstance(summary, str)
                else None
            )
            if cleaned_text or cleaned_summary or isinstance(title, str):
                return IrisDocument(
                    url=str(response.request.url),
                    text=cleaned_text,
                    summary=cleaned_summary,
                    title=title if isinstance(title, str) else None,
                    language=language if isinstance(language, str) else None,
                )
            fallback_text = self._fallback_text(response)
        else:
            fallback_text = self._fallback_text(response)

        if fallback_text:
            return IrisDocument(
                url=str(response.request.url),
                text=fallback_text,
                summary=None,
                title=None,
                language=None,
            )

        return None

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
            return document.summary
        if document.text:
            return document.text[:500]
        return fallback
