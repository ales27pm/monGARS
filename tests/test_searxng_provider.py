from __future__ import annotations

from datetime import timezone
from typing import Any

import httpx
import pytest

from monGARS.core.search.providers.searxng import SearxNGConfig, SearxNGProvider


@pytest.mark.asyncio
async def test_searxng_provider_normalises_payload() -> None:
    captured: dict[str, Any] = {}
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = request.url.params
        captured_headers.update(request.headers)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Open source intelligence",
                        "url": "https://example.org/osint",
                        "content": "Overview of open data collection techniques.",
                        "publishedDate": "2024-05-01T10:30:00Z",
                        "language": None,
                    },
                    {
                        "title": "Network diagnostics with SearxNG",
                        "url": "https://example.com/diagnostics",
                        "description": "How to use SearxNG for research.",
                        "published": "Wed, 01 May 2024 11:15:00 GMT",
                    },
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://searx.local"
    ) as client:
        provider = SearxNGProvider(
            client,
            config=SearxNGConfig(
                base_url="https://searx.local",
                api_key="Bearer test-token",
                categories=("general", "news"),
                safesearch=1,
                default_language="en",
                max_results=10,
                engines=("google", "bing"),
                time_range="day",
                sitelimit="site:example.com",
                page_size=7,
                language_strict=True,
            ),
            timeout=2.0,
        )
        results = await provider.search("open source", lang="fr", max_results=10)

    assert captured["query"]["q"] == "open source"
    assert captured["query"]["language"] == "fr"
    assert captured["query"]["categories"] == "general,news"
    assert captured["query"]["safesearch"] == "1"
    assert captured["query"]["engines"] == "google,bing"
    assert captured["query"]["time_range"] == "day"
    assert captured["query"]["sitelimit"] == "site:example.com"
    assert captured["query"]["count"] == "7"
    assert captured["query"]["pageno"] == "1"
    assert captured_headers["accept-language"] == "fr"
    assert captured_headers["authorization"] == "Bearer test-token"

    assert len(results) == 2
    first, second = results

    assert first.provider == "searxng"
    assert first.url == "https://example.org/osint"
    assert first.lang == "fr"
    assert first.published_at is not None
    assert first.published_at.tzinfo == timezone.utc
    assert "open data" in first.snippet

    assert second.published_at is not None
    assert second.published_at.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_searxng_provider_respects_result_cap() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "https://example.com/a"},
                    {"title": "B", "url": "https://example.com/b"},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://searx.local"
    ) as client:
        provider = SearxNGProvider(
            client,
            config=SearxNGConfig(
                base_url="https://searx.local",
                max_results=1,
                max_pages=1,
            ),
            timeout=2.0,
        )
        results = await provider.search("searx", max_results=5)

    assert len(results) == 1
    assert results[0].title == "A"


@pytest.mark.asyncio
async def test_searxng_provider_fetches_multiple_pages() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        payloads = [
            {
                "results": [
                    {
                        "title": "First",
                        "url": "https://example.com/first",
                        "content": "one",
                    }
                ]
            },
            {
                "results": [
                    {
                        "title": "Second",
                        "url": "https://example.com/second",
                        "content": "two",
                    }
                ]
            },
        ]
        index = len(calls) - 1
        return httpx.Response(200, json=payloads[index])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://searx.local"
    ) as client:
        provider = SearxNGProvider(
            client,
            config=SearxNGConfig(
                base_url="https://searx.local",
                max_results=5,
                max_pages=2,
                page_size=1,
            ),
            timeout=2.0,
        )
        results = await provider.search("deep dive", max_results=2)

    assert len(results) == 2
    assert {hit.title for hit in results} == {"First", "Second"}
    assert calls[0]["pageno"] == "1"
    assert calls[1]["pageno"] == "2"


@pytest.mark.asyncio
async def test_searxng_provider_promotes_infobox_urls() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [],
                "infoboxes": [
                    {
                        "infobox": "IETF",
                        "urls": [{"url": "https://www.ietf.org"}],
                        "content": "Internet standards development organisation.",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://searx.local"
    ) as client:
        provider = SearxNGProvider(
            client,
            config=SearxNGConfig(
                base_url="https://searx.local", max_results=3, max_pages=1
            ),
            timeout=2.0,
        )
        results = await provider.search("ietf", max_results=1)

    assert len(results) == 1
    hit = results[0]
    assert hit.url == "https://www.ietf.org"
    assert hit.snippet.startswith("Internet standards")
    assert hit.raw.get("infobox") is True
