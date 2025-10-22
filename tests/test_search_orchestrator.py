from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from monGARS.core.search import NormalizedHit, SearchOrchestrator
from monGARS.core.search.providers.ddg import DDGProvider


class AllowRobotsCache:
    async def can_fetch(
        self, user_agent: str, url: str
    ) -> bool:  # pragma: no cover - trivial
        return True


class DummyProvider:
    def __init__(self, hits: list[NormalizedHit]) -> None:
        self._hits = hits

    async def search(
        self, _query: str, *, _lang: str | None = None, _max_results: int = 8
    ):
        return list(self._hits)[:_max_results]


class FailingProvider:
    async def search(
        self, _query: str, *, _lang: str | None = None, _max_results: int = 8
    ):
        raise RuntimeError("Provider failure")


class TimeoutProvider:
    async def search(
        self, _query: str, *, _lang: str | None = None, _max_results: int = 8
    ):
        raise asyncio.TimeoutError


class EmptySearxProvider:
    async def search(
        self, _query: str, *, _lang: str | None = None, _max_results: int = 8
    ) -> list[NormalizedHit]:
        return []


class DummyDDGProvider(DDGProvider):
    def __init__(self, hits: list[NormalizedHit]) -> None:
        self._hits = hits
        self.calls = 0

    async def search(
        self, _query: str, *, _lang: str | None = None, _max_results: int = 8
    ) -> list[NormalizedHit]:
        self.calls += 1
        return list(self._hits)[:_max_results]


class SnippetSearxProvider:
    def __init__(self, hits: list[NormalizedHit]) -> None:
        self._hits = hits

    async def search(
        self, _query: str, *, _lang: str | None = None, _max_results: int = 8
    ) -> list[NormalizedHit]:
        return list(self._hits)[:_max_results]


@pytest.mark.asyncio
async def test_orchestrator_prioritises_trust_and_recency() -> None:
    now = datetime.now(timezone.utc)
    trusted_recent = NormalizedHit(
        provider="gnews",
        title="CDC releases new guidance",
        url="https://www.cdc.gov/update",
        snippet="The CDC announced new health guidance on vaccines.",
        published_at=now - timedelta(days=1),
        event_date=now - timedelta(days=2),
        source_domain="cdc.gov",
        lang="en",
        raw={},
    )
    stale_generic = NormalizedHit(
        provider="ddg",
        title="Blog discusses vaccines",
        url="https://example.com/blog",
        snippet="A blog post about vaccines with older data.",
        published_at=now - timedelta(days=45),
        event_date=None,
        source_domain="example.com",
        lang="en",
        raw={},
    )
    orchestrator = SearchOrchestrator(
        providers=[DummyProvider([stale_generic, trusted_recent])],
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("vaccine guidance", lang="en")

    assert results
    assert results[0].url == trusted_recent.url


@pytest.mark.asyncio
async def test_orchestrator_deduplicates_urls() -> None:
    duplicate_a = NormalizedHit(
        provider="ddg",
        title="News",
        url="https://example.com/article?ref=1",
        snippet="Snippet",
        published_at=None,
        event_date=None,
        source_domain="example.com",
        lang="en",
        raw={},
    )
    duplicate_b = NormalizedHit(
        provider="wikipedia",
        title="News",
        url="https://example.com/article?utm_source=feed",
        snippet="Another snippet",
        published_at=None,
        event_date=None,
        source_domain="example.com",
        lang="en",
        raw={},
    )
    orchestrator = SearchOrchestrator(
        providers=[DummyProvider([duplicate_a, duplicate_b])],
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("news", lang="en")

    assert len(results) == 1
    assert results[0].url.startswith("https://example.com/article")


@pytest.mark.asyncio
async def test_orchestrator_handles_provider_failure() -> None:
    now = datetime.now(timezone.utc)
    healthy_hit = NormalizedHit(
        provider="wikipedia",
        title="Encyclopedic entry",
        url="https://wikipedia.org/wiki/Entry",
        snippet="Authoritative information.",
        published_at=now,
        event_date=None,
        source_domain="wikipedia.org",
        lang="en",
        raw={},
    )
    orchestrator = SearchOrchestrator(
        providers=[FailingProvider(), DummyProvider([healthy_hit])],
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("test", lang="en")

    assert results == [healthy_hit]


@pytest.mark.asyncio
async def test_orchestrator_handles_provider_timeout() -> None:
    orchestrator = SearchOrchestrator(
        providers=[TimeoutProvider(), DummyProvider([])],
        timeout=0.1,
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("timeout", lang="en")

    assert results == []


@pytest.mark.asyncio
async def test_orchestrator_enriches_missing_dates() -> None:
    now = datetime.now(timezone.utc)

    class DummyDocument:
        def __init__(self) -> None:
            self.published_at = now
            self.event_date = now - timedelta(days=1)
            self.language = "en"

    missing_dates = NormalizedHit(
        provider="ddg",
        title="Update",
        url="https://example.org/story",
        snippet="",
        published_at=None,
        event_date=None,
        source_domain="example.org",
        lang=None,
        raw={},
    )

    calls = 0

    async def fetch_document(url: str) -> DummyDocument | None:
        nonlocal calls
        calls += 1
        assert url == missing_dates.url
        return DummyDocument()

    orchestrator = SearchOrchestrator(
        providers=[DummyProvider([missing_dates])],
        robots_cache=AllowRobotsCache(),
        document_fetcher=fetch_document,
    )

    results = await orchestrator.search("story", lang="en")

    assert results
    enriched = results[0]
    assert enriched.published_at == now
    assert enriched.event_date == now - timedelta(days=1)
    assert enriched.lang == "en"
    assert calls == 1


@pytest.mark.asyncio
async def test_orchestrator_uses_ddg_when_searx_returns_nothing() -> None:
    ddg_hit = NormalizedHit(
        provider="ddg",
        title="Fallback result",
        url="https://example.com/fallback",
        snippet="DuckDuckGo snippet",
        published_at=None,
        event_date=None,
        source_domain="example.com",
        lang="en",
        raw={},
    )
    fallback_provider = DummyDDGProvider([ddg_hit])
    orchestrator = SearchOrchestrator(
        providers=[EmptySearxProvider(), fallback_provider],
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("ddg fallback", lang="en")

    assert results
    assert any(hit.provider == "ddg" for hit in results)
    assert fallback_provider.calls == 1


@pytest.mark.asyncio
async def test_orchestrator_skips_ddg_when_searx_snippet_available() -> None:
    now = datetime.now(timezone.utc)
    searx_hit = NormalizedHit(
        provider="searxng",
        title="Primary result",
        url="https://example.org/primary",
        snippet="Authoritative summary",
        published_at=now,
        event_date=None,
        source_domain="example.org",
        lang="en",
        raw={},
    )
    fallback_hit = NormalizedHit(
        provider="ddg",
        title="Fallback",
        url="https://example.com/fallback",
        snippet="",
        published_at=None,
        event_date=None,
        source_domain="example.com",
        lang="en",
        raw={},
    )
    fallback_provider = DummyDDGProvider([fallback_hit])
    orchestrator = SearchOrchestrator(
        providers=[SnippetSearxProvider([searx_hit]), fallback_provider],
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("prefer searx", lang="en")

    assert results
    assert results[0].provider == "searxng"
    assert fallback_provider.calls == 0


@pytest.mark.asyncio
async def test_orchestrator_uses_ddg_when_searx_snippets_blank() -> None:
    searx_blank = NormalizedHit(
        provider="searxng",
        title="Blank",
        url="https://example.org/blank",
        snippet="   ",
        published_at=None,
        event_date=None,
        source_domain="example.org",
        lang="en",
        raw={},
    )
    ddg_hit = NormalizedHit(
        provider="ddg",
        title="DDG",
        url="https://example.com/ddg",
        snippet="ddg snippet",
        published_at=None,
        event_date=None,
        source_domain="example.com",
        lang="en",
        raw={},
    )
    fallback_provider = DummyDDGProvider([ddg_hit])
    orchestrator = SearchOrchestrator(
        providers=[SnippetSearxProvider([searx_blank]), fallback_provider],
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("blank snippets", lang="en")

    assert results
    assert any(hit.provider == "ddg" for hit in results)
    assert fallback_provider.calls == 1
