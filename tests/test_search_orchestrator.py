from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from monGARS.core.search import NormalizedHit, SearchOrchestrator


class AllowRobotsCache:
    async def can_fetch(
        self, user_agent: str, url: str
    ) -> bool:  # pragma: no cover - trivial
        return True


class DummyProvider:
    def __init__(self, hits: list[NormalizedHit]) -> None:
        self._hits = hits

    async def search(
        self, query: str, *, lang: str | None = None, max_results: int = 8
    ):
        return list(self._hits)[:max_results]


class FailingProvider:
    async def search(
        self, query: str, *, lang: str | None = None, max_results: int = 8
    ):
        raise RuntimeError("Provider failure")


class TimeoutProvider:
    async def search(
        self, query: str, *, lang: str | None = None, max_results: int = 8
    ):
        raise asyncio.TimeoutError


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
