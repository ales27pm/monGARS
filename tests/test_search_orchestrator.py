from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from monGARS.core.search import NormalizedHit, SearchOrchestrator


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
        providers=[DummyProvider([stale_generic, trusted_recent])]
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
        providers=[DummyProvider([duplicate_a, duplicate_b])]
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
        providers=[FailingProvider(), DummyProvider([healthy_hit])]
    )

    results = await orchestrator.search("test", lang="en")

    assert results == [healthy_hit]


@pytest.mark.asyncio
async def test_orchestrator_handles_provider_timeout() -> None:
    orchestrator = SearchOrchestrator(
        providers=[TimeoutProvider(), DummyProvider([])],
        timeout=0.1,
    )

    results = await orchestrator.search("timeout", lang="en")

    assert results == []
