from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from monGARS.core.search import NormalizedHit
from monGARS.core.search.metadata import parse_date_from_text, parse_schema_dates
from monGARS.core.search.orchestrator import SearchOrchestrator
from monGARS.core.search.policy import DomainPolicy
from monGARS.core.search.providers.arxiv import ArxivProvider
from monGARS.core.search.providers.crossref import CrossrefProvider
from monGARS.core.search.providers.factcheckers import (
    PolitiFactProvider,
    SnopesProvider,
)
from monGARS.core.search.providers.pubmed import PubMedProvider


class DummyAsyncClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses

    async def get(self, url: str, *args, **kwargs) -> httpx.Response:
        if not self._responses:
            raise AssertionError("Unexpected HTTP call")
        return self._responses.pop(0)


class AllowRobotsCache:
    async def can_fetch(
        self, user_agent: str, url: str
    ) -> bool:  # pragma: no cover - trivial
        return True


class BlockRobotsCache:
    def __init__(self, blocked: set[str]) -> None:
        self._blocked = blocked

    async def can_fetch(self, user_agent: str, url: str) -> bool:
        return url not in self._blocked


class StaticProvider:
    def __init__(self, hits: list[NormalizedHit]) -> None:
        self._hits = hits

    async def search(self, query: str, **_: object) -> list[NormalizedHit]:
        return list(self._hits)


def make_response(url: str, data: dict) -> httpx.Response:
    request = httpx.Request("GET", url)
    return httpx.Response(200, request=request, json=data)


@pytest.mark.asyncio
async def test_crossref_provider_returns_published_date() -> None:
    response = make_response(
        CrossrefProvider.BASE_URL,
        {
            "message": {
                "items": [
                    {
                        "URL": "https://doi.org/10.1000/test",
                        "title": ["Sample Paper"],
                        "issued": {"date-parts": [[2024, 1, 15]]},
                    }
                ]
            }
        },
    )
    client = DummyAsyncClient([response])
    provider = CrossrefProvider(client)

    hits = await provider.search("sample", max_results=1)

    assert hits
    assert hits[0].published_at is not None


@pytest.mark.asyncio
async def test_arxiv_provider_uses_latest_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("feedparser")
    updated = SimpleNamespace(
        title="Latest",
        link="https://arxiv.org/abs/2",
        summary="",
        updated_parsed=(2024, 1, 2, 0, 0, 0),
    )
    older = SimpleNamespace(
        title="Older",
        link="https://arxiv.org/abs/1",
        summary="",
        updated_parsed=(2023, 12, 25, 0, 0, 0),
    )
    feed = SimpleNamespace(entries=[updated, older])

    monkeypatch.setattr(
        "monGARS.core.search.providers.arxiv.feedparser.parse", lambda *_: feed
    )

    provider = ArxivProvider()
    hits = await provider.search("quantum", max_results=2)

    assert hits[0].published_at is not None
    assert hits[0].published_at.year == 2024


@pytest.mark.asyncio
async def test_pubmed_provider_constructs_results() -> None:
    esearch = make_response(
        PubMedProvider.ESEARCH_URL,
        {"esearchresult": {"idlist": ["12345"]}},
    )
    esummary = make_response(
        PubMedProvider.ESUMMARY_URL,
        {
            "result": {
                "12345": {
                    "title": "Clinical Study",
                    "pubdate": "2024 Oct 12",
                    "sortfirstauthor": "Doe",
                    "source": "Medical Journal",
                }
            }
        },
    )
    client = DummyAsyncClient([esearch, esummary])
    provider = PubMedProvider(client)
    hits = await provider.search("immunology", max_results=1)

    assert hits
    assert hits[0].url.endswith("12345/")
    assert hits[0].published_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_cls", [PolitiFactProvider, SnopesProvider])
async def test_factcheck_providers_include_verdict(provider_cls: type) -> None:
    pytest.importorskip("feedparser")
    entry = SimpleNamespace(
        title="Claim",
        summary="This claim is mostly true",
        link="https://example.com/fact",
        published_parsed=(2024, 1, 1, 0, 0, 0),
    )
    feed = SimpleNamespace(entries=[entry])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            f"monGARS.core.search.providers.factcheckers.feedparser.parse",
            lambda *_: feed,
        )
        provider = provider_cls()
        hits = await provider.search("mostly", max_results=1)

    assert hits
    assert "true" in hits[0].snippet.lower()


def test_domain_policy_denied_domain() -> None:
    policy = DomainPolicy(deny_patterns=[r"example\.com$"])
    assert not policy.is_allowed_domain("news.example.com")


@pytest.mark.asyncio
async def test_domain_policy_budget_depletion() -> None:
    now = datetime.now(timezone.utc)
    hits = [
        NormalizedHit(
            provider="ddg",
            title="First",
            url=f"https://example.org/a{i}",
            snippet="",
            published_at=now,
            event_date=None,
            source_domain="example.org",
            lang="en",
            raw={},
        )
        for i in range(2)
    ]
    policy = DomainPolicy(per_host_budget=1)
    orchestrator = SearchOrchestrator(
        providers=[StaticProvider(hits)],
        policy=policy,
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("budget test")

    assert len(results) == 1


@pytest.mark.asyncio
async def test_robots_disallow_filters_hits() -> None:
    now = datetime.now(timezone.utc)
    hit = NormalizedHit(
        provider="ddg",
        title="Blocked",
        url="https://blocked.example.com/article",
        snippet="",
        published_at=now,
        event_date=None,
        source_domain="blocked.example.com",
        lang="en",
        raw={},
    )
    orchestrator = SearchOrchestrator(
        providers=[StaticProvider([hit])],
        robots_cache=BlockRobotsCache({hit.url}),
    )

    results = await orchestrator.search("blocked")

    assert results == []


def test_metadata_parsing_json_ld() -> None:
    html = """
    <html><head>
    <script type="application/ld+json">{"datePublished":"2024-01-05","startDate":"2024-01-04"}</script>
    </head><body></body></html>
    """
    event_dt, pub_dt = parse_schema_dates(html)
    assert event_dt and pub_dt
    assert event_dt <= pub_dt


def test_metadata_parsing_opengraph_and_fallback() -> None:
    html = """
    <html><head>
        <meta property="article:published_time" content="2024-02-01T00:00:00Z" />
    </head><body>Updated Feb 2, 2024 in text.</body></html>
    """
    event_dt, pub_dt = parse_schema_dates(html)
    assert pub_dt and pub_dt.year == 2024
    fallback = parse_date_from_text("Updated Feb 2, 2024")
    assert fallback and fallback.year == 2024


@pytest.mark.asyncio
async def test_ranking_prefers_fresh_trusted_sources() -> None:
    now = datetime.now(timezone.utc)
    fresh_trusted = NormalizedHit(
        provider="gnews",
        title="Trusted",
        url="https://cdc.gov/update",
        snippet="",
        published_at=now - timedelta(hours=6),
        event_date=None,
        source_domain="cdc.gov",
        lang="en",
        raw={},
    )
    stale_generic = NormalizedHit(
        provider="ddg",
        title="Old",
        url="https://example.net/post",
        snippet="",
        published_at=now - timedelta(days=40),
        event_date=None,
        source_domain="example.net",
        lang="en",
        raw={},
    )
    orchestrator = SearchOrchestrator(
        providers=[StaticProvider([stale_generic, fresh_trusted])],
        robots_cache=AllowRobotsCache(),
    )

    results = await orchestrator.search("ranking")

    assert results and results[0].url == fresh_trusted.url
