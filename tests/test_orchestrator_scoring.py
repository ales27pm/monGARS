from datetime import datetime, timedelta, timezone

import httpx
import pytest

from monGARS.core.search.contracts import NormalizedHit
from monGARS.core.search.orchestrator import (
    SearchOrchestrator,
    domain_weight,
    recency_weight,
)


class _StubProviderFreshGov:
    async def search(self, query: str, lang="en", max_results: int = 8):
        now = datetime.now(timezone.utc)
        return [
            NormalizedHit(
                provider="gnews",
                title="Gov update",
                url="https://cdc.gov/notice",
                snippet="update",
                published_at=now - timedelta(hours=2),
                event_date=now - timedelta(hours=3),
                source_domain="cdc.gov",
                lang="en",
                raw={},
            )
        ]


class _StubProviderOldBlog:
    async def search(self, query: str, lang="en", max_results: int = 8):
        old = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return [
            NormalizedHit(
                provider="ddg",
                title="Old blog post",
                url="https://myblog.medium.com/post",
                snippet="opinion",
                published_at=old,
                event_date=None,
                source_domain="medium.com",
                lang="en",
                raw={},
            )
        ]


class _TestOrchestrator(SearchOrchestrator):
    def __init__(self, client: httpx.AsyncClient):
        super().__init__(client)
        self.providers = [_StubProviderFreshGov(), _StubProviderOldBlog()]


async def _search():
    async with httpx.AsyncClient() as client:
        orch = _TestOrchestrator(client)
        hits = await orch.search("query", lang="en", max_results=4)
        return hits


@pytest.mark.asyncio
async def test_scoring_prefers_fresh_trustworthy():
    hits = await _search()
    assert hits
    assert hits[0].source_domain.endswith(".gov")


def test_domain_and_recency_weights():
    assert domain_weight("cdc.gov") > domain_weight("medium.com")
    now = datetime.now(timezone.utc)
    assert recency_weight(now - timedelta(hours=1), now) > recency_weight(
        now - timedelta(days=400), now
    )
