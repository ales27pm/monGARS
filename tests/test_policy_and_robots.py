
import httpx
import pytest

from monGARS.core.search.policy import DomainPolicy
from monGARS.core.search.robots import RobotsCache


def _robots_transport(text: str, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/robots.txt"):
            return httpx.Response(status, text=text)
        return httpx.Response(200, text="<html></html>")

    return httpx.MockTransport(handler)


async def _can_fetch(robots_txt: str, url: str) -> bool:
    client = httpx.AsyncClient(transport=_robots_transport(robots_txt))
    try:
        rc = RobotsCache(client, ttl_sec=3600)
        return await rc.can_fetch("IrisBot/1.0", url)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_robots_allow_all():
    assert await _can_fetch("", "https://example.com/page") is True


@pytest.mark.asyncio
async def test_robots_disallow_path():
    robots = "User-agent: *\nDisallow: /private\n"
    assert await _can_fetch(robots, "https://example.com/private/thing") is False
    assert await _can_fetch(robots, "https://example.com/public") is True


def test_domain_policy_allow_deny_and_budget(event_loop):
    pol = DomainPolicy(
        allow_patterns=[],  # allow all by default
        deny_patterns=[r"(^|\.)pinterest\.com$"],
        per_host_budget=3,
    )
    assert pol.is_allowed_domain("reuters.com") is True
    assert pol.is_allowed_domain("sub.pinterest.com") is False

    async def burst():
        ok = []
        for _ in range(4):
            ok.append(await pol.acquire_budget("example.com"))
        return ok

    results = event_loop.run_until_complete(burst())
    assert results == [True, True, True, False]
