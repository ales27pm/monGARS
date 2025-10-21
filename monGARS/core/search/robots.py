"""Robots.txt cache that cooperates with the search orchestrator."""

from __future__ import annotations

import asyncio
import time
import urllib.robotparser
from typing import Dict
from urllib.parse import urlparse

import httpx


class RobotsCache:
    """Cache robots.txt directives for polite crawling."""

    def __init__(self, client: httpx.AsyncClient, ttl_sec: int = 86_400) -> None:
        self.client = client
        self.ttl = ttl_sec
        self._store: Dict[str, tuple[float, urllib.robotparser.RobotFileParser]] = {}
        self._lock = asyncio.Lock()

    async def can_fetch(self, user_agent: str, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        async with self._lock:
            cached = self._store.get(base)
            if cached is None or (time.time() - cached[0]) > self.ttl:
                parser = urllib.robotparser.RobotFileParser()
                robots_url = f"{base}/robots.txt"
                try:
                    response = await self.client.get(robots_url, timeout=5.0)
                except httpx.HTTPError:
                    parser.parse([])
                else:
                    if response.status_code >= 400:
                        parser.parse([])
                    else:
                        parser.parse(response.text.splitlines())
                self._store[base] = (time.time(), parser)
            parser = self._store[base][1]
        try:
            return parser.can_fetch(user_agent, url)
        except Exception:
            return True


__all__ = ["RobotsCache"]
