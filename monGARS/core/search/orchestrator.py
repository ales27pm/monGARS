"""Search orchestrator that fuses multiple providers."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx
from cachetools import TTLCache

from .contracts import NormalizedHit, SearchProvider
from .providers.ddg import DDGProvider
from .providers.gnews import GNewsProvider
from .providers.wikipedia import WikipediaProvider

logger = logging.getLogger(__name__)

AsyncClientFactory = Callable[[], AsyncIterator[httpx.AsyncClient]]


TRUSTED_DOMAIN_WEIGHTS: Dict[str, float] = {
    "wikipedia.org": 0.15,
    "who.int": 0.35,
    "cdc.gov": 0.45,
    "fda.gov": 0.4,
    "nytimes.com": 0.2,
    "bbc.com": 0.2,
    "bbc.co.uk": 0.2,
    "reuters.com": 0.25,
    "apnews.com": 0.25,
    "nature.com": 0.35,
    "science.org": 0.35,
}

PROVIDER_PRIORS: Dict[str, float] = {
    "gnews": 0.35,
    "wikipedia": 0.25,
    "ddg": 0.15,
}


def _domain_weight(domain: str) -> float:
    domain = domain.lower()
    for suffix in (".gov", ".edu", ".gouv", ".gouv.fr"):
        if domain.endswith(suffix):
            return 0.4 if suffix == ".gov" else 0.35
    return next(
        (
            weight
            for key, weight in TRUSTED_DOMAIN_WEIGHTS.items()
            if domain == key or domain.endswith(f".{key}")
        ),
        0.0,
    )


def _recency_weight(published_at: Optional[datetime], *, now: datetime) -> float:
    if published_at is None:
        return 0.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    delta = now - published_at
    days = delta.days
    if days <= 1:
        return 0.4
    if days <= 7:
        return 0.3
    if days <= 30:
        return 0.2
    return 0.1 if days <= 180 else 0.02


def _event_bonus(event_date: Optional[datetime], *, now: datetime) -> float:
    if event_date is None:
        return 0.0
    if event_date.tzinfo is None:
        event_date = event_date.replace(tzinfo=timezone.utc)
    if now < event_date:
        return 0.0
    if (now - event_date).days <= 7:
        return 0.08
    return 0.04 if (now - event_date).days <= 30 else 0.0


def _canonical_url(url: str) -> str:
    pieces = urlsplit(url)
    scheme = pieces.scheme.lower()
    host = pieces.netloc.lower()
    return f"{scheme}://{host}{pieces.path}"


class SearchOrchestrator:
    """Coordinate multiple providers and apply a trust-aware ranking."""

    def __init__(
        self,
        *,
        http_client_factory: Optional[AsyncClientFactory] = None,
        providers: Optional[Sequence[SearchProvider]] = None,
        timeout: float = 6.0,
        cache_ttl_seconds: int = 600,
        cache_max_entries: int = 256,
    ) -> None:
        self._timeout = timeout
        self._providers = list(providers) if providers is not None else None
        if http_client_factory is None:
            self._http_client_factory = self._build_default_client_factory(timeout)
        else:
            self._http_client_factory = http_client_factory
        self._cache: TTLCache[Tuple[str, str, int], Tuple[NormalizedHit, ...]] = (
            TTLCache(
                maxsize=cache_max_entries,
                ttl=cache_ttl_seconds,
            )
        )
        self._cache_lock = asyncio.Lock()

    async def search(
        self, query: str, *, lang: str = "en", max_results: int = 16
    ) -> List[NormalizedHit]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        cache_key = (normalized_query.lower(), lang, max_results)
        cached_hits = await self._get_cached(cache_key)
        if cached_hits is not None:
            logger.debug(
                "search.orchestrator.cache_hit",
                extra={"query_len": len(normalized_query), "lang": lang},
            )
            return list(cached_hits)

        logger.debug(
            "search.orchestrator.cache_miss",
            extra={"query_len": len(normalized_query), "lang": lang},
        )

        hits = await self._gather_hits(
            normalized_query, lang=lang, max_results=max_results
        )
        ranked = self._rank_hits(hits, max_results=max_results)
        await self._set_cached(cache_key, tuple(ranked))
        return ranked

    async def _gather_hits(
        self, query: str, *, lang: str, max_results: int
    ) -> List[NormalizedHit]:
        providers = self._providers
        if providers is not None:
            return await self._collect_from_providers(
                providers, query, lang=lang, max_results=max_results
            )

        async with self._http_client_factory() as client:
            built_providers: list[SearchProvider] = [
                DDGProvider(client, timeout=self._timeout),
                WikipediaProvider(client, timeout=self._timeout),
                GNewsProvider(timeout=self._timeout),
            ]
            return await self._collect_from_providers(
                built_providers, query, lang=lang, max_results=max_results
            )

    async def _collect_from_providers(
        self,
        providers: Sequence[SearchProvider],
        query: str,
        *,
        lang: str,
        max_results: int,
    ) -> List[NormalizedHit]:
        tasks: dict[asyncio.Task[Sequence[NormalizedHit]], str] = {}
        for provider in providers:
            kwargs = self._build_provider_kwargs(
                provider, lang=lang, max_results=max_results
            )
            task = asyncio.create_task(provider.search(query, **kwargs))
            tasks[task] = provider.__class__.__name__

        hits: list[NormalizedHit] = []
        try:
            for task in asyncio.as_completed(tasks, timeout=self._timeout + 2):
                provider_name = tasks[task]
                try:
                    provider_hits = await task
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    logger.warning(
                        "search.provider.timeout",
                        extra={
                            "provider": provider_name,
                            "timeout_s": self._timeout,
                        },
                    )
                    continue
                except httpx.TimeoutException as exc:
                    logger.warning(
                        "search.provider.http_timeout",
                        extra={"provider": provider_name, "error": str(exc)},
                        exc_info=True,
                    )
                    continue
                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "search.provider.http_error",
                        extra={
                            "provider": provider_name,
                            "status_code": exc.response.status_code,
                        },
                        exc_info=True,
                    )
                    continue
                except httpx.HTTPError as exc:
                    logger.warning(
                        "search.provider.network_error",
                        extra={"provider": provider_name, "error": str(exc)},
                        exc_info=True,
                    )
                    continue
                except Exception as exc:  # pragma: no cover - provider-dependent
                    logger.warning(
                        "search.provider.failure",
                        extra={"provider": provider_name, "error": str(exc)},
                        exc_info=True,
                    )
                    continue
                hits.extend(provider_hits)
        except asyncio.TimeoutError:
            logger.warning(
                "search.providers.timeout",
                extra={"timeout_s": self._timeout + 2, "task_count": len(tasks)},
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks.keys(), return_exceptions=True)
        return hits

    def _build_provider_kwargs(
        self, provider: SearchProvider, *, lang: str, max_results: int
    ) -> Dict[str, object]:
        kwargs: Dict[str, object] = {}
        signature = inspect.signature(provider.search)
        if "lang" in signature.parameters:
            kwargs["lang"] = lang
        if "max_results" in signature.parameters:
            kwargs["max_results"] = max_results
        return kwargs

    def _rank_hits(
        self, hits: Sequence[NormalizedHit], *, max_results: int
    ) -> List[NormalizedHit]:
        now = datetime.now(timezone.utc)
        deduped: Dict[str, NormalizedHit] = {}
        for hit in hits:
            key = _canonical_url(hit.url)
            if key not in deduped:
                deduped[key] = hit
        scored: list[tuple[float, NormalizedHit]] = []
        for hit in deduped.values():
            score = PROVIDER_PRIORS.get(hit.provider, 0.1)
            score += _domain_weight(hit.source_domain)
            score += _recency_weight(hit.published_at, now=now)
            score += _event_bonus(hit.event_date, now=now)
            if hit.is_trustworthy():
                score += 0.05
            scored.append((score, hit))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in scored[:max_results]]

    async def _get_cached(
        self, key: Tuple[str, str, int]
    ) -> Optional[Tuple[NormalizedHit, ...]]:
        async with self._cache_lock:
            return self._cache.get(key)

    async def _set_cached(
        self, key: Tuple[str, str, int], value: Tuple[NormalizedHit, ...]
    ) -> None:
        async with self._cache_lock:
            self._cache[key] = value

    @staticmethod
    def _build_default_client_factory(timeout: float) -> AsyncClientFactory:
        @asynccontextmanager
        async def _factory() -> AsyncIterator[httpx.AsyncClient]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                yield client

        return _factory
