"""Search orchestrator that fuses multiple providers with policy controls."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)
from urllib.parse import urlsplit

import httpx
from cachetools import TTLCache

from monGARS.config import get_settings

from .contracts import NormalizedHit, SearchProvider
from .policy import DomainPolicy
from .providers import (
    ArxivProvider,
    CrossrefProvider,
    DDGProvider,
    PolitiFactProvider,
    PubMedProvider,
    SearxNGConfig,
    SearxNGProvider,
    SnopesProvider,
    WikipediaProvider,
)
from .robots import RobotsCache

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from monGARS.core.iris import IrisDocument


DocumentFetcher = Callable[[str], Awaitable["IrisDocument | None"]]

try:  # pragma: no cover - optional dependency
    from .providers.gnews import GNewsProvider
except ModuleNotFoundError:  # pragma: no cover - feedparser missing
    GNewsProvider = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

AsyncClientFactory = Callable[[], AsyncIterator[httpx.AsyncClient]]

TRUST_PRIORS: Dict[str, float] = {
    "gnews": 0.35,
    "wikipedia": 0.25,
    "ddg": 0.1,
    "searxng": 0.3,
    "crossref": 0.35,
    "arxiv": 0.3,
    "pubmed": 0.35,
    "politifact": 0.4,
    "snopes": 0.35,
}

TRUST_DOMAINS: Dict[str, float] = {
    "wikipedia.org": 0.15,
    "who.int": 0.4,
    "cdc.gov": 0.45,
    "fda.gov": 0.4,
    "nytimes.com": 0.2,
    "bbc.com": 0.2,
    "reuters.com": 0.25,
    "apnews.com": 0.25,
    "nature.com": 0.35,
    "science.org": 0.35,
    ".gov": 0.4,
    ".edu": 0.35,
    ".gouv": 0.35,
    ".eu": 0.25,
}


def domain_weight(domain: str) -> float:
    domain = domain.lower()
    for suffix, weight in TRUST_DOMAINS.items():
        if suffix.startswith("."):
            if domain.endswith(suffix):
                return weight
        elif domain == suffix or domain.endswith(f".{suffix}"):
            return weight
    return 0.0


def recency_weight(published_at: Optional[datetime], now: datetime) -> float:
    if published_at is None:
        return 0.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    delta_days = (now - published_at).days
    if delta_days <= 1:
        return 0.4
    if delta_days <= 7:
        return 0.3
    if delta_days <= 30:
        return 0.2
    if delta_days <= 180:
        return 0.1
    return 0.02


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


def _dedupe_hits(hits: Sequence[NormalizedHit]) -> List[NormalizedHit]:
    seen: set[str] = set()
    deduped: List[NormalizedHit] = []
    for hit in hits:
        key = re.sub(r"[#?].*$", "", hit.url.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped


class SearchOrchestrator:
    """Coordinate multiple providers and apply policy-aware ranking."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        http_client_factory: Optional[AsyncClientFactory] = None,
        providers: Optional[Sequence[SearchProvider]] = None,
        timeout: float = 6.0,
        cache_ttl_seconds: int = 600,
        cache_max_entries: int = 256,
        policy: DomainPolicy | None = None,
        robots_cache: RobotsCache | None = None,
        robots_user_agent: str = "IrisBot/1.0",
        document_fetcher: DocumentFetcher | None = None,
        enrichment_limit: int = 5,
        document_fetch_timeout: float | None = None,
    ) -> None:
        self._timeout = timeout
        self._providers = list(providers) if providers is not None else None
        self._policy = policy or DomainPolicy(
            allow_patterns=[],
            deny_patterns=[r"(^|\.)pinterest\.com$", r"(^|\.)quora\.com$"],
            per_host_budget=60,
        )
        self._robots_cache = robots_cache
        self._document_fetcher = document_fetcher
        self._enrichment_limit = max(0, enrichment_limit)
        self._document_fetch_timeout = (
            timeout if document_fetch_timeout is None else document_fetch_timeout
        )
        self._searx_warned_missing = False
        if client is not None and http_client_factory is not None:
            raise ValueError(
                "Provide either an httpx.AsyncClient or an http_client_factory, not both."
            )
        if client is not None:
            self._http_client_factory = self._build_singleton_factory(client)
        elif http_client_factory is None:
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
        self._robots_user_agent = robots_user_agent

    @property
    def providers(self) -> Sequence[SearchProvider] | None:
        """Return the configured provider list."""

        return self._providers

    @providers.setter
    def providers(self, value: Sequence[SearchProvider] | None) -> None:
        """Set the provider list, normalising to a concrete list."""

        self._providers = list(value) if value is not None else None

    async def search(
        self, query: str, *, lang: str = "en", max_results: int = 16
    ) -> List[NormalizedHit]:
        normalised_query = query.strip()
        if not normalised_query:
            return []
        cache_key = (normalised_query.lower(), lang, max_results)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            logger.debug(
                "search.orchestrator.cache_hit",
                extra={"query_len": len(normalised_query), "lang": lang},
            )
            return list(cached)

        logger.debug(
            "search.orchestrator.cache_miss",
            extra={"query_len": len(normalised_query), "lang": lang},
        )

        async with self._http_client_factory() as client:
            providers = self._providers
            if providers is None:
                providers = self._build_default_providers(client)
            hits = await self._collect_hits(
                providers, normalised_query, lang=lang, max_results=max_results
            )
            robots_cache = self._robots_cache or RobotsCache(client)
            filtered = await self._filter_hits(hits, robots_cache)
            await self._enrich_hits(filtered)
        ranked = self._rank_hits(filtered, max_results=max_results)
        await self._set_cached(cache_key, tuple(ranked))
        return ranked

    def _build_default_providers(
        self, client: httpx.AsyncClient
    ) -> List[SearchProvider]:
        settings = get_settings()
        providers: List[SearchProvider] = []
        if settings.search_searx_enabled:
            base_url = settings.search_searx_base_url
            if base_url:
                config = SearxNGConfig(
                    base_url=str(base_url),
                    api_key=settings.search_searx_api_key,
                    categories=tuple(settings.search_searx_categories) or None,
                    safesearch=settings.search_searx_safesearch,
                    default_language=settings.search_searx_default_language,
                    max_results=settings.search_searx_result_cap,
                    engines=tuple(settings.search_searx_engines) or None,
                    time_range=settings.search_searx_time_range,
                    sitelimit=settings.search_searx_sitelimit,
                    page_size=settings.search_searx_page_size,
                    max_pages=settings.search_searx_max_pages,
                    language_strict=settings.search_searx_language_strict,
                )
                providers.append(
                    SearxNGProvider(
                        client,
                        config=config,
                        timeout=settings.search_searx_timeout_seconds or self._timeout,
                    )
                )
            else:
                if not self._searx_warned_missing:
                    logger.warning(
                        "search.searxng.missing_base_url",
                        extra={"enabled": True},
                    )
                    self._searx_warned_missing = True
        providers.extend(
            [
                WikipediaProvider(client, timeout=self._timeout),
                CrossrefProvider(client, timeout=self._timeout),
                PubMedProvider(client, timeout=self._timeout),
                ArxivProvider(timeout=self._timeout),
                PolitiFactProvider(),
                SnopesProvider(),
            ]
        )
        providers.append(DDGProvider(client, timeout=self._timeout))
        if GNewsProvider is not None:
            providers.append(GNewsProvider(timeout=self._timeout))
        return providers

    async def _collect_hits(
        self,
        providers: Sequence[SearchProvider],
        query: str,
        *,
        lang: str,
        max_results: int,
    ) -> List[NormalizedHit]:
        if not providers:
            return []
        primary: list[SearchProvider] = []
        fallback: list[SearchProvider] = []
        searx_present = False
        for provider in providers:
            if isinstance(provider, DDGProvider):
                fallback.append(provider)
            else:
                primary.append(provider)
            provider_name = provider.__class__.__name__.lower()
            if isinstance(provider, SearxNGProvider) or "searx" in provider_name:
                searx_present = True

        hits: list[NormalizedHit] = []
        fallback_reason: str | None = None
        if primary:
            primary_hits = await self._collect_from_providers(
                primary, query, lang=lang, max_results=max_results
            )
            hits.extend(primary_hits)
            if searx_present:
                searx_hits = [hit for hit in primary_hits if hit.provider == "searxng"]
                if not searx_hits:
                    fallback_reason = "no_searx_hits"
                elif all(not (hit.snippet or "").strip() for hit in searx_hits):
                    fallback_reason = "blank_searx_snippets"
        else:
            primary_hits = []
            fallback_reason = "no_primary"

        if fallback and fallback_reason:
            logger.debug(
                "search.providers.ddg_fallback",
                extra={"query_len": len(query), "reason": fallback_reason},
            )
            fallback_hits = await self._collect_from_providers(
                fallback, query, lang=lang, max_results=max_results
            )
            hits.extend(fallback_hits)
        return hits

    async def _collect_from_providers(
        self,
        providers: Sequence[SearchProvider],
        query: str,
        *,
        lang: str,
        max_results: int,
    ) -> List[NormalizedHit]:
        task_to_provider: dict[asyncio.Task[Sequence[NormalizedHit]], str] = {}
        for provider in providers:
            kwargs = self._build_provider_kwargs(
                provider, lang=lang, max_results=max_results
            )
            task = asyncio.create_task(provider.search(query, **kwargs))
            task_to_provider[task] = provider.__class__.__name__

        pending = set(task_to_provider.keys())
        hits: list[NormalizedHit] = []
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=self._timeout + 2,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise asyncio.TimeoutError
                for task in done:
                    provider_name = task_to_provider.get(task, "unknown")
                    try:
                        provider_hits = list(task.result())
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
                    except Exception as exc:  # pragma: no cover - provider-specific
                        logger.warning(
                            "search.provider.failure",
                            extra={"provider": provider_name, "error": str(exc)},
                            exc_info=True,
                        )
                        continue
                    if provider_hits:
                        hits.extend(provider_hits)
        except asyncio.TimeoutError:
            logger.warning(
                "search.providers.timeout",
                extra={
                    "timeout_s": self._timeout + 2,
                    "task_count": len(task_to_provider),
                },
            )
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            if task_to_provider:
                await asyncio.gather(*task_to_provider.keys(), return_exceptions=True)
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

    async def _filter_hits(
        self, hits: Sequence[NormalizedHit], robots_cache: RobotsCache
    ) -> List[NormalizedHit]:
        if not hits:
            return []
        filtered: List[NormalizedHit] = []
        for hit in hits:
            domain = hit.source_domain
            if not self._policy.is_allowed_domain(domain):
                continue
            allowed = await self._policy.acquire_budget(domain)
            if not allowed:
                continue
            try:
                can_fetch = await robots_cache.can_fetch(
                    self._robots_user_agent, hit.url
                )
            except Exception:  # pragma: no cover - defensive fallback
                can_fetch = True
            if not can_fetch:
                continue
            filtered.append(hit)
        return _dedupe_hits(filtered)

    async def _enrich_hits(self, hits: Sequence[NormalizedHit]) -> None:
        if not hits or self._document_fetcher is None or self._enrichment_limit <= 0:
            return
        tasks: List[asyncio.Task[None]] = []
        for hit in hits[: self._enrichment_limit]:
            if hit.published_at and hit.event_date and hit.lang:
                continue
            tasks.append(asyncio.create_task(self._enrich_single_hit(hit)))
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.debug(
                    "search.orchestrator.enrich_failed",
                    extra={"error": str(result)},
                )

    async def _enrich_single_hit(self, hit: NormalizedHit) -> None:
        if self._document_fetcher is None:
            return
        try:
            document = await asyncio.wait_for(
                self._document_fetcher(hit.url),
                timeout=self._document_fetch_timeout,
            )
        except asyncio.TimeoutError:
            logger.debug(
                "search.orchestrator.document_fetch_timeout",
                extra={
                    "url": hit.url,
                    "timeout_s": self._document_fetch_timeout,
                },
            )
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "search.orchestrator.document_fetch_failed",
                extra={"error": str(exc), "url": hit.url},
            )
            return
        if document is None:
            return
        published_at = getattr(document, "published_at", None)
        event_date = getattr(document, "event_date", None)
        language = getattr(document, "language", None)
        if hit.published_at is None and isinstance(published_at, datetime):
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            hit.published_at = published_at
        if hit.event_date is None and isinstance(event_date, datetime):
            if event_date.tzinfo is None:
                event_date = event_date.replace(tzinfo=timezone.utc)
            hit.event_date = event_date
        if hit.lang is None and isinstance(language, str) and language:
            hit.lang = language

    def _rank_hits(
        self, hits: Sequence[NormalizedHit], *, max_results: int
    ) -> List[NormalizedHit]:
        now = datetime.now(timezone.utc)
        deduped: Dict[str, NormalizedHit] = {}
        for hit in hits:
            key = _canonical_url(hit.url)
            if key not in deduped:
                deduped[key] = hit
        scored: List[tuple[float, NormalizedHit]] = []
        for hit in deduped.values():
            score = TRUST_PRIORS.get(hit.provider, 0.1)
            score += domain_weight(hit.source_domain)
            score += recency_weight(hit.published_at, now)
            score += _event_bonus(hit.event_date, now=now)
            if (
                hit.event_date
                and hit.published_at
                and hit.event_date <= hit.published_at <= now
            ):
                score += 0.1
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
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                yield client

        return _factory

    @staticmethod
    def _build_singleton_factory(client: httpx.AsyncClient) -> AsyncClientFactory:
        @asynccontextmanager
        async def _factory() -> AsyncIterator[httpx.AsyncClient]:
            yield client

        return _factory


__all__ = ["SearchOrchestrator", "domain_weight", "recency_weight"]
