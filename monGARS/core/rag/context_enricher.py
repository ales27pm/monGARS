"""RAG context enrichment service integration."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from monGARS.config import get_settings

log = logging.getLogger(__name__)

AsyncClientFactory = Callable[[], AsyncIterator[httpx.AsyncClient]]


@dataclass(slots=True)
class RagCodeReference:
    """Represents a single code reference returned by the RAG service."""

    repository: str
    file_path: str
    summary: str
    score: float | None = None
    url: str | None = None


@dataclass(slots=True)
class RagEnrichmentResult:
    """Structured payload returned by :class:`RagContextEnricher`."""

    focus_areas: list[str]
    references: list[RagCodeReference]


class RagDisabledError(RuntimeError):
    """Raised when RAG context enrichment is disabled via configuration."""


class RagServiceError(RuntimeError):
    """Raised when the upstream RAG service fails to respond successfully."""


class RagContextEnricher:
    """Client for the external RAG context enrichment service."""

    def __init__(
        self,
        *,
        http_client_factory: AsyncClientFactory | None = None,
    ) -> None:
        self._settings = get_settings()
        if http_client_factory is None:
            timeout = httpx.Timeout(10.0, connect=5.0)

            @asynccontextmanager
            async def default_http_client_factory() -> AsyncIterator[httpx.AsyncClient]:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    yield client

            self._http_client_factory = default_http_client_factory
        else:
            self._http_client_factory = http_client_factory

    async def enrich(
        self,
        query: str,
        *,
        repositories: Sequence[str] | None = None,
        max_results: int | None = None,
    ) -> RagEnrichmentResult:
        """Return focus areas and references relevant to *query*.

        Parameters
        ----------
        query:
            Natural language description of the pull request or task requiring
            additional context.
        repositories:
            Optional override for the repositories considered during retrieval.
        max_results:
            Optional override for the number of references requested.
        """

        if not bool(getattr(self._settings, "rag_enabled", False)):
            raise RagDisabledError("RAG context enrichment is disabled in settings.")

        trimmed_query = query.strip()
        if not trimmed_query:
            raise ValueError("query cannot be empty")

        payload: dict[str, Any] = {
            "query": trimmed_query,
            "max_results": self._normalise_limit(max_results),
        }
        resolved_repositories = self._resolve_repositories(repositories)
        if resolved_repositories:
            payload["repositories"] = resolved_repositories

        endpoint = f"{self._service_base_url()}/api/rag/context"
        try:
            async with self._http_client_factory() as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - depends on service
            log.warning(
                "rag.context_enrichment.status_error",
                extra={
                    "status_code": exc.response.status_code,
                    "endpoint": endpoint,
                },
            )
            raise RagServiceError("RAG service returned an error response.") from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            log.warning(
                "rag.context_enrichment.transport_error",
                extra={"error": str(exc)},
            )
            raise RagServiceError("Failed to contact RAG service.") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            log.warning(
                "rag.context_enrichment.invalid_json",
                extra={"error": str(exc)},
            )
            return RagEnrichmentResult(focus_areas=[], references=[])
        if not isinstance(data, Mapping):
            log.debug(
                "rag.context_enrichment.invalid_payload",
                extra={"payload_type": type(data).__name__},
            )
            return RagEnrichmentResult(focus_areas=[], references=[])

        focus_areas = self._extract_focus_areas(data)
        references = self._extract_references(data.get("references"))
        return RagEnrichmentResult(focus_areas=focus_areas, references=references)

    def _service_base_url(self) -> str:
        base = getattr(self._settings, "rag_service_url", None) or getattr(
            self._settings, "DOC_RETRIEVAL_URL", ""
        )
        if not isinstance(base, str):
            base = str(base)
        return base.rstrip("/")

    def _resolve_repositories(
        self, overrides: Sequence[str] | None
    ) -> list[str] | None:
        source: Sequence[str] | None
        if overrides is not None:
            source = overrides
        else:
            source = getattr(self._settings, "rag_repo_list", None)
        if not source:
            return None
        cleaned: list[str] = []
        wildcard = False
        for item in source:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value:
                continue
            if value.lower() == "all":
                wildcard = True
                break
            if value not in cleaned:
                cleaned.append(value)
        if wildcard:
            return ["all"]
        return cleaned or None

    def _normalise_limit(self, requested: int | None) -> int:
        configured = getattr(self._settings, "rag_max_results", 5)
        try:
            configured_limit = int(configured)
        except (TypeError, ValueError):
            configured_limit = 5
        if configured_limit <= 0:
            configured_limit = 5
        if requested is None:
            return configured_limit
        return max(1, min(requested, configured_limit))

    def _extract_focus_areas(self, payload: Mapping[str, Any]) -> list[str]:
        raw = payload.get("focus_areas") or payload.get("focusAreas")
        return self._clean_string_list(raw)

    def _extract_references(self, payload: Any) -> list[RagCodeReference]:
        if not isinstance(payload, Sequence):
            return []
        references: list[RagCodeReference] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            file_path = self._extract_string(
                item, ("file_path", "filePath", "path"), required=True
            )
            if file_path is None:
                continue
            repository = self._extract_string(
                item, ("repository", "repo", "project"), default="unknown"
            )
            summary = self._extract_string(
                item, ("summary", "description", "snippet"), default=file_path
            )
            score_value = item.get("score")
            score = None
            if isinstance(score_value, (int, float)):
                score = float(score_value)
            url = self._extract_string(item, ("url", "link"))
            references.append(
                RagCodeReference(
                    repository=repository or "unknown",
                    file_path=file_path,
                    summary=summary or file_path,
                    score=score,
                    url=url,
                )
            )
        return references

    def _clean_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
            return []
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            trimmed = item.strip()
            if trimmed:
                cleaned.append(trimmed)
        return cleaned

    def _extract_string(
        self,
        payload: Mapping[str, Any],
        keys: Sequence[str],
        *,
        required: bool = False,
        default: str | None = None,
    ) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed
        if required:
            return None
        if default is not None:
            return default
        return None
