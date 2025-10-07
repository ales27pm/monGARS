from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from monGARS.config import Settings
from monGARS.core.rag.context_enricher import (
    RagContextEnricher,
    RagDisabledError,
    RagServiceError,
)


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        status_code: int = 200,
        *,
        json_exception: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self._json_exception = json_exception

    def json(self) -> Any:
        if self._json_exception is not None:
            raise self._json_exception
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(self.status_code),
            )


class FakeClient:
    def __init__(self, response: FakeResponse, calls: list[dict]) -> None:
        self._response = response
        self.calls = calls

    async def post(
        self, url: str, json: dict, timeout: int | None = None
    ) -> FakeResponse:
        self.calls.append({"url": url, "json": json})
        return self._response


def make_client_factory(response: FakeResponse, calls: list[dict]):
    @asynccontextmanager
    async def factory():
        yield FakeClient(response, calls)

    return factory


@pytest.mark.asyncio
async def test_enrich_returns_structured_payload(monkeypatch):
    settings = Settings(
        rag_enabled=True,
        rag_repo_list=["acme/api"],
        rag_service_url="http://rag.local",
        DOC_RETRIEVAL_URL="http://documents.local",
    )
    monkeypatch.setattr(
        "monGARS.core.rag.context_enricher.get_settings", lambda: settings
    )

    payload = {
        "focusAreas": ["Update validation logic"],
        "references": [
            {
                "repository": "acme/api",
                "file_path": "src/routes.py",
                "summary": "Ensure empty payloads raise 422",
                "score": 0.91,
                "url": "https://example.com/ref",
            }
        ],
    }
    calls: list[dict] = []
    enricher = RagContextEnricher(
        http_client_factory=make_client_factory(FakeResponse(payload), calls)
    )

    result = await enricher.enrich(
        " Review validation edge cases ", repositories=["all"], max_results=5
    )

    assert result.focus_areas == ["Update validation logic"]
    assert len(result.references) == 1
    reference = result.references[0]
    assert reference.repository == "acme/api"
    assert reference.file_path == "src/routes.py"
    assert reference.summary == "Ensure empty payloads raise 422"
    assert reference.score == pytest.approx(0.91)
    assert reference.url == "https://example.com/ref"

    assert calls[0]["json"] == {
        "query": "Review validation edge cases",
        "max_results": 50,
        "repositories": ["all"],
    }
    assert calls[0]["url"] == "http://rag.local/api/rag/context"


@pytest.mark.asyncio
async def test_enrich_raises_when_disabled(monkeypatch):
    settings = Settings(
        rag_enabled=False,
        DOC_RETRIEVAL_URL="http://documents.local",
    )
    monkeypatch.setattr(
        "monGARS.core.rag.context_enricher.get_settings", lambda: settings
    )
    enricher = RagContextEnricher()

    with pytest.raises(RagDisabledError):
        await enricher.enrich("Investigate crash")


@pytest.mark.asyncio
async def test_enrich_raises_service_error_on_http_failure(monkeypatch):
    settings = Settings(
        rag_enabled=True,
        rag_service_url="http://rag.local",
        DOC_RETRIEVAL_URL="http://documents.local",
    )
    monkeypatch.setattr(
        "monGARS.core.rag.context_enricher.get_settings", lambda: settings
    )
    response = FakeResponse({}, status_code=503)
    enricher = RagContextEnricher(http_client_factory=make_client_factory(response, []))

    with pytest.raises(RagServiceError):
        await enricher.enrich("Investigate crash")


@pytest.mark.asyncio
async def test_enrich_returns_empty_on_invalid_json(monkeypatch, caplog):
    settings = Settings(
        rag_enabled=True,
        DOC_RETRIEVAL_URL="http://documents.local",
    )
    monkeypatch.setattr(
        "monGARS.core.rag.context_enricher.get_settings", lambda: settings
    )
    response = FakeResponse(
        "",
        json_exception=json.JSONDecodeError("invalid json", "", 0),
    )
    enricher = RagContextEnricher(http_client_factory=make_client_factory(response, []))

    with caplog.at_level(logging.WARNING):
        result = await enricher.enrich("Investigate crash")

    assert result.focus_areas == []
    assert result.references == []
    assert any(
        record.message == "rag.context_enrichment.invalid_json"
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_enrich_returns_empty_on_non_mapping_payload(monkeypatch, caplog):
    settings = Settings(
        rag_enabled=True,
        DOC_RETRIEVAL_URL="http://documents.local",
    )
    monkeypatch.setattr(
        "monGARS.core.rag.context_enricher.get_settings", lambda: settings
    )
    response = FakeResponse(["unexpected", "list"])
    enricher = RagContextEnricher(http_client_factory=make_client_factory(response, []))

    with caplog.at_level(logging.DEBUG):
        result = await enricher.enrich("Investigate crash")

    assert result.focus_areas == []
    assert result.references == []
    assert any(
        record.message == "rag.context_enrichment.invalid_payload"
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_enrich_rejects_blank_queries(monkeypatch):
    settings = Settings(
        rag_enabled=True,
        DOC_RETRIEVAL_URL="http://documents.local",
    )
    monkeypatch.setattr(
        "monGARS.core.rag.context_enricher.get_settings", lambda: settings
    )
    enricher = RagContextEnricher()

    with pytest.raises(ValueError):
        await enricher.enrich("   ")
