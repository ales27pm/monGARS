from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import pytest

from monGARS.config import Settings
from monGARS.core.rag.context_enricher import (
    RagContextEnricher,
    RagDisabledError,
    RagServiceError,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
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
        "max_results": 5,
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
