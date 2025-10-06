from __future__ import annotations

import os
import types

os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("SECRET_KEY", "test")

import pytest
from fastapi.testclient import TestClient

from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app
from monGARS.core.rag import (
    RagCodeReference,
    RagDisabledError,
    RagEnrichmentResult,
    RagServiceError,
)

pytestmark = pytest.mark.usefixtures("ensure_test_users")


class DummyRagEnricher:
    def __init__(self) -> None:
        self.last_call: dict | None = None
        self.error: Exception | None = None
        self.result = RagEnrichmentResult(
            focus_areas=["Refactor validation"],
            references=[
                RagCodeReference(
                    repository="acme/api",
                    file_path="src/routes.py",
                    summary="Ensure empty payloads raise 422",
                    score=0.91,
                    url="https://example.com/ref",
                )
            ],
        )

    async def enrich(
        self,
        query: str,
        *,
        repositories: list[str] | None = None,
        max_results: int | None = None,
    ) -> RagEnrichmentResult:
        self.last_call = {
            "query": query,
            "repositories": repositories,
            "max_results": max_results,
        }
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(monkeypatch):
    hippocampus._memory.clear()
    hippocampus._locks.clear()

    dummy_enricher = DummyRagEnricher()

    async def fake_enrich(
        self,
        query: str,
        *,
        repositories: list[str] | None = None,
        max_results: int | None = None,
    ) -> RagEnrichmentResult:
        return await dummy_enricher.enrich(
            query, repositories=repositories, max_results=max_results
        )

    monkeypatch.setattr(
        "monGARS.core.rag.context_enricher.RagContextEnricher.enrich",
        fake_enrich,
    )
    monkeypatch.setattr(
        "monGARS.api.dependencies.get_personality_engine",
        lambda: types.SimpleNamespace(),
    )

    with TestClient(app) as client:
        yield client, dummy_enricher
    hippocampus._memory.clear()
    hippocampus._locks.clear()


@pytest.mark.asyncio
async def test_fetch_rag_context_returns_data(client):
    client_obj, enricher = client
    token = client_obj.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    response = client_obj.post(
        "/api/v1/review/rag-context",
        json={
            "query": "Check validation logic",
            "repositories": ["acme/api"],
            "max_results": 3,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["focus_areas"] == ["Refactor validation"]
    assert data["references"][0]["file_path"] == "src/routes.py"
    assert enricher.last_call == {
        "query": "Check validation logic",
        "repositories": ["acme/api"],
        "max_results": 3,
    }


@pytest.mark.asyncio
async def test_fetch_rag_context_reports_disabled(client):
    client_obj, enricher = client
    enricher.error = RagDisabledError("disabled")
    token = client_obj.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    response = client_obj.post(
        "/api/v1/review/rag-context",
        json={"query": "Investigate"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert data["references"] == []


@pytest.mark.asyncio
async def test_fetch_rag_context_handles_service_error(client):
    client_obj, enricher = client
    enricher.error = RagServiceError("service down")
    token = client_obj.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    response = client_obj.post(
        "/api/v1/review/rag-context",
        json={"query": "Investigate"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_fetch_rag_context_requires_auth(client):
    client_obj, _ = client
    response = client_obj.post(
        "/api/v1/review/rag-context",
        json={"query": "Investigate"},
    )
    assert response.status_code == 401
