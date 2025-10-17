"""Integration tests exercising the chat and embedding inference pipeline."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
from fastapi.testclient import TestClient

from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app as main_app
from monGARS.api.web_api import get_conversational_module
from scripts import run_llm2vec_service

UTC = getattr(datetime, "UTC", timezone.utc)


@pytest.fixture(autouse=True, scope="module")
def _setup_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure test environment variables for all tests in this module."""
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("SECRET_KEY", "test")

UTC = getattr(datetime, "UTC", timezone.utc)


@pytest.fixture
def chat_test_client() -> TestClient:
    """Return a FastAPI client with a deterministic conversational module."""

    hippocampus._memory.clear()
    hippocampus._locks.clear()

    class StubConversationalModule:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def generate_response(
            self,
            user_id: str,
            query: str,
            *,
            session_id: str | None = None,
            image_data: bytes | None = None,
        ) -> dict[str, Any]:
            payload_text = f"echo::{query}"
            self.calls.append(
                {
                    "user_id": user_id,
                    "query": query,
                    "session_id": session_id,
                    "image_data": image_data,
                }
            )
            created_at = datetime.now(UTC).isoformat()
            return {
                "text": payload_text,
                "confidence": 0.88,
                "processing_time": 0.42,
                "speech_turn": {
                    "turn_id": "turn-001",
                    "text": payload_text,
                    "created_at": created_at,
                    "segments": [
                        {
                            "text": payload_text,
                            "estimated_duration": 0.5,
                            "pause_after": 0.1,
                        }
                    ],
                    "average_words_per_second": 2.5,
                    "tempo": 1.0,
                },
            }

    stub_module = StubConversationalModule()

    def override_get_conversational_module(
        personality: Any = None, dynamic: Any = None
    ) -> Any:
        return stub_module

    main_app.dependency_overrides[get_conversational_module] = (
        override_get_conversational_module
    )

    with TestClient(main_app) as client:
        client.stub_module = stub_module  # type: ignore[attr-defined]
        yield client

    hippocampus._memory.clear()
    hippocampus._locks.clear()
    main_app.dependency_overrides.pop(get_conversational_module, None)


@pytest.mark.usefixtures("ensure_test_users")
def test_chat_endpoint_returns_structured_payload(chat_test_client: TestClient) -> None:
    """The chat endpoint should surface a fully structured JSON payload."""

    token = chat_test_client.post(
        "/token",
        data={"username": "u1", "password": "x"},
    ).json()["access_token"]

    payload = {"message": "ping", "session_id": "session-42"}
    response = chat_test_client.post(
        "/api/v1/conversation/chat",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["response"] == "echo::ping"
    assert data["confidence"] == pytest.approx(0.88)
    assert data["processing_time"] == pytest.approx(0.42)

    speech_turn = data["speech_turn"]
    assert speech_turn["turn_id"] == "turn-001"
    assert speech_turn["text"] == "echo::ping"
    assert speech_turn["segments"][0]["text"] == "echo::ping"
    assert speech_turn["segments"][0]["estimated_duration"] == pytest.approx(0.5)

    recorded_call = chat_test_client.stub_module.calls[-1]  # type: ignore[attr-defined]
    assert recorded_call == {
        "user_id": "u1",
        "query": "ping",
        "session_id": "session-42",
        "image_data": None,
    }


class _FakeTensor:
    """Minimal tensor stand-in returned by the fake embedding wrapper."""

    def __init__(self, matrix: Sequence[Sequence[float]]) -> None:
        self._matrix = [list(row) for row in matrix]
        cols = len(self._matrix[0]) if self._matrix else 0
        self.shape = (len(self._matrix), cols)

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self._matrix]


@pytest.fixture
def embedding_test_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> TestClient:
    """Return a FastAPI client for the embedding service with a fake backend."""

    fake_config = {
        "base_model_id": "dolphin-test",
        "embedding_backend": "huggingface",
        "embedding_options": {"normalise": False, "pooling_mode": "mean"},
    }
    monkeypatch.setattr(
        run_llm2vec_service,
        "load_wrapper_config",
        lambda _path: fake_config,
    )

    class DummyWrapper:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def embed(self, texts: list[str], normalise: bool | None = None) -> _FakeTensor:
            self.calls.append({"texts": list(texts), "normalise": normalise})
            matrix = [[float(i + idx) for idx in range(3)] for i, _ in enumerate(texts)]
            return _FakeTensor(matrix)

    wrapper = DummyWrapper()

    settings = SimpleNamespace(
        embedding_backend="huggingface",
        ollama_embedding_model="ollama/dolphin",
        llm2vec_vector_dimensions=3,
        llm2vec_instruction="test instruction",
    )

    service = run_llm2vec_service.EmbeddingService(
        model_dir=tmp_path_factory.mktemp("wrapper"),
        backend="huggingface",
        wrapper_factory=lambda: wrapper,
        settings=settings,
    )
    app = run_llm2vec_service.create_app(service)

    with TestClient(app) as client:
        client.wrapper = wrapper  # type: ignore[attr-defined]
        yield client


def test_embedding_endpoint_returns_vectors(embedding_test_client: TestClient) -> None:
    """The embedding endpoint should return vectors with metadata."""

    response = embedding_test_client.post(
        "/embed",
        json={"inputs": ["alpha", "beta"], "normalise": True},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["count"] == 2
    assert body["dims"] == 3
    assert body["backend"] == "huggingface"
    assert body["model"] == "dolphin-test"
    assert body["normalised"] is True

    vectors = body["vectors"]
    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert vectors[0] == [0.0, 1.0, 2.0]
    assert vectors[1] == [1.0, 2.0, 3.0]

    wrapper = embedding_test_client.wrapper  # type: ignore[attr-defined]
    assert wrapper.calls[-1] == {"texts": ["alpha", "beta"], "normalise": True}
