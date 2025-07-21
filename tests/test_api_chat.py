import os
import sys
import types

os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("SECRET_KEY", "test")

import pytest
from fastapi.testclient import TestClient

from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app
from monGARS.core.conversation import ConversationalModule


@pytest.fixture
def client(monkeypatch):
    hippocampus._memory.clear()
    hippocampus._locks.clear()

    monkeypatch.setitem(
        sys.modules, "spacy", types.SimpleNamespace(load=lambda n: object())
    )
    import monGARS.core.cortex.curiosity_engine as ce

    monkeypatch.setattr(ce, "spacy", types.SimpleNamespace(load=lambda n: object()))

    async def fake_generate_response(
        self, user_id, query, session_id=None, image_data=None
    ):
        return {"text": "ok", "confidence": 0.9, "processing_time": 0.1}

    monkeypatch.setattr(
        ConversationalModule, "generate_response", fake_generate_response
    )
    client = TestClient(app)
    try:
        yield client
    finally:
        hippocampus._memory.clear()
        hippocampus._locks.clear()


@pytest.mark.asyncio
async def test_chat_returns_response(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Validate response fields and types
    assert set(data) == {"response", "confidence", "processing_time"}
    assert isinstance(data["response"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["processing_time"], float)

    assert data["response"] == "ok"
    assert data["confidence"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_chat_empty_message_returns_422(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.post(
        "/api/v1/conversation/chat",
        json={"message": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_message_too_long_returns_422(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    long_message = "x" * 1001
    resp = client.post(
        "/api/v1/conversation/chat",
        json={"message": long_message},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_requires_auth(client: TestClient):
    resp = client.post("/api/v1/conversation/chat", json={"message": "hello"})
    assert resp.status_code == 401
