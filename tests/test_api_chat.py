import asyncio
import os
import sys
import types
from datetime import datetime, timezone

os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("SECRET_KEY", "test")

import pytest
import pytest_asyncio
from fastapi import status
from fastapi.testclient import TestClient

from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app, reset_chat_rate_limiter_async
from monGARS.core.conversation import ConversationalModule, PromptTooLargeError
from monGARS.core.security import SecurityManager

UTC = getattr(datetime, "UTC", timezone.utc)

pytestmark = pytest.mark.usefixtures("ensure_test_users")


def _speech_turn_payload(text: str) -> dict:
    return {
        "turn_id": "turn-1",
        "text": text,
        "created_at": datetime.now(UTC).isoformat(),
        "segments": [
            {"text": text, "estimated_duration": 0.5, "pause_after": 0.3},
        ],
        "average_words_per_second": 2.5,
        "tempo": 1.0,
    }


@pytest_asyncio.fixture
async def client(monkeypatch):
    hippocampus._memory.clear()
    hippocampus._locks.clear()
    await reset_chat_rate_limiter_async()

    monkeypatch.setitem(
        sys.modules, "spacy", types.SimpleNamespace(load=lambda n: object())
    )
    import monGARS.core.cortex.curiosity_engine as ce

    monkeypatch.setattr(ce, "spacy", types.SimpleNamespace(load=lambda n: object()))

    captured_sessions: list[str | None] = []

    async def fake_generate_response(
        self, user_id, query, session_id=None, image_data=None
    ):
        captured_sessions.append(session_id)
        return {
            "text": "ok",
            "confidence": 0.9,
            "processing_time": 0.1,
            "speech_turn": _speech_turn_payload("ok"),
        }

    monkeypatch.setattr(
        ConversationalModule, "generate_response", fake_generate_response
    )
    with TestClient(app) as client:
        client.captured_sessions = captured_sessions
        yield client
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
    assert set(data) == {"response", "confidence", "processing_time", "speech_turn"}
    assert isinstance(data["response"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["processing_time"], float)
    assert isinstance(data["speech_turn"], dict)
    assert data["speech_turn"]["text"] == "ok"

    assert data["response"] == "ok"
    assert data["confidence"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_chat_forwards_session_id(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    session_id = "session-123"
    resp = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hi", "session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert client.captured_sessions[-1] == session_id


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
async def test_chat_prompt_too_large_returns_413(
    client: TestClient, monkeypatch
) -> None:
    async def _raise_prompt_limit(*args, **kwargs):  # noqa: ANN002, ANN003
        raise PromptTooLargeError(prompt_tokens=5000, limit=4096)

    monkeypatch.setattr(ConversationalModule, "generate_response", _raise_prompt_limit)

    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    resp = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert resp.json()["detail"].startswith(
        "Prompt exceeds the maximum supported token limit"
    )


@pytest.mark.asyncio
async def test_chat_session_id_too_long_returns_422(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    long_session_id = "s" * 101
    resp = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hello", "session_id": long_session_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_requires_auth(client: TestClient):
    resp = client.post("/api/v1/conversation/chat", json={"message": "hello"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_missing_sub_in_token_returns_401(client: TestClient):
    token = SecurityManager(secret_key="test", algorithm="HS256").create_access_token(
        {"admin": False}
    )
    resp = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid token: missing subject"


@pytest.mark.asyncio
async def test_chat_rate_limit_returns_429(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    first = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hello again"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert second.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert (
        second.json()["detail"]
        == "Too many requests: please wait before sending another message."
    )


@pytest.mark.asyncio
async def test_chat_rate_limit_recovers_after_cooldown(client: TestClient):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]

    first = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hello again"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert second.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    await asyncio.sleep(1.1)

    third = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hello after cooldown"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert third.status_code == 200
