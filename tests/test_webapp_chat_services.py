from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

import httpx
import pytest

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from monGARS.api.dependencies import get_persistence_repository, hippocampus
from monGARS.api.web_api import DEFAULT_USERS, app, get_conversational_module
from webapp.chat import services


class DummyResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):  # noqa: D401 - mimic httpx API
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class DummyAsyncClient:
    def __init__(self, response: DummyResponse, recorder: dict[str, object]):
        self._response = response
        self._recorder = recorder

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def post(self, url: str, *, json=None, headers=None):
        self._recorder["url"] = url
        self._recorder["json"] = json
        self._recorder["headers"] = headers
        return self._response

    async def get(self, url: str, *, params=None, headers=None):
        self._recorder["url"] = url
        self._recorder["params"] = params
        self._recorder["headers"] = headers
        return self._response


@pytest.mark.asyncio
async def test_post_chat_message_success(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: dict[str, object] = {}
    response = DummyResponse(
        200, {"response": "pong", "confidence": 0.9, "processing_time": 1.2}
    )

    async_client_factory = lambda *args, **kwargs: DummyAsyncClient(response, recorder)
    monkeypatch.setattr(services, "FASTAPI_URL", "http://api")
    monkeypatch.setattr(services.httpx, "AsyncClient", async_client_factory)

    result = await services.post_chat_message("alice", "token-123", "hello")

    assert recorder["url"] == "http://api/api/v1/conversation/chat"
    assert recorder["json"] == {"message": "hello"}
    assert recorder["headers"] == {"Authorization": "Bearer token-123"}
    assert result["response"] == "pong"
    assert pytest.approx(result["confidence"], rel=1e-3) == 0.9
    assert pytest.approx(result["processing_time"], rel=1e-3) == 1.2


@pytest.mark.asyncio
async def test_post_chat_message_error(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: dict[str, object] = {}
    response = DummyResponse(503, {"detail": "maintenance"})

    async_client_factory = lambda *args, **kwargs: DummyAsyncClient(response, recorder)
    monkeypatch.setattr(services, "FASTAPI_URL", "http://api")
    monkeypatch.setattr(services.httpx, "AsyncClient", async_client_factory)

    result = await services.post_chat_message("alice", "token-123", "hello")

    assert "maintenance" in result["error"]
    assert recorder["url"] == "http://api/api/v1/conversation/chat"


@dataclass
class _StubUser:
    username: str
    password_hash: str
    is_admin: bool = False


class _InMemoryRepo:
    def __init__(self) -> None:
        self._users: dict[str, _StubUser] = {}

    async def get_user_by_username(self, username: str) -> _StubUser | None:
        return self._users.get(username)

    async def create_user_atomic(
        self,
        username: str,
        password_hash: str,
        *,
        is_admin: bool = False,
        reserved_usernames: Iterable[str] | None = None,
    ) -> _StubUser:
        reserved = set(reserved_usernames or ())
        if username in reserved or username in self._users:
            raise ValueError("username already exists")
        user = _StubUser(
            username=username, password_hash=password_hash, is_admin=is_admin
        )
        self._users[username] = user
        return user

    async def create_user(
        self, username: str, password_hash: str, *, is_admin: bool = False
    ) -> _StubUser:
        if username in self._users:
            raise ValueError("username already exists")
        user = _StubUser(
            username=username, password_hash=password_hash, is_admin=is_admin
        )
        self._users[username] = user
        return user


class _StubConversationalModule:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_response(
        self,
        user_id: str,
        query: str,
        session_id: str | None = None,
        image_data: bytes | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "user_id": user_id,
                "query": query,
                "session_id": session_id,
            }
        )
        await hippocampus.store(user_id, query, "stubbed-response")
        speech_turn = {
            "turn_id": "turn-1",
            "text": "stubbed-response",
            "created_at": datetime.now(UTC).isoformat(),
            "segments": [],
            "average_words_per_second": 2.0,
            "tempo": 1.0,
        }
        return {
            "text": "stubbed-response",
            "confidence": 0.75,
            "processing_time": 0.2,
            "speech_turn": speech_turn,
        }


@pytest.mark.asyncio
async def test_services_roundtrip_against_fastapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hippocampus._memory.clear()
    hippocampus._locks.clear()
    hippocampus._hydrated_users.clear()

    repo = _InMemoryRepo()
    defaults = DEFAULT_USERS["u1"]
    repo._users["u1"] = _StubUser(
        username="u1",
        password_hash=defaults["password_hash"],
        is_admin=defaults.get("is_admin", False),
    )
    convo = _StubConversationalModule()

    app.dependency_overrides[get_persistence_repository] = lambda: repo
    app.dependency_overrides[get_conversational_module] = lambda: convo

    original_async_client = httpx.AsyncClient

    def _async_client_factory(*args, **kwargs):
        kwargs.setdefault("transport", httpx.ASGITransport(app=app))
        kwargs.setdefault("base_url", "http://testserver")
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(services.httpx, "AsyncClient", _async_client_factory)
    monkeypatch.setattr(services, "FASTAPI_URL", "http://testserver")

    try:
        await app.router.startup()

        token = await services.authenticate_user("u1", "x")
        assert token

        result = await services.post_chat_message(
            "u1", token, "hi<script>alert(1)</script>"
        )
        assert result["response"] == "stubbed-response"
        assert "error" not in result
        assert result["confidence"] == pytest.approx(0.75)
        assert result["processing_time"] == pytest.approx(0.2)

        assert convo.calls == [
            {"user_id": "u1", "query": "hialert(1)", "session_id": None}
        ]

        history = await services.fetch_history("u1", token)
        assert isinstance(history, list)
        assert history
        latest = history[0]
        assert latest["query"] == "hialert(1)"
        assert latest["response"] == "stubbed-response"
    finally:
        await app.router.shutdown()
        app.dependency_overrides.pop(get_persistence_repository, None)
        app.dependency_overrides.pop(get_conversational_module, None)
        hippocampus._memory.clear()
        hippocampus._locks.clear()
        hippocampus._hydrated_users.clear()
