from __future__ import annotations

import pytest

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
