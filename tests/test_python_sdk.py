from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT / "sdks" / "python"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from monGARS_sdk import (  # noqa: E402  # isort: skip
    APIError,
    AuthenticationError,
    ChatRequest,
    MonGARSAsyncClient,
    MonGARSSyncClient,
    PeerTelemetryPayload,
    ProvisionRequest,
)  # type: ignore


def _make_transport(responders: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.path}"
        responder = responders.get(key)
        if responder is None:
            return httpx.Response(404, json={"detail": "not found"})
        if callable(responder):
            return responder(request)
        status, payload = responder
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def test_sync_client_happy_path() -> None:
    responders: dict[str, Any] = {
        "POST /token": (200, {"access_token": "abc", "token_type": "bearer"}),
        "POST /api/v1/conversation/chat": (
            200,
            {
                "response": "hi",
                "confidence": 0.8,
                "processing_time": 0.1,
                "speech_turn": {
                    "turn_id": "t1",
                    "text": "hi",
                    "created_at": "2024-01-01T00:00:00Z",
                    "segments": [
                        {"text": "hi", "estimated_duration": 0.1, "pause_after": 0.0}
                    ],
                    "average_words_per_second": 2.0,
                    "tempo": 1.0,
                },
            },
        ),
        "GET /api/v1/conversation/history": (
            200,
            [
                {
                    "user_id": "alice",
                    "query": "Hello",
                    "response": "Hi!",
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            ],
        ),
        "POST /api/v1/peer/telemetry": (202, {"status": "accepted"}),
        "POST /api/v1/models/provision": (200, {"statuses": []}),
    }

    transport = _make_transport(responders)
    client = MonGARSSyncClient("https://api.example", transport=transport)

    token = client.login("alice", "secret")
    assert token.access_token == "abc"

    reply = client.chat(ChatRequest(message="Hello"))
    assert reply.response == "hi"

    history = client.history("alice")
    assert history[0].query == "Hello"

    result = client.publish_peer_telemetry(
        PeerTelemetryPayload(
            queue_depth=0,
            active_workers=0,
            concurrency=0,
            load_factor=0.0,
            worker_uptime_seconds=1.0,
            tasks_processed=1,
            tasks_failed=0,
            task_failure_rate=0.0,
        )
    )
    assert result["status"] == "accepted"

    provision = client.provision_models(ProvisionRequest(roles=["general"]))
    assert provision.statuses == []

    client.close()


def test_sync_client_raises_api_error() -> None:
    responders = {
        "POST /token": (500, {"detail": "boom"}),
    }
    client = MonGARSSyncClient(
        "https://api.example", transport=_make_transport(responders)
    )
    with pytest.raises(APIError):
        client.login("alice", "secret")


@pytest.mark.asyncio
async def test_async_client_handles_authentication_error() -> None:
    def auth_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid"})

    responders = {
        "POST /token": auth_handler,
    }
    client = MonGARSAsyncClient(
        "https://api.example", transport=_make_transport(responders)
    )
    with pytest.raises(AuthenticationError):
        await client.login("alice", "bad")


@pytest.mark.asyncio
async def test_async_client_history_round_trip() -> None:
    responders = {
        "POST /token": (200, {"access_token": "abc", "token_type": "bearer"}),
        "GET /api/v1/conversation/history": (
            200,
            [
                {
                    "user_id": "alice",
                    "query": "Hi",
                    "response": "Hello",
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            ],
        ),
    }
    async_client = MonGARSAsyncClient(
        "https://api.example", transport=_make_transport(responders)
    )
    await async_client.login("alice", "secret")
    history = await async_client.history("alice", limit=1)
    assert len(history) == 1
    await async_client.aclose()
