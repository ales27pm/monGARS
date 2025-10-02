import os

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import monGARS.api.ws_manager as ws_module
from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app, ws_manager
from monGARS.config import get_settings
from monGARS.core.conversation import ConversationalModule
from monGARS.core.ui_events import make_event


@pytest_asyncio.fixture
async def client(monkeypatch):
    hippocampus._memory.clear()
    hippocampus._locks.clear()
    await ws_manager.reset()

    async def fake_generate_response(
        self, user_id, query, session_id=None, image_data=None
    ):
        return {
            "text": "resp",
            "confidence": 1.0,
            "processing_time": 0.1,
            "speech_turn": {
                "turn_id": "turn-1",
                "text": "resp",
                "created_at": datetime.now(UTC).isoformat(),
                "segments": [
                    {"text": "resp", "estimated_duration": 0.5, "pause_after": 0.3}
                ],
                "average_words_per_second": 2.4,
                "tempo": 1.0,
            },
        }

    monkeypatch.setattr(
        ConversationalModule, "generate_response", fake_generate_response
    )

    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        hippocampus._memory.clear()
        hippocampus._locks.clear()
        await ws_manager.reset()


def _issue_ws_ticket(client: TestClient, token: str) -> str:
    response = client.post(
        "/api/v1/auth/ws/ticket",
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    return response.json()["ticket"]


def _connect_ws(client: TestClient, ticket: str):
    settings = get_settings()
    origin = str(settings.WS_ALLOWED_ORIGINS[0]) if settings.WS_ALLOWED_ORIGINS else ""
    return client.websocket_connect(
        f"/ws/chat/?t={ticket}",
        headers={"origin": origin},
    )


@pytest.mark.asyncio
async def test_websocket_sends_history_and_updates(client):
    await hippocampus.store("u1", "hello", "hi")
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    ticket = _issue_ws_ticket(client, token)

    with _connect_ws(client, ticket) as ws:
        connected = ws.receive_json()
        assert connected["type"] == "ws.connected"

        snapshot = ws.receive_json()
        assert snapshot["type"] == "history.snapshot"
        items = snapshot["data"]["items"]
        assert items[0]["query"] == "hello"
        assert items[0]["response"] == "hi"

        client.post(
            "/api/v1/conversation/chat",
            json={"message": "new"},
            headers={"Authorization": f"Bearer {token}"},
        )
        second = ws.receive_json()
        assert second["type"] == "chat.message"
        assert second["data"]["query"] == "new"
        assert second["data"]["response"] == "resp"


@pytest.mark.asyncio
async def test_websocket_multiple_clients_receive_updates(client):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    ticket = _issue_ws_ticket(client, token)
    ticket_two = _issue_ws_ticket(client, token)
    await hippocampus.store("u1", "old", "resp")

    with _connect_ws(client, ticket) as ws1:
        ws1.receive_json()
        ws1.receive_json()
        with _connect_ws(client, ticket_two) as ws2:
            ws2.receive_json()
            ws2.receive_json()
            client.post(
                "/api/v1/conversation/chat",
                json={"message": "m"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert ws1.receive_json()["data"]["query"] == "m"
            assert ws2.receive_json()["data"]["query"] == "m"


@pytest.mark.asyncio
async def test_websocket_disconnect_removes_connection(client):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    ticket = _issue_ws_ticket(client, token)

    with _connect_ws(client, ticket) as ws:
        ws.receive_json()
        ws.receive_json()
        assert ws_manager.connections
    assert not ws_manager.connections


class _DummyWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def accept(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_rate_limiter_drops_events(monkeypatch):
    manager = ws_module.WebSocketManager()
    monkeypatch.setattr(ws_module.settings, "WS_RATE_LIMIT_MAX_TOKENS", 2)
    monkeypatch.setattr(ws_module.settings, "WS_RATE_LIMIT_REFILL_SECONDS", 60.0)
    ws = _DummyWebSocket()
    await manager.connect(ws, "rate-user")

    event = make_event("chat.message", "rate-user", {"seq": 1})
    await manager.send_event(event)
    await manager.send_event(event)
    await manager.send_event(event)

    assert len(ws.sent) == 2

    await manager.disconnect(ws, "rate-user")
    assert not manager.connections
