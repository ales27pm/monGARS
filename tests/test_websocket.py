import asyncio
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import monGARS.api.ws_manager as ws_module
from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app, reset_chat_rate_limiter_async, ws_manager
from monGARS.config import get_settings
from monGARS.core.conversation import ConversationalModule
from monGARS.core.ui_events import make_event

UTC = getattr(datetime, "UTC", timezone.utc)

pytestmark = pytest.mark.usefixtures("ensure_test_users")


@pytest_asyncio.fixture
async def client(monkeypatch):
    hippocampus._memory.clear()
    hippocampus._locks.clear()
    await reset_chat_rate_limiter_async()
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

    with TestClient(app) as client:
        yield client
    hippocampus._memory.clear()
    hippocampus._locks.clear()
    await reset_chat_rate_limiter_async()
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
        self.event = asyncio.Event()

    async def accept(self) -> None:
        return None

    async def close(self, code: int | None = None) -> None:  # noqa: ARG002
        return None

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)
        self.event.set()


@pytest.mark.asyncio
async def test_rate_limiter_drops_events(monkeypatch):
    manager = ws_module.WebSocketManager()
    monkeypatch.setattr(ws_module.settings, "WS_RATE_LIMIT_MAX_TOKENS", 2)
    monkeypatch.setattr(ws_module.settings, "WS_RATE_LIMIT_REFILL_SECONDS", 60.0)
    ws = _DummyWebSocket()
    state = await manager.connect(ws, "rate-user")
    state.sender_task = asyncio.create_task(ws_module._sender_loop(state))

    event = make_event("chat.message", "rate-user", {"seq": 1})
    await manager.send_event(event)
    await asyncio.wait_for(ws.event.wait(), timeout=0.5)
    ws.event.clear()
    await manager.send_event(event)
    await asyncio.wait_for(ws.event.wait(), timeout=0.5)
    ws.event.clear()
    await manager.send_event(event)
    await asyncio.sleep(0.05)

    assert len(ws.sent) == 2

    await manager.disconnect(ws, "rate-user")
    assert not manager.connections


@pytest.mark.asyncio
async def test_websocket_acknowledges_client_messages(client):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    ticket = _issue_ws_ticket(client, token)

    with _connect_ws(client, ticket) as ws:
        ws.receive_json()
        ws.receive_json()
        payload = {
            "id": str(uuid.uuid4()),
            "type": "client.message",
            "payload": {"ok": True},
        }
        ws.send_json(payload)
        ack = ws.receive_json()
        assert ack["type"] == "ack"
        assert ack["id"] == payload["id"]
        assert ack["payload"]["status"] == "ok"


@pytest.mark.asyncio
async def test_websocket_heartbeat_ping_pong(client, monkeypatch):
    monkeypatch.setattr(ws_module.settings, "WS_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(ws_module.settings, "WS_HEARTBEAT_TIMEOUT_SECONDS", 0.2)

    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    ticket = _issue_ws_ticket(client, token)

    with _connect_ws(client, ticket) as ws:
        ws.receive_json()
        ws.receive_json()
        ping = ws.receive_json()
        assert ping["type"] == "ping"
        ws.send_json({"id": ping["id"], "type": "pong", "payload": None})
        ack = ws.receive_json()
        assert ack["type"] == "ack"
        assert ack["id"] == ping["id"]


@pytest.mark.asyncio
async def test_websocket_accepts_client_ping(client, monkeypatch):
    monkeypatch.setattr(ws_module.settings, "WS_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(ws_module.settings, "WS_HEARTBEAT_TIMEOUT_SECONDS", 0.2)

    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    ticket = _issue_ws_ticket(client, token)

    with _connect_ws(client, ticket) as ws:
        ws.receive_json()  # history
        ws.receive_json()  # user_state

        # 1. Test simple client.ping without an ID
        ws.send_json({"type": "client.ping", "ts": "now"})
        ack_ping = ws.receive_json()
        assert ack_ping["type"] == "ack"
        assert ack_ping["payload"]["status"] == "ok"
        assert ack_ping["payload"].get("detail") == "client.ping"

        # 2. Test that client.ping doesn't break the server ping/pong flow
        server_ping = ws.receive_json()
        assert server_ping["type"] == "ping"

        # Send a client.ping before the required pong
        ws.send_json({"type": "client.ping", "ts": "now"})
        ws.receive_json()  # Ack for client.ping

        # Now send the required pong
        ws.send_json({"id": server_ping["id"], "type": "pong", "payload": None})
        ack_pong = ws.receive_json()
        assert ack_pong["type"] == "ack"
        assert ack_pong["id"] == server_ping["id"]
        assert ack_pong["payload"]["status"] == "ok", "Pong should be accepted"
