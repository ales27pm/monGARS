import os

os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app, ws_manager
from monGARS.core.conversation import ConversationalModule


@pytest_asyncio.fixture
async def client(monkeypatch):
    hippocampus._memory.clear()
    hippocampus._locks.clear()
    await ws_manager.reset()

    async def fake_generate_response(
        self, user_id, query, session_id=None, image_data=None
    ):
        return {"text": "resp", "confidence": 1.0, "processing_time": 0.1}

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


@pytest.mark.asyncio
async def test_websocket_sends_history_and_updates(client):
    await hippocampus.store("u1", "hello", "hi")
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]

    with client.websocket_connect(f"/ws/chat/?token={token}") as ws:
        first = ws.receive_json()
        assert first["query"] == "hello"
        assert first["response"] == "hi"

        client.post(
            "/api/v1/conversation/chat",
            json={"message": "new"},
            headers={"Authorization": f"Bearer {token}"},
        )
        second = ws.receive_json()
        assert second["query"] == "new"
        assert second["response"] == "resp"


@pytest.mark.asyncio
async def test_websocket_multiple_clients_receive_updates(client):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    await hippocampus.store("u1", "old", "resp")

    with client.websocket_connect(f"/ws/chat/?token={token}") as ws1:
        ws1.receive_json()
        with client.websocket_connect(f"/ws/chat/?token={token}") as ws2:
            ws2.receive_json()
            client.post(
                "/api/v1/conversation/chat",
                json={"message": "m"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert ws1.receive_json()["query"] == "m"
            assert ws2.receive_json()["query"] == "m"


@pytest.mark.asyncio
async def test_websocket_disconnect_removes_connection(client):
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]

    with client.websocket_connect(f"/ws/chat/?token={token}") as ws:
        ws.receive_json()
        assert ws_manager.connections
    assert not ws_manager.connections
