import pytest
from fastapi.testclient import TestClient

from monGARS.api.dependencies import hippocampus
from monGARS.api.web_api import app, websocket_connections
from monGARS.core.conversation import ConversationalModule


@pytest.fixture
def client(monkeypatch):
    hippocampus._memory.clear()
    hippocampus._locks.clear()

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
        hippocampus._memory.clear()
        hippocampus._locks.clear()
        websocket_connections.clear()


@pytest.mark.asyncio
async def test_websocket_sends_history_and_updates(client):
    await hippocampus.store("u1", "hello", "hi")
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]

    with client.websocket_connect("/ws/chat/?user_id=u1") as ws:
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
