from datetime import datetime, timezone
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from monGARS.api.web_api import (
    app,
    get_conversational_module,
    get_hippocampus,
    reset_chat_rate_limiter_async,
)
from monGARS.core.hippocampus import Hippocampus

UTC = getattr(datetime, "UTC", timezone.utc)

pytestmark = pytest.mark.usefixtures("ensure_test_users")


def _speech_turn_payload(text: str, turn_index: int) -> dict[str, Any]:
    return {
        "turn_id": f"turn-{turn_index}",
        "text": text,
        "created_at": datetime.now(UTC).isoformat(),
        "segments": [
            {"text": text, "estimated_duration": 0.5, "pause_after": 0.3},
        ],
        "average_words_per_second": 2.5,
        "tempo": 1.0,
    }


class _FakeConversationalModule:
    def __init__(self, store: Hippocampus) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[str] = []
        self._store = store

    async def generate_response(
        self,
        user_id: str,
        query: str,
        session_id: str | None = None,
        image_data: bytes | None = None,
    ) -> dict[str, Any]:
        index = len(self.calls)
        if index < len(self.responses):
            response_text = self.responses[index]
        else:
            response_text = f"auto-response-{index}"
        speech_turn = _speech_turn_payload(response_text, index + 1)
        await self._store.store(user_id, query, response_text)
        call = {
            "user_id": user_id,
            "query": query,
            "session_id": session_id,
            "response": response_text,
        }
        self.calls.append(call)
        return {
            "text": response_text,
            "confidence": 0.9,
            "processing_time": 0.05,
            "speech_turn": speech_turn,
        }


@pytest_asyncio.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    store = Hippocampus()
    await reset_chat_rate_limiter_async()

    app.dependency_overrides[get_hippocampus] = lambda: store

    fake_module = _FakeConversationalModule(store)

    async def _allow_rate_limit(_: str) -> None:
        return None

    monkeypatch.setattr(
        "monGARS.api.web_api._chat_rate_limiter.ensure_permitted",
        _allow_rate_limit,
    )
    app.dependency_overrides[get_conversational_module] = lambda: fake_module

    with TestClient(app) as client:
        client.fake_conversation_module = fake_module  # type: ignore[attr-defined]
        yield client

    app.dependency_overrides.pop(get_conversational_module, None)
    app.dependency_overrides.pop(get_hippocampus, None)


@pytest.mark.asyncio
async def test_conversation_history_flow(client: TestClient) -> None:
    messages = [
        "Hello there!",
        "Can you help me with diagnostics?",
        "Thanks for the assistance.",
    ]
    responses = [
        "Hello! How can I support your research today?",
        "Certainly — here are the diagnostic steps.",
        "Happy to help!",
    ]
    client.fake_conversation_module.responses = responses  # type: ignore[attr-defined]

    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]

    session_id = "session-flow-123"
    for message, expected_response in zip(messages, responses):
        resp = client.post(
            "/api/v1/conversation/chat",
            json={"message": message, "session_id": session_id},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["response"] == expected_response
        assert pytest.approx(payload["confidence"], rel=1e-6) == 0.9
        assert payload["speech_turn"]["text"] == expected_response

    fake_calls = client.fake_conversation_module.calls  # type: ignore[attr-defined]
    assert [call["session_id"] for call in fake_calls] == [session_id] * len(messages)
    assert [call["query"] for call in fake_calls] == messages

    history_resp = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u1", "limit": len(messages)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history) == len(messages)
    assert [item["query"] for item in history] == list(reversed(messages))
    assert [item["response"] for item in history] == list(reversed(responses))
