from __future__ import annotations

import pytest
from pydantic import ValidationError

from monGARS.api.schemas import (
    ChatRequest,
    PeerMessage,
    PeerRegistration,
    SuggestRequest,
    UserRegistration,
)


def test_user_registration_strips_and_validates_username() -> None:
    payload = UserRegistration(username="  alice-01  ", password="supersecret")
    assert payload.username == "alice-01"


def test_user_registration_rejects_bad_username() -> None:
    with pytest.raises(ValidationError):
        UserRegistration(username="bad email@example.com", password="supersecret")


def test_chat_request_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="   ", session_id=None)


def test_peer_registration_normalises_url() -> None:
    reg = PeerRegistration(url="https://example.com/api/")
    assert reg.url == "https://example.com/api"


def test_peer_message_requires_payload_content() -> None:
    with pytest.raises(ValidationError):
        PeerMessage(payload="   ")


def test_suggest_request_deduplicates_actions() -> None:
    request = SuggestRequest(prompt="generate", actions=[" code ", "code", "summarize"])
    assert request.actions == ["code", "summarize"]


def test_suggest_request_rejects_empty_actions() -> None:
    with pytest.raises(ValidationError):
        SuggestRequest(prompt="explain", actions=["   "])
