from __future__ import annotations

import re

from pydantic import BaseModel, Field, HttpUrl, field_validator

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class UserRegistration(BaseModel):
    """Input payload for user registration requests."""

    username: str = Field(..., min_length=1, max_length=150)
    password: str = Field(..., min_length=8)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("username cannot be blank")
        if not USERNAME_PATTERN.match(cleaned):
            raise ValueError(
                "username may only contain letters, numbers, hyphens, or underscores"
            )
        return cleaned


class ChatRequest(BaseModel):
    """Incoming chat message sent to the conversational endpoint."""

    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=100)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be empty")
        return cleaned

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("session_id cannot be empty")
        return cleaned


class ChatResponse(BaseModel):
    """Canonical response body returned by the chat endpoint."""

    response: str
    confidence: float
    processing_time: float


class PeerMessage(BaseModel):
    """Payload accepted by the peer message endpoint."""

    payload: str = Field(..., min_length=1)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("payload cannot be empty")
        return cleaned


class PeerRegistration(BaseModel):
    """Model describing a peer registration request."""

    url: HttpUrl

    @field_validator("url")
    @classmethod
    def normalise_url(cls, value: HttpUrl) -> str:
        return str(value).rstrip("/")


class SuggestRequest(BaseModel):
    """Request body for the UI suggestion endpoint."""

    prompt: str = Field(..., min_length=1, max_length=8000)
    actions: list[str] | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("prompt cannot be empty")
        return cleaned

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        seen: set[str] = set()
        normalised: list[str] = []
        for action in value:
            cleaned = action.strip()
            if not cleaned:
                raise ValueError("actions cannot contain empty values")
            if cleaned not in seen:
                seen.add(cleaned)
                normalised.append(cleaned)
        if not normalised:
            raise ValueError("actions must include at least one non-empty value")
        return normalised


class SuggestResponse(BaseModel):
    """Response model returned by the UI suggestion endpoint."""

    actions: list[str]
    scores: dict[str, float]
    model: str


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "PeerMessage",
    "PeerRegistration",
    "SuggestRequest",
    "SuggestResponse",
    "UserRegistration",
]
