"""Typed models shared by the monGARS Python SDK."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class TokenResponse(BaseModel):
    """Response payload returned by the ``/token`` endpoint."""

    access_token: str = Field(..., alias="access_token")
    token_type: str = Field(..., alias="token_type")


class UserRegistration(BaseModel):
    """Payload describing a new user registration request."""

    username: str
    password: str


class ChatRequest(BaseModel):
    """Chat request sent to the conversational endpoint."""

    message: str
    session_id: str | None = None
    allowed_actions: list[str] | None = None
    approval_token: str | None = None
    token_ref: str | None = None


class SpeechSegment(BaseModel):
    text: str
    estimated_duration: float
    pause_after: float


class SpeechTurn(BaseModel):
    turn_id: str
    text: str
    created_at: datetime
    segments: list[SpeechSegment]
    average_words_per_second: float
    tempo: float


class ChatResponse(BaseModel):
    """Canonical response returned by ``/api/v1/conversation/chat``."""

    response: str
    confidence: float
    processing_time: float
    speech_turn: SpeechTurn


class MemoryItem(BaseModel):
    """Conversation history entry returned by the API."""

    user_id: str
    query: str
    response: str
    timestamp: datetime


class RagContextRequest(BaseModel):
    query: str
    repositories: list[str] | None = None
    max_results: int | None = None


class RagReference(BaseModel):
    repository: str
    file_path: str
    summary: str
    score: float | None = None
    url: str | None = None


class RagContextResponse(BaseModel):
    enabled: bool
    focus_areas: list[str]
    references: list[RagReference]


class PeerRegistration(BaseModel):
    url: HttpUrl


class PeerLoadSnapshot(BaseModel):
    scheduler_id: str | None = None
    queue_depth: int = 0
    active_workers: int = 0
    concurrency: int = 0
    load_factor: float = 0.0


class PeerTelemetryPayload(PeerLoadSnapshot):
    worker_uptime_seconds: float = 0.0
    tasks_processed: int = 0
    tasks_failed: int = 0
    task_failure_rate: float = 0.0
    observed_at: datetime | None = None
    source: str | None = None


class PeerTelemetryEnvelope(BaseModel):
    telemetry: list[PeerTelemetryPayload] = Field(default_factory=list)


class SuggestRequest(BaseModel):
    prompt: str
    actions: list[str] | None = None


class SuggestResponse(BaseModel):
    actions: list[str]
    scores: dict[str, float]
    model: str


class ModelDefinition(BaseModel):
    role: str
    name: str
    provider: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    auto_download: bool = True
    description: str | None = None


class ModelProfile(BaseModel):
    name: str
    models: dict[str, ModelDefinition]


class ModelConfiguration(BaseModel):
    active_profile: str
    available_profiles: list[str]
    profile: ModelProfile


class ProvisionRequest(BaseModel):
    roles: list[str] | None = None
    force: bool = False


class ProvisionStatus(BaseModel):
    role: str
    name: str
    provider: str
    action: str
    detail: str | None = None


class ProvisionReport(BaseModel):
    statuses: list[ProvisionStatus]
