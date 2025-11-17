from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from monGARS.core.model_manager import ModelDefinition as CoreModelDefinition
    from monGARS.core.model_manager import ModelProfile as CoreModelProfile
    from monGARS.core.model_manager import (
        ModelProvisionReport as CoreModelProvisionReport,
    )
    from monGARS.core.model_manager import (
        ModelProvisionStatus as CoreModelProvisionStatus,
    )

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


class UserListResponse(BaseModel):
    """Response payload containing the registered usernames."""

    users: list[str] = Field(default_factory=list)


class PasswordChangeRequest(BaseModel):
    """Payload for password change requests."""

    old_password: str = Field(..., min_length=8)
    new_password: str = Field(..., min_length=8)


class ChatRequest(BaseModel):
    """Incoming chat message sent to the conversational endpoint."""

    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=100)
    allowed_actions: list[str] | None = None
    approval_token: str | None = Field(default=None, max_length=128)
    token_ref: str | None = Field(default=None, max_length=64)

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

    @field_validator("allowed_actions")
    @classmethod
    def validate_allowed_actions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalised: list[str] = []
        for action in value:
            cleaned = action.strip()
            if not cleaned:
                raise ValueError("allowed_actions cannot contain empty values")
            if cleaned not in normalised:
                normalised.append(cleaned)
        if not normalised:
            raise ValueError("allowed_actions must include at least one value")
        return normalised

    @field_validator("approval_token", "token_ref")
    @classmethod
    def validate_tokens(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SpeechSegmentSchema(BaseModel):
    """Schema describing a single speech segment."""

    text: str
    estimated_duration: float
    pause_after: float


class SpeechTurnSchema(BaseModel):
    """Schema describing the structure of a conversational speech turn."""

    turn_id: str
    text: str
    created_at: datetime
    segments: list[SpeechSegmentSchema]
    average_words_per_second: float
    tempo: float


class ChatResponse(BaseModel):
    """Canonical response body returned by the chat endpoint."""

    response: str
    confidence: float
    processing_time: float
    speech_turn: SpeechTurnSchema


class LLMHealthResponse(BaseModel):
    """Response returned by the low-level LLM health endpoint."""

    status: Literal["healthy", "unhealthy"]
    backend: Literal["local", "ray", "unavailable"]
    model: str | None = None
    last_check: float | None = None
    detail: str | None = None


class RagContextRequest(BaseModel):
    """Request payload for the RAG context enrichment endpoint."""

    query: str = Field(..., min_length=1, max_length=4000)
    repositories: list[str] | None = None
    max_results: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query cannot be empty")
        return cleaned

    @field_validator("repositories")
    @classmethod
    def validate_repositories(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value:
            trimmed = item.strip()
            if not trimmed:
                raise ValueError("repositories cannot contain empty values")
            lowered = trimmed.lower()
            if lowered == "all":
                return ["all"]
            if lowered not in seen:
                seen.add(lowered)
                cleaned.append(trimmed)
        return cleaned or None


class RagReferenceSchema(BaseModel):
    """Single reference entry returned by the RAG service."""

    repository: str
    file_path: str
    summary: str
    score: float | None = None
    url: str | None = None

    @field_validator("repository", "file_path", "summary")
    @classmethod
    def validate_required_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be empty")
        return cleaned


class RagContextResponse(BaseModel):
    """Response payload for the RAG context enrichment endpoint."""

    enabled: bool = True
    focus_areas: list[str] = Field(default_factory=list)
    references: list[RagReferenceSchema] = Field(default_factory=list)


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


class PeerLoadSnapshot(BaseModel):
    """Minimal load report shared between peer schedulers."""

    scheduler_id: str | None = None
    queue_depth: int = Field(default=0, ge=0)
    active_workers: int = Field(default=0, ge=0)
    concurrency: int = Field(default=0, ge=0)
    load_factor: float = Field(default=0.0, ge=0.0)

    @field_validator("load_factor")
    @classmethod
    def validate_load_factor(cls, value: float) -> float:
        if value < 0:
            raise ValueError("load_factor cannot be negative")
        return value


class PeerTelemetryPayload(PeerLoadSnapshot):
    """Detailed telemetry snapshot propagated between schedulers."""

    worker_uptime_seconds: float = Field(default=0.0, ge=0.0)
    tasks_processed: int = Field(default=0, ge=0)
    tasks_failed: int = Field(default=0, ge=0)
    task_failure_rate: float = Field(default=0.0, ge=0.0)
    observed_at: datetime | None = None
    source: str | None = Field(default=None, max_length=2048)

    @field_validator("task_failure_rate")
    @classmethod
    def validate_failure_rate(cls, value: float) -> float:
        if value < 0:
            raise ValueError("task_failure_rate cannot be negative")
        return value


class PeerTelemetryEnvelope(BaseModel):
    """Aggregated telemetry view returned by the peer telemetry endpoint."""

    telemetry: list[PeerTelemetryPayload] = Field(default_factory=list)


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


class LLMModelDefinitionSchema(BaseModel):
    """Serialised representation of a model configuration entry."""

    role: str
    name: str
    provider: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    auto_download: bool = True
    description: str | None = None

    @classmethod
    def from_definition(
        cls, definition: "CoreModelDefinition"
    ) -> "LLMModelDefinitionSchema":
        payload = definition.to_payload()
        return cls(**payload)


class LLMModelProfileSummary(BaseModel):
    """Summary of models defined under a profile."""

    name: str
    models: dict[str, LLMModelDefinitionSchema]

    @classmethod
    def from_profile(cls, profile: "CoreModelProfile") -> "LLMModelProfileSummary":
        return cls(
            name=profile.name,
            models={
                role: LLMModelDefinitionSchema.from_definition(definition)
                for role, definition in profile.models.items()
            },
        )


class LLMModelConfigurationResponse(BaseModel):
    """Response describing the active profile and available options."""

    active_profile: str
    available_profiles: list[str]
    profile: LLMModelProfileSummary

    @classmethod
    def from_profile(
        cls,
        *,
        active_profile: str,
        available_profiles: list[str],
        profile: "CoreModelProfile",
    ) -> "LLMModelConfigurationResponse":
        return cls(
            active_profile=active_profile,
            available_profiles=available_profiles,
            profile=LLMModelProfileSummary.from_profile(profile),
        )


class LLMModelProvisionStatusResponse(BaseModel):
    """Result entry returned after attempting to ensure a model."""

    role: str
    name: str
    provider: str
    action: str
    detail: str | None = None

    @classmethod
    def from_status(
        cls, status: "CoreModelProvisionStatus"
    ) -> "LLMModelProvisionStatusResponse":
        payload = status.to_payload()
        return cls(**payload)


class LLMModelProvisionReportResponse(BaseModel):
    """Aggregated provisioning report returned by the API."""

    statuses: list[LLMModelProvisionStatusResponse]

    @classmethod
    def from_report(
        cls, report: "CoreModelProvisionReport"
    ) -> "LLMModelProvisionReportResponse":
        return cls(
            statuses=[
                LLMModelProvisionStatusResponse.from_status(status)
                for status in report.statuses
            ]
        )


class LLMModelProvisionRequest(BaseModel):
    """Request body for provisioning LLM models."""

    roles: list[str] | None = None
    force: bool = False

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, value: list[str] | None) -> list[str] | None:
        if not value:
            return None
        seen: set[str] = set()
        normalised: list[str] = []
        for role in value:
            cleaned = role.strip()
            if not cleaned:
                raise ValueError("roles cannot contain empty values")
            lowered = cleaned.lower()
            if lowered not in seen:
                seen.add(lowered)
                normalised.append(lowered)
        return normalised or None


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "PeerMessage",
    "PeerLoadSnapshot",
    "PeerRegistration",
    "SuggestRequest",
    "SuggestResponse",
    "UserRegistration",
    "LLMModelConfigurationResponse",
    "LLMModelDefinitionSchema",
    "LLMModelProfileSummary",
    "LLMModelProvisionReportResponse",
    "LLMModelProvisionRequest",
    "LLMModelProvisionStatusResponse",
]
