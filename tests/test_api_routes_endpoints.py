"""Comprehensive coverage for recently added FastAPI routes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio
from fastapi import status
from httpx import ASGITransport, AsyncClient

from monGARS.api import ui
from monGARS.api.dependencies import (
    get_hippocampus,
    get_model_manager,
    get_peer_communicator,
    get_persistence_repository,
    get_rag_context_enricher,
)
from monGARS.api.schemas import RagContextResponse
from monGARS.api.web_api import (
    app,
    get_conversational_module,
    reset_chat_rate_limiter_async,
    sec_manager,
)
from monGARS.core.conversation import PromptTooLargeError
from monGARS.core.hippocampus import MemoryItem
from monGARS.core.model_manager import (
    LLMModelManager,
    ModelDefinition,
    ModelProfile,
    ModelProvisionReport,
    ModelProvisionStatus,
)
from monGARS.core.peer import PeerCommunicator
from monGARS.core.rag import RagCodeReference, RagEnrichmentResult, RagServiceError

os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("SECRET_KEY", "super-secret")


class FakeUser:
    def __init__(self, username: str, password_hash: str, *, is_admin: bool) -> None:
        self.username = username
        self.password_hash = password_hash
        self.is_admin = is_admin


class FakePersistenceRepository:
    """In-memory substitute for ``PersistenceRepository`` used in API tests."""

    def __init__(self) -> None:
        self._users: dict[str, FakeUser] = {}
        self.raise_on_has_admin: Exception | None = None
        self.update_result_override: bool | None = None

    def seed_user(
        self, username: str, password: str, *, is_admin: bool = False
    ) -> None:
        self._users[username] = FakeUser(
            username,
            sec_manager.get_password_hash(password),
            is_admin=is_admin,
        )

    async def create_user_atomic(
        self, username: str, password_hash: str, *, is_admin: bool = False
    ) -> None:
        if username in self._users:
            raise ValueError("Username already exists")
        self._users[username] = FakeUser(username, password_hash, is_admin=is_admin)

    async def create_user(
        self, username: str, password_hash: str, *, is_admin: bool = False
    ) -> None:
        await self.create_user_atomic(username, password_hash, is_admin=is_admin)

    async def has_admin_user(self) -> bool:
        if self.raise_on_has_admin is not None:
            exc = self.raise_on_has_admin
            self.raise_on_has_admin = None
            raise exc
        return any(user.is_admin for user in self._users.values())

    async def get_user_by_username(self, username: str) -> FakeUser | None:
        return self._users.get(username)

    async def list_usernames(self) -> list[str]:
        return sorted(self._users)

    async def update_user_password(self, username: str, password_hash: str) -> bool:
        if self.update_result_override is not None:
            return self.update_result_override
        user = self._users.get(username)
        if user is None:
            return False
        self._users[username] = FakeUser(
            username, password_hash, is_admin=user.is_admin
        )
        return True


class FakeHippocampus:
    def __init__(self) -> None:
        self._entries: dict[str, list[MemoryItem]] = {}

    def add_history(self, user_id: str, query: str, response: str) -> None:
        entries = self._entries.setdefault(user_id, [])
        entries.append(MemoryItem(user_id=user_id, query=query, response=response))

    async def history(self, user_id: str, limit: int = 10) -> list[MemoryItem]:
        entries = self._entries.get(user_id, [])
        return list(reversed(entries[-limit:]))


class FakeConversationalModule:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None
        self.response: dict[str, Any] = {
            "text": "ack",
            "confidence": 0.75,
            "processing_time": 0.12,
            "speech_turn": {
                "turn_id": "turn-1",
                "text": "ack",
                "created_at": "2024-01-01T00:00:00Z",
                "segments": [
                    {
                        "text": "ack",
                        "estimated_duration": 0.5,
                        "pause_after": 0.1,
                    }
                ],
                "average_words_per_second": 2.5,
                "tempo": 1.0,
            },
        }
        self.raise_exc: Exception | None = None

    async def generate_response(
        self,
        user_id: str,
        query: str,
        *,
        session_id: str | None = None,
        image_data: bytes | None = None,
    ) -> dict[str, Any]:
        self.last_call = {
            "user_id": user_id,
            "query": query,
            "session_id": session_id,
            "image_data": image_data,
        }
        if self.raise_exc:
            raise self.raise_exc
        return self.response


class FakeRagEnricher:
    def __init__(self) -> None:
        self.raise_error: Exception | None = None
        self.calls: list[dict[str, Any]] = []
        self.result = RagEnrichmentResult(
            focus_areas=["Improve validation"],
            references=[
                RagCodeReference(
                    repository="repo",
                    file_path="api/routes.py",
                    summary="Ensure 422 on invalid payload",
                    score=0.9,
                    url="https://example.com/routes",
                )
            ],
        )

    async def enrich(
        self,
        query: str,
        *,
        repositories: list[str] | None = None,
        max_results: int | None = None,
    ) -> RagEnrichmentResult:
        self.calls.append(
            {
                "query": query,
                "repositories": repositories,
                "max_results": max_results,
            }
        )
        if self.raise_error:
            raise self.raise_error
        return self.result


class FakePeerCommunicator(PeerCommunicator):
    def __init__(self) -> None:
        super().__init__()
        self.last_payload: Any = None
        self.decode_error: Exception | None = None
        self.local_load: dict[str, Any] = {
            "scheduler_id": "node-a",
            "queue_depth": 1,
            "active_workers": 1,
            "concurrency": 2,
            "load_factor": 0.5,
        }
        self.telemetry_records: list[dict[str, Any]] = []

    def decode(self, payload: str) -> Any:  # type: ignore[override]
        if self.decode_error:
            raise self.decode_error
        data = json.loads(payload)
        self.last_payload = data
        return data

    async def get_local_load(self) -> dict[str, Any]:  # type: ignore[override]
        return dict(self.local_load)

    def ingest_remote_telemetry(self, source: str, payload: dict[str, Any]) -> None:
        self.telemetry_records.append({"source": source, "payload": payload})

    def get_peer_telemetry(self, include_self: bool = True) -> list[dict[str, Any]]:
        entries = [
            {
                "source": "remote",
                "scheduler_id": "remote",
                "queue_depth": 3,
                "active_workers": 2,
                "concurrency": 4,
                "load_factor": 0.7,
            }
        ]
        if include_self:
            entries.append({"source": "self", **self.local_load})
        return entries


class FakeModelManager(LLMModelManager):
    def __init__(self) -> None:
        super().__init__()
        self._profile = ModelProfile(
            name="default",
            models={
                "general": ModelDefinition(
                    role="general",
                    name="fake/general",
                    provider="ollama",
                    parameters={"temperature": 0.1},
                    description="General responses",
                ),
                "coding": ModelDefinition(
                    role="coding",
                    name="fake/coder",
                    provider="ollama",
                    auto_download=False,
                ),
            },
        )
        self._profiles = ["default", "experimental"]
        self.raise_on_provision: Exception | None = None
        self.ensure_calls: list[dict[str, Any]] = []

    def get_profile_snapshot(self, name: str | None = None) -> ModelProfile:  # type: ignore[override]
        return ModelProfile(name=self._profile.name, models=dict(self._profile.models))

    def available_profile_names(self) -> list[str]:  # type: ignore[override]
        return list(self._profiles)

    def active_profile_name(self) -> str:  # type: ignore[override]
        return self._profile.name

    async def ensure_models_installed(
        self, roles: list[str] | None = None, *, force: bool = False
    ) -> ModelProvisionReport:
        self.ensure_calls.append({"roles": roles, "force": force})
        if self.raise_on_provision:
            raise self.raise_on_provision
        return ModelProvisionReport(
            statuses=[
                ModelProvisionStatus(
                    role="general",
                    name="fake/general",
                    provider="ollama",
                    action="exists",
                ),
                ModelProvisionStatus(
                    role="coding",
                    name="fake/coder",
                    provider="ollama",
                    action="installed" if force else "skipped",
                ),
            ]
        )


class FakeSuggester:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.model_name = "fake-suggester"

    async def order(self, prompt: str, actions: list[tuple[str, str]]):
        self.calls.append({"prompt": prompt, "actions": actions})
        ordered = [name for name, _ in actions]
        scores = {name: 1.0 for name in ordered}
        return ordered, scores


@dataclass(slots=True)
class ApiTestContext:
    client: AsyncClient
    repo: FakePersistenceRepository
    hippocampus: FakeHippocampus
    conv: FakeConversationalModule
    rag: FakeRagEnricher
    peer: FakePeerCommunicator
    model_manager: FakeModelManager
    suggester: FakeSuggester


@pytest_asyncio.fixture
async def api_context(monkeypatch) -> ApiTestContext:
    repo = FakePersistenceRepository()
    hippocampus = FakeHippocampus()
    conv = FakeConversationalModule()
    rag = FakeRagEnricher()
    peer = FakePeerCommunicator()
    model_manager = FakeModelManager()
    suggester = FakeSuggester()

    overrides = {
        get_persistence_repository: lambda: repo,
        get_hippocampus: lambda: hippocampus,
        get_peer_communicator: lambda: peer,
        get_rag_context_enricher: lambda: rag,
        get_model_manager: lambda: model_manager,
        get_conversational_module: lambda: conv,
    }

    for dependency, factory in overrides.items():
        app.dependency_overrides[dependency] = factory

    original_suggester = ui._suggester
    ui._suggester = suggester  # type: ignore[assignment]

    await reset_chat_rate_limiter_async()
    try:
        transport = ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                yield ApiTestContext(
                    client=client,
                    repo=repo,
                    hippocampus=hippocampus,
                    conv=conv,
                    rag=rag,
                    peer=peer,
                    model_manager=model_manager,
                    suggester=suggester,
                )
    finally:
        app.dependency_overrides.clear()
        ui._suggester = original_suggester  # type: ignore[assignment]
        await reset_chat_rate_limiter_async()
        from monGARS.api import web_api

        web_api.conversation_module = None


async def _get_token(client: AsyncClient, username: str, password: str) -> str:
    response = await client.post(
        "/token", data={"username": username, "password": password}
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    return response.json()["access_token"]


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_login_returns_token_with_admin_claim(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.seed_user("alice", "wonderland", is_admin=True)
    response = await api_context.client.post(
        "/token", data={"username": "alice", "password": "wonderland"}
    )
    assert response.status_code == status.HTTP_200_OK
    token = response.json()["access_token"]
    payload = sec_manager.verify_token(token)
    assert payload["sub"] == "alice"
    assert payload["admin"] is True


@pytest.mark.asyncio
async def test_login_rejects_invalid_credentials(api_context: ApiTestContext) -> None:
    response = await api_context.client.post(
        "/token", data={"username": "ghost", "password": "nope"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_register_user_persists_credentials(api_context: ApiTestContext) -> None:
    response = await api_context.client.post(
        "/api/v1/user/register",
        json={"username": "bob", "password": "complex-pass"},
    )
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body == {"status": "registered", "is_admin": False}
    user = await api_context.repo.get_user_by_username("bob")
    assert user is not None
    assert user.is_admin is False
    assert sec_manager.verify_password("complex-pass", user.password_hash)


@pytest.mark.asyncio
async def test_register_user_conflict_returns_409(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("taken", "password123")
    response = await api_context.client.post(
        "/api/v1/user/register",
        json={"username": "taken", "password": "password123"},
    )
    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["detail"] == "Username already exists"


@pytest.mark.asyncio
async def test_register_admin_allows_single_creation(
    api_context: ApiTestContext,
) -> None:
    first = await api_context.client.post(
        "/api/v1/user/register/admin",
        json={"username": "founder", "password": "founderpw"},
    )
    assert first.status_code == status.HTTP_200_OK
    assert first.json() == {"status": "registered", "is_admin": True}

    second = await api_context.client.post(
        "/api/v1/user/register/admin",
        json={"username": "cofounder", "password": "cofounderpw"},
    )
    assert second.status_code == status.HTTP_403_FORBIDDEN
    assert second.json()["detail"] == "Admin already exists"


@pytest.mark.asyncio
async def test_register_admin_reports_state_failure(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.raise_on_has_admin = RuntimeError("database offline")
    response = await api_context.client.post(
        "/api/v1/user/register/admin",
        json={"username": "founder", "password": "founderpw"},
    )
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"] == "Unable to determine admin availability"


@pytest.mark.asyncio
async def test_change_password_updates_hash(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("carol", "initialpw")
    token = await _get_token(api_context.client, "carol", "initialpw")

    response = await api_context.client.post(
        "/api/v1/user/change-password",
        headers=_bearer(token),
        json={"old_password": "initialpw", "new_password": "replacement"},
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "changed"}

    user = await api_context.repo.get_user_by_username("carol")
    assert user is not None
    assert sec_manager.verify_password("replacement", user.password_hash)

    old_login = await api_context.client.post(
        "/token", data={"username": "carol", "password": "initialpw"}
    )
    assert old_login.status_code == status.HTTP_401_UNAUTHORIZED
    new_login = await api_context.client.post(
        "/token", data={"username": "carol", "password": "replacement"}
    )
    assert new_login.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_change_password_rejects_wrong_old_password(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.seed_user("dave", "secretpw")
    token = await _get_token(api_context.client, "dave", "secretpw")

    response = await api_context.client.post(
        "/api/v1/user/change-password",
        headers=_bearer(token),
        json={"old_password": "wrongpwd", "new_password": "updated1"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"] == "Incorrect password"


@pytest.mark.asyncio
async def test_change_password_missing_user_returns_404(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.seed_user("erin", "secretpw")
    token = await _get_token(api_context.client, "erin", "secretpw")
    api_context.repo._users.pop("erin")

    response = await api_context.client.post(
        "/api/v1/user/change-password",
        headers=_bearer(token),
        json={"old_password": "secretpw", "new_password": "updated1"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "User not found"


@pytest.mark.asyncio
async def test_change_password_update_failure_returns_404(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.seed_user("frank", "secretpw")
    token = await _get_token(api_context.client, "frank", "secretpw")
    api_context.repo.update_result_override = False

    response = await api_context.client.post(
        "/api/v1/user/change-password",
        headers=_bearer(token),
        json={"old_password": "secretpw", "new_password": "updated1"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "User not found"


@pytest.mark.asyncio
async def test_change_password_invalid_token_subject(
    api_context: ApiTestContext,
) -> None:
    token = sec_manager.create_access_token({"sub": "", "admin": False})
    response = await api_context.client.post(
        "/api/v1/user/change-password",
        headers=_bearer(token),
        json={"old_password": "secretpw", "new_password": "updated1"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid token: missing subject"


@pytest.mark.asyncio
async def test_list_users_requires_admin(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("admin", "secret", is_admin=True)
    api_context.repo.seed_user("member", "pass")
    admin_token = await _get_token(api_context.client, "admin", "secret")
    member_token = await _get_token(api_context.client, "member", "pass")

    no_auth = await api_context.client.get("/api/v1/user/list")
    assert no_auth.status_code == status.HTTP_401_UNAUTHORIZED

    forbidden = await api_context.client.get(
        "/api/v1/user/list", headers=_bearer(member_token)
    )
    assert forbidden.status_code == status.HTTP_403_FORBIDDEN

    response = await api_context.client.get(
        "/api/v1/user/list", headers=_bearer(admin_token)
    )
    assert response.status_code == status.HTTP_200_OK
    assert set(response.json()["users"]).issuperset({"admin", "member"})


@pytest.mark.asyncio
async def test_conversation_history_returns_latest_entries(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.seed_user("hist", "pw")
    api_context.hippocampus.add_history("hist", "q1", "r1")
    api_context.hippocampus.add_history("hist", "q2", "r2")
    token = await _get_token(api_context.client, "hist", "pw")

    response = await api_context.client.get(
        "/api/v1/conversation/history",
        params={"user_id": "hist", "limit": 1},
        headers=_bearer(token),
    )
    assert response.status_code == status.HTTP_200_OK
    items = response.json()
    assert len(items) == 1
    assert items[0]["query"] == "q2"


@pytest.mark.asyncio
async def test_conversation_history_forbidden_for_other_user(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.seed_user("owner", "pw")
    api_context.repo.seed_user("intruder", "pw")
    token = await _get_token(api_context.client, "owner", "pw")

    response = await api_context.client.get(
        "/api/v1/conversation/history",
        params={"user_id": "intruder"},
        headers=_bearer(token),
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_conversation_history_invalid_limit_rejected(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.seed_user("limit", "pw")
    token = await _get_token(api_context.client, "limit", "pw")
    response = await api_context.client.get(
        "/api/v1/conversation/history",
        params={"user_id": "limit", "limit": 0},
        headers=_bearer(token),
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_chat_returns_response_payload(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("chatter", "pw")
    token = await _get_token(api_context.client, "chatter", "pw")

    response = await api_context.client.post(
        "/api/v1/conversation/chat",
        headers=_bearer(token),
        json={"message": "hello there", "session_id": "sess-1"},
    )
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["response"] == "ack"
    assert api_context.conv.last_call == {
        "user_id": "chatter",
        "query": "hello there",
        "session_id": "sess-1",
        "image_data": None,
    }


@pytest.mark.asyncio
async def test_chat_validation_error_from_user_input(
    api_context: ApiTestContext, monkeypatch
) -> None:
    api_context.repo.seed_user("validator", "pw")
    token = await _get_token(api_context.client, "validator", "pw")

    def invalid_input(_: dict) -> dict:
        raise ValueError("query rejected")

    monkeypatch.setattr("monGARS.core.security.validate_user_input", invalid_input)
    monkeypatch.setattr("monGARS.api.web_api.validate_user_input", invalid_input)

    response = await api_context.client.post(
        "/api/v1/conversation/chat",
        headers=_bearer(token),
        json={"message": "  whitespace  "},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json()["detail"] == "query rejected"


@pytest.mark.asyncio
async def test_chat_prompt_too_large_returns_413(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("prompter", "pw")
    token = await _get_token(api_context.client, "prompter", "pw")
    api_context.conv.raise_exc = PromptTooLargeError(prompt_tokens=6000, limit=4096)

    response = await api_context.client.post(
        "/api/v1/conversation/chat",
        headers=_bearer(token),
        json={"message": "expand this"},
    )
    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert "Prompt exceeds" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_unexpected_error_returns_502(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("breaker", "pw")
    token = await _get_token(api_context.client, "breaker", "pw")
    api_context.conv.raise_exc = RuntimeError("backend down")

    response = await api_context.client.post(
        "/api/v1/conversation/chat",
        headers=_bearer(token),
        json={"message": "run"},
    )
    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == "Failed to generate chat response"


@pytest.mark.asyncio
async def test_rag_context_success(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("reviewer", "pw")
    token = await _get_token(api_context.client, "reviewer", "pw")

    response = await api_context.client.post(
        "/api/v1/review/rag-context",
        headers=_bearer(token),
        json={"query": "Check validation", "repositories": ["repo"], "max_results": 3},
    )
    assert response.status_code == status.HTTP_200_OK
    body = RagContextResponse.model_validate(response.json())
    assert body.references[0].file_path == "api/routes.py"
    assert api_context.rag.calls[-1]["repositories"] == ["repo"]


@pytest.mark.asyncio
async def test_rag_context_disabled(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("reviewer", "pw")
    token = await _get_token(api_context.client, "reviewer", "pw")

    from monGARS.core.rag import RagDisabledError

    api_context.rag.raise_error = RagDisabledError("disabled")
    response = await api_context.client.post(
        "/api/v1/review/rag-context",
        headers=_bearer(token),
        json={"query": "Check"},
    )
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"] == "RAG disabled"


@pytest.mark.asyncio
async def test_rag_context_service_error(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("reviewer", "pw")
    token = await _get_token(api_context.client, "reviewer", "pw")
    api_context.rag.raise_error = RagServiceError("unavailable")

    response = await api_context.client.post(
        "/api/v1/review/rag-context",
        headers=_bearer(token),
        json={"query": "Check"},
    )
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_rag_context_validation_error(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("reviewer", "pw")
    token = await _get_token(api_context.client, "reviewer", "pw")
    api_context.rag.raise_error = ValueError("invalid query")

    response = await api_context.client.post(
        "/api/v1/review/rag-context",
        headers=_bearer(token),
        json={"query": "Check"},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json()["detail"] == "invalid query"


@pytest.mark.asyncio
async def test_suggestions_returns_ranked_actions(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("suggest", "pw")
    token = await _get_token(api_context.client, "suggest", "pw")

    response = await api_context.client.post(
        "/api/v1/ui/suggestions",
        headers=_bearer(token),
        json={"prompt": "help", "actions": ["code", "summarize"]},
    )
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["actions"] == ["code", "summarize"]
    assert api_context.suggester.calls[-1]["actions"][0][0] == "code"


@pytest.mark.asyncio
async def test_peer_message_success(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("peer", "pw")
    token = await _get_token(api_context.client, "peer", "pw")

    payload = json.dumps({"hello": "world"})
    response = await api_context.client.post(
        "/api/v1/peer/message",
        headers=_bearer(token),
        json={"payload": payload},
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "received"
    assert api_context.peer.last_payload == {"hello": "world"}


@pytest.mark.asyncio
async def test_peer_message_validation_errors(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("peer", "pw")
    token = await _get_token(api_context.client, "peer", "pw")

    api_context.peer.decode_error = ValueError("bad payload")
    response = await api_context.client.post(
        "/api/v1/peer/message",
        headers=_bearer(token),
        json={"payload": json.dumps({})},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST

    api_context.peer.decode_error = RuntimeError("decoder broke")
    response = await api_context.client.post(
        "/api/v1/peer/message",
        headers=_bearer(token),
        json={"payload": json.dumps({})},
    )
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_peer_registration_flow(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("admin", "secret", is_admin=True)
    token = await _get_token(api_context.client, "admin", "secret")

    register = await api_context.client.post(
        "/api/v1/peer/register",
        headers=_bearer(token),
        json={"url": "http://peer.example/api"},
    )
    assert register.status_code == status.HTTP_200_OK
    assert register.json()["status"] == "registered"

    duplicate = await api_context.client.post(
        "/api/v1/peer/register",
        headers=_bearer(token),
        json={"url": "http://peer.example/api"},
    )
    assert duplicate.status_code == status.HTTP_200_OK
    assert duplicate.json()["status"] == "already registered"

    peer_list = await api_context.client.get(
        "/api/v1/peer/list", headers=_bearer(token)
    )
    assert peer_list.status_code == status.HTTP_200_OK
    assert peer_list.json() == ["http://peer.example/api"]

    unreg = await api_context.client.post(
        "/api/v1/peer/unregister",
        headers=_bearer(token),
        json={"url": "http://peer.example/api"},
    )
    assert unreg.status_code == status.HTTP_200_OK
    assert unreg.json()["status"] == "unregistered"

    missing = await api_context.client.post(
        "/api/v1/peer/unregister",
        headers=_bearer(token),
        json={"url": "http://peer.example/api"},
    )
    assert missing.status_code == status.HTTP_200_OK
    assert missing.json()["status"] == "not registered"


@pytest.mark.asyncio
async def test_peer_load_and_telemetry(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("admin", "secret", is_admin=True)
    token = await _get_token(api_context.client, "admin", "secret")

    load_resp = await api_context.client.get(
        "/api/v1/peer/load", headers=_bearer(token)
    )
    assert load_resp.status_code == status.HTTP_200_OK
    assert load_resp.json()["scheduler_id"] == "node-a"

    ingest = await api_context.client.post(
        "/api/v1/peer/telemetry",
        headers=_bearer(token),
        json={
            "source": "peer-1",
            "scheduler_id": "peer-1",
            "queue_depth": 4,
            "active_workers": 2,
            "concurrency": 4,
            "load_factor": 0.75,
        },
    )
    assert ingest.status_code == status.HTTP_202_ACCEPTED
    assert api_context.peer.telemetry_records[-1]["source"] == "peer-1"

    snapshot = await api_context.client.get(
        "/api/v1/peer/telemetry", headers=_bearer(token)
    )
    assert snapshot.status_code == status.HTTP_200_OK
    payload = snapshot.json()["telemetry"]
    assert any(entry["source"] == "remote" for entry in payload)


@pytest.mark.asyncio
async def test_model_configuration_requires_admin(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("user", "pw")
    token = await _get_token(api_context.client, "user", "pw")

    response = await api_context.client.get("/api/v1/models", headers=_bearer(token))
    assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_model_configuration_returns_snapshot(
    api_context: ApiTestContext,
) -> None:
    api_context.repo.seed_user("admin", "secret", is_admin=True)
    token = await _get_token(api_context.client, "admin", "secret")

    response = await api_context.client.get("/api/v1/models", headers=_bearer(token))
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["active_profile"] == "default"
    assert body["available_profiles"] == ["default", "experimental"]
    assert body["profile"]["models"]["coding"]["auto_download"] is False


@pytest.mark.asyncio
async def test_model_provision_invokes_manager(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("admin", "secret", is_admin=True)
    token = await _get_token(api_context.client, "admin", "secret")

    response = await api_context.client.post(
        "/api/v1/models/provision",
        headers=_bearer(token),
        json={"roles": ["GENERAL", "coding"], "force": True},
    )
    assert response.status_code == status.HTTP_200_OK
    assert api_context.model_manager.ensure_calls == [
        {"roles": ["general", "coding"], "force": True}
    ]


@pytest.mark.asyncio
async def test_model_provision_failure_returns_502(api_context: ApiTestContext) -> None:
    api_context.repo.seed_user("admin", "secret", is_admin=True)
    token = await _get_token(api_context.client, "admin", "secret")
    from httpx import HTTPError

    api_context.model_manager.raise_on_provision = HTTPError("down")

    response = await api_context.client.post(
        "/api/v1/models/provision",
        headers=_bearer(token),
        json={"roles": None, "force": False},
    )
    assert response.status_code == status.HTTP_502_BAD_GATEWAY
