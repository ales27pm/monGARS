"""Executable API surface contract tests."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple, get_args, get_origin

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from monGARS.api.dependencies import (  # noqa: E402  # isort:skip
    get_hippocampus,
    get_peer_communicator,
    get_persistence_repository,
)
from monGARS.api.web_api import (  # noqa: E402  # isort:skip
    app,
    get_conversational_module,
    sec_manager,
)
from monGARS.api.schemas import (  # noqa: E402  # isort:skip
    ChatRequest,
    ChatResponse,
    PeerLoadSnapshot,
    PeerMessage,
    PeerRegistration,
    UserRegistration,
)
from monGARS.core.hippocampus import MemoryItem  # noqa: E402  # isort:skip


@dataclass
class _FakeAccount:
    username: str
    password_hash: str
    is_admin: bool = False


class _FakePersistenceRepository:
    def __init__(self) -> None:
        self._users: dict[str, _FakeAccount] = {}

    async def get_user_by_username(self, username: str) -> _FakeAccount | None:
        return self._users.get(username)

    async def has_admin_user(self) -> bool:
        return any(user.is_admin for user in self._users.values())

    async def create_user(
        self,
        username: str,
        password_hash: str,
        *,
        is_admin: bool = False,
    ) -> _FakeAccount:
        if username in self._users:
            raise ValueError("username already exists")
        account = _FakeAccount(
            username=username, password_hash=password_hash, is_admin=is_admin
        )
        self._users[username] = account
        return account

    async def create_user_atomic(
        self,
        username: str,
        password_hash: str,
        *,
        is_admin: bool = False,
        reserved_usernames: Iterable[str] = (),
    ) -> None:
        if username in reserved_usernames or username in self._users:
            raise ValueError("username already exists")
        self._users[username] = _FakeAccount(
            username=username, password_hash=password_hash, is_admin=is_admin
        )


class _FakeHippocampus:
    async def history(self, user_id: str, *, limit: int = 10) -> list[MemoryItem]:
        return [
            MemoryItem(user_id=user_id, query="q", response="r"),
        ][:limit]


class _FakeConversationModule:
    async def generate_response(
        self, user_id: str, query: str, session_id: str | None = None
    ) -> dict[str, Any]:
        return {
            "text": "ok",
            "confidence": 0.95,
            "processing_time": 0.01,
            "speech_turn": {
                "turn_id": "turn-1",
                "text": "ok",
                "created_at": "1970-01-01T00:00:00Z",
                "segments": [],
                "average_words_per_second": 1.0,
                "tempo": 1.0,
            },
        }


class _FakePeerCommunicator:
    def __init__(self) -> None:
        self.peers: set[str] = set()

    def decode(self, payload: str) -> str:
        return payload

    async def get_local_load(self) -> dict[str, Any]:
        return {
            "scheduler_id": "local",
            "queue_depth": 0,
            "active_workers": 0,
            "concurrency": 0,
            "load_factor": 0.0,
        }


@pytest.fixture()
def contract_client() -> Iterable[Tuple[TestClient, _FakePersistenceRepository]]:
    repo = _FakePersistenceRepository()
    hippocampus = _FakeHippocampus()
    communicator = _FakePeerCommunicator()
    conversation = _FakeConversationModule()

    repo._users["u1"] = _FakeAccount(
        username="u1",
        password_hash=sec_manager.get_password_hash("x"),
        is_admin=True,
    )
    repo._users["u2"] = _FakeAccount(
        username="u2",
        password_hash=sec_manager.get_password_hash("y"),
        is_admin=False,
    )

    app.dependency_overrides[get_persistence_repository] = lambda: repo
    app.dependency_overrides[get_hippocampus] = lambda: hippocampus
    app.dependency_overrides[get_peer_communicator] = lambda: communicator
    app.dependency_overrides[get_conversational_module] = lambda: conversation

    with TestClient(app) as client:
        yield client, repo
    app.dependency_overrides.clear()


def _get_route(path: str, method: str) -> APIRoute:
    for route in app.routes:
        if (
            isinstance(route, APIRoute)
            and route.path == path
            and method.upper() in route.methods
        ):
            return route
    raise AssertionError(f"Route {method} {path} not registered")


def test_required_routes_registered() -> None:
    expected = {
        ("POST", "/token"),
        ("POST", "/api/v1/user/register"),
        ("POST", "/api/v1/user/register/admin"),
        ("GET", "/healthz"),
        ("GET", "/ready"),
        ("GET", "/api/v1/conversation/history"),
        ("POST", "/api/v1/conversation/chat"),
        ("POST", "/api/v1/peer/message"),
        ("POST", "/api/v1/peer/register"),
        ("POST", "/api/v1/peer/unregister"),
        ("GET", "/api/v1/peer/list"),
        ("GET", "/api/v1/peer/load"),
    }
    http_methods = {"GET", "POST", "PUT", "PATCH", "DELETE"}
    registered = {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
        if method in http_methods
    }
    missing = expected - registered
    assert not missing, f"Missing required routes: {missing}"


def test_chat_route_contract_models() -> None:
    route = _get_route("/api/v1/conversation/chat", "POST")
    assert route.body_field is not None
    assert route.body_field.type_ is ChatRequest
    assert route.response_model is ChatResponse


def test_history_route_contract_models() -> None:
    route = _get_route("/api/v1/conversation/history", "GET")
    origin = get_origin(route.response_model)
    args = get_args(route.response_model)
    assert origin is list
    assert args == (MemoryItem,)
    user_param = next(
        param for param in route.dependant.query_params if param.name == "user_id"
    )
    assert user_param.type_ is str
    limit_param = next(
        param for param in route.dependant.query_params if param.name == "limit"
    )
    assert limit_param.type_ is int
    assert limit_param.default == 10


def test_peer_routes_contract_models() -> None:
    message_route = _get_route("/api/v1/peer/message", "POST")
    assert message_route.body_field is not None
    assert message_route.body_field.type_ is PeerMessage

    register_route = _get_route("/api/v1/peer/register", "POST")
    assert register_route.body_field is not None
    assert register_route.body_field.type_ is PeerRegistration

    unregister_route = _get_route("/api/v1/peer/unregister", "POST")
    assert unregister_route.body_field is not None
    assert unregister_route.body_field.type_ is PeerRegistration

    list_route = _get_route("/api/v1/peer/list", "GET")
    list_origin = get_origin(list_route.response_model)
    list_args = get_args(list_route.response_model)
    assert list_origin is list
    assert list_args == (str,)

    load_route = _get_route("/api/v1/peer/load", "GET")
    assert load_route.response_model is PeerLoadSnapshot


def test_user_registration_contract_model() -> None:
    route = _get_route("/api/v1/user/register", "POST")
    assert route.body_field is not None
    assert route.body_field.type_ is UserRegistration

    admin_route = _get_route("/api/v1/user/register/admin", "POST")
    assert admin_route.body_field is not None
    assert admin_route.body_field.type_ is UserRegistration


@pytest.mark.asyncio
async def test_chat_invalid_payload_returns_422(
    contract_client: Iterable[Tuple[TestClient, _FakePersistenceRepository]],
):
    client, repo = contract_client
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    response = client.post(
        "/api/v1/conversation/chat",
        json={"message": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_history_invalid_limit_returns_422(
    contract_client: Iterable[Tuple[TestClient, _FakePersistenceRepository]],
):
    client, _ = contract_client
    token = client.post("/token", data={"username": "u1", "password": "x"}).json()[
        "access_token"
    ]
    response = client.get(
        "/api/v1/conversation/history",
        params={"user_id": "u1", "limit": 0},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "limit must be a positive integer"


@pytest.mark.asyncio
async def test_register_conflict_returns_409(
    contract_client: Iterable[Tuple[TestClient, _FakePersistenceRepository]],
):
    client, repo = contract_client
    await repo.create_user("existing", "hash")
    response = client.post(
        "/api/v1/user/register",
        json={"username": "existing", "password": "password123"},
    )
    assert response.status_code == 409
    assert "already" in response.json()["detail"].lower()


def test_openapi_schema_matches_lockfile() -> None:
    app.dependency_overrides.clear()
    schema = app.openapi()
    lock_path = Path("openapi.lock.json")
    assert lock_path.exists(), "openapi.lock.json is missing"
    locked_schema = json.loads(lock_path.read_text())
    assert schema == locked_schema
