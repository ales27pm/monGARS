from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from monGARS.api.dependencies import get_approval_db_session
from monGARS.api.schemas import ChatRequest
from monGARS.api.web_api import _build_guard_context, app
from monGARS.core import operator_approvals as approvals_module
from monGARS.core.operator_approvals import log_blocked_attempt
from monGARS.core.pii_detection import PIIEntity
from monGARS.core.security import SecurityManager, pre_generation_guard


@pytest.fixture(autouse=True)
def _isolated_operator_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    approvals_path = tmp_path / "approvals.json"
    monkeypatch.setattr(approvals_module, "_DEFAULT_APPROVALS_PATH", approvals_path)
    monkeypatch.setattr(approvals_module, "_GLOBAL_REGISTRY", None)
    yield


def test_pii_block_and_operator_approval_flow() -> None:
    prompt = "My credit card number is 4111-1111-1111-1111"
    context = {"user_id": "user-1", "allowed_actions": ["financial_operation"]}

    guard_response = pre_generation_guard(prompt, context)
    assert guard_response is not None
    assert guard_response["error"] == "approval_required"

    token_ref = guard_response["token_ref"]
    assert "approval_token" not in guard_response

    client = TestClient(app)
    sec_manager = SecurityManager()
    operator_token = sec_manager.create_access_token({"sub": "ops", "role": "operator"})
    response = client.post(
        "/llm/security/approve",
        params={"token": token_ref, "operator_id": "ops"},
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "approved", "token_ref": token_ref}

    assert pre_generation_guard(prompt, {**context, "token_ref": token_ref}) is None

    replay = pre_generation_guard(prompt, {**context, "token_ref": token_ref})
    assert replay is not None
    assert replay["error"] == "approval_required"
    assert replay["token_ref"] != token_ref


def test_security_approve_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/llm/security/approve",
        params={"token": "dummy-token", "operator_id": "ops"},
    )
    assert response.status_code == 401


def test_chat_caller_cannot_downgrade_baseline_pii_action() -> None:
    context = _build_guard_context(
        ChatRequest(message="hello", allowed_actions=["code"]),
        {"sub": "alice"},
    )

    assert context["allowed_actions"] == ["code", "personal_data_access"]


def test_security_approve_persists_only_the_opaque_reference() -> None:
    class FakeQuery:
        def __init__(self, session: "FakeSession") -> None:
            self.session = session

        def filter_by(self, **filters: str) -> "FakeQuery":
            self.session.filters = filters
            return self

        def first(self):
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.filters: dict[str, str] = {}
            self.added = []

        def query(self, _model):
            return FakeQuery(self)

        def add(self, record) -> None:
            self.added.append(record)

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    registry = approvals_module.get_operator_approval_registry()
    token_ref, compatibility_proof = log_blocked_attempt(
        user_id="user-1",
        prompt_hash="deadbeef",
        pii_entities=[
            PIIEntity(type="email", value="user@example.com", start=0, end=16)
        ],
        required_action="approval",
        registry=registry,
    )

    fake_session = FakeSession()

    def _database_override():
        yield fake_session

    app.dependency_overrides[get_approval_db_session] = _database_override
    try:
        client = TestClient(app)
        sec_manager = SecurityManager()
        operator_token = sec_manager.create_access_token(
            {"sub": "ops", "role": "operator"}
        )
        response = client.post(
            "/llm/security/approve",
            params={"token": compatibility_proof, "operator_id": "ops"},
            headers={"Authorization": f"Bearer {operator_token}"},
        )
    finally:
        app.dependency_overrides.pop(get_approval_db_session, None)

    assert response.status_code == 200
    assert fake_session.filters == {"approval_token": token_ref}
    assert len(fake_session.added) == 1
    assert fake_session.added[0].approval_token == token_ref
