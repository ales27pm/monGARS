import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from monGARS.api.web_api import app
from monGARS.core import operator_approvals as approvals_module
from monGARS.core.operator_approvals import verify_approval_token
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
    approval_token = guard_response["approval_token"]
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    assert not verify_approval_token(
        user_id="user-1",
        token_ref=token_ref,
        approval_token=approval_token,
        prompt_hash=prompt_hash,
    )

    client = TestClient(app)
    sec_manager = SecurityManager()
    operator_token = sec_manager.create_access_token({"sub": "ops", "role": "operator"})
    response = client.post(
        "/llm/security/approve",
        params={"token": approval_token, "operator_id": "ops"},
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "approved", "token_ref": token_ref}

    assert verify_approval_token(
        user_id="user-1",
        token_ref=token_ref,
        approval_token=approval_token,
        prompt_hash=prompt_hash,
    )


def test_security_approve_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/llm/security/approve",
        params={"token": "dummy-token", "operator_id": "ops"},
    )
    assert response.status_code == 401
