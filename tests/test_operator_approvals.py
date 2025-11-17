from pathlib import Path

import pytest

from monGARS.core.operator_approvals import (
    OperatorApprovalRegistry,
    generate_approval_token,
    log_blocked_attempt,
    verify_approval_token,
)
from monGARS.core.pii_detection import PIIEntity


def test_operator_approval_registry_deduplicates_requests(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "approvals.json")
    payload = {"metrics": {"accuracy": 0.75}, "adapter": "path"}

    first = registry.submit(source="reinforcement.reasoning", payload=payload)
    assert first.is_pending

    second = registry.submit(source="reinforcement.reasoning", payload=payload)
    assert second.request_id == first.request_id
    assert second.is_pending

    approved = registry.approve(first.request_id, operator="tester")
    assert approved.is_approved

    assert registry.require_approval(source="reinforcement.reasoning", payload=payload)


def test_operator_approval_registry_auto_policy(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "auto.json")
    payload = {"metrics": {"accuracy": 0.95}, "adapter": "path"}

    request = registry.submit(
        source="reinforcement.reasoning",
        payload=payload,
        policy=lambda data: data.get("metrics", {}).get("accuracy", 0.0) > 0.9,
    )

    assert request.is_approved
    pending = list(registry.pending())
    assert not pending


def test_log_blocked_attempt_records_audit_fields(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "audit.json")
    entity = PIIEntity(type="email", value="user@example.com", start=0, end=16)

    token_ref, approval_token = log_blocked_attempt(
        user_id="alice",
        prompt_hash="deadbeef",
        pii_entities=[entity],
        required_action="approval",
        context={"allowed_actions": ["personal_data_access"], "user_id": "alice"},
        registry=registry,
    )

    assert token_ref
    assert len(approval_token) == 64
    pending = list(registry.pending(source="security.guardrail"))
    assert len(pending) == 1
    payload = pending[0].payload
    assert payload["user_id"] == "alice"
    assert payload["prompt_hash"] == "deadbeef"
    assert payload["required_action"] == "approval"
    assert payload["pii_entities"][0]["type"] == "email"
    assert payload["context_snapshot"]["allowed_actions"] == ["personal_data_access"]
    assert payload["approval_token"] == approval_token


def test_generate_approval_token_changes_with_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = iter([1000.0, 1001.0])
    monkeypatch.setattr("monGARS.core.security.time.time", lambda: next(sequence))

    first = generate_approval_token("alice", "ref123")
    second = generate_approval_token("alice", "ref123")

    assert len(first) == 64
    assert len(second) == 64
    assert first != second


def test_verify_approval_token_requires_approved_request(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "verify.json")
    entity = PIIEntity(type="email", value="user@example.com", start=0, end=16)
    prompt_hash = "feedbead1234abcd"
    token_ref, approval_token = log_blocked_attempt(
        user_id="carol",
        prompt_hash=prompt_hash,
        pii_entities=[entity],
        required_action="approval",
        context={"allowed_actions": ["personal_data_access"], "user_id": "carol"},
        registry=registry,
    )

    assert not verify_approval_token(
        user_id="carol",
        token_ref=token_ref,
        approval_token=approval_token,
        prompt_hash=prompt_hash,
        registry=registry,
    )

    registry.approve(token_ref, operator="ops")

    assert verify_approval_token(
        user_id="carol",
        token_ref=token_ref,
        approval_token=approval_token,
        prompt_hash=prompt_hash,
        registry=registry,
    )
    assert not verify_approval_token(
        user_id="carol",
        token_ref=token_ref,
        approval_token=approval_token,
        prompt_hash="wrong-hash",
        registry=registry,
    )
