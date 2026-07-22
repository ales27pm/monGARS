import json
from pathlib import Path

import pytest

from monGARS.core.operator_approvals import (
    OperatorApprovalRegistry,
    consume_approval,
    generate_approval_token,
    log_blocked_attempt,
    verify_approval_token,
)
from monGARS.core.pii_detection import PIIEntity
from monGARS.core.security import pre_generation_guard


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


def test_registry_rejects_unattributed_or_malformed_requests(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "invalid-submit.json")

    with pytest.raises(ValueError, match="source"):
        registry.submit(source=" ", payload={})
    with pytest.raises(TypeError, match="payload"):
        registry.submit(source="tools", payload=[])  # type: ignore[arg-type]

    pending = registry.submit(source="tools", payload={"id": "operation"})
    with pytest.raises(ValueError, match="operator"):
        registry.approve(pending.request_id, operator="")
    assert pending.is_pending


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
    assert payload["approval_token_hash"]
    assert "approval_token" not in payload
    assert "value_preview" not in payload["pii_entities"][0]
    assert "value_hash" not in payload["pii_entities"][0]


def test_log_blocked_attempt_rotates_compatibility_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "reuse.json")
    entity = PIIEntity(type="email", value="user@example.com", start=0, end=16)

    monkeypatch.setattr(
        "monGARS.core.operator_approvals._utcnow_isoformat",
        lambda: "2025-01-01T00:00:00+00:00",
    )

    first_ref, first_token = log_blocked_attempt(
        user_id="alice",
        prompt_hash="deadbeef",
        pii_entities=[entity],
        required_action="approval",
        context={"allowed_actions": ["personal_data_access"], "user_id": "alice"},
        registry=registry,
    )

    second_ref, second_token = log_blocked_attempt(
        user_id="alice",
        prompt_hash="deadbeef",
        pii_entities=[entity],
        required_action="approval",
        context={"allowed_actions": ["personal_data_access"], "user_id": "alice"},
        registry=registry,
    )

    assert first_ref == second_ref
    assert first_token != second_token
    stored = registry.get(first_ref)
    assert stored is not None
    assert "approval_token" not in stored.payload
    assert not verify_approval_token(
        user_id="alice",
        token_ref=first_ref,
        approval_token=first_token,
        prompt_hash="deadbeef",
        registry=registry,
    )
    registry.approve(first_ref, operator="ops")
    assert verify_approval_token(
        user_id="alice",
        token_ref=first_ref,
        approval_token=second_token,
        prompt_hash="deadbeef",
        registry=registry,
    )


def test_generate_approval_token_uses_fresh_randomness() -> None:
    first = generate_approval_token("alice", "ref123")
    second = generate_approval_token("alice", "ref123")

    assert len(first) == 64
    assert len(second) == 64
    assert first != second


def test_generate_approval_token_uniqueness_across_users_and_refs() -> None:
    token_a1 = generate_approval_token("alice", "ref123")
    token_a2 = generate_approval_token("alice", "ref456")
    token_b1 = generate_approval_token("bob", "ref123")
    token_b2 = generate_approval_token("bob", "ref456")

    assert token_a1 != token_a2
    assert token_a1 != token_b1
    assert token_a1 != token_b2
    assert token_a2 != token_b1
    assert token_a2 != token_b2
    assert token_b1 != token_b2


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
        user_id="mallory",
        token_ref=token_ref,
        approval_token=approval_token,
        prompt_hash=prompt_hash,
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


def test_approval_is_consumed_once_and_bound_to_owner_and_prompt(
    tmp_path: Path,
) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "consume.json")
    entity = PIIEntity(type="email", value="user@example.com", start=0, end=16)
    token_ref, approval_token = log_blocked_attempt(
        user_id="alice",
        prompt_hash="prompt-a",
        pii_entities=[entity],
        required_action="approval",
        registry=registry,
    )
    registry.approve(token_ref, operator="ops")

    assert not consume_approval(
        user_id="mallory",
        token_ref=token_ref,
        prompt_hash="prompt-a",
        approval_token=approval_token,
        registry=registry,
    )
    assert not consume_approval(
        user_id="alice",
        token_ref=token_ref,
        prompt_hash="prompt-b",
        approval_token=approval_token,
        registry=registry,
    )
    assert consume_approval(
        user_id="alice",
        token_ref=token_ref,
        prompt_hash="prompt-a",
        required_action="approval",
        approval_token=approval_token,
        registry=registry,
    )
    assert not consume_approval(
        user_id="alice",
        token_ref=token_ref,
        prompt_hash="prompt-a",
        required_action="approval",
        approval_token=approval_token,
        registry=registry,
    )
    request = registry.get(token_ref)
    assert request is not None
    assert request.status == "consumed"


def test_expired_rejected_and_revoked_requests_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = {"value": "2026-01-01T00:00:00+00:00"}
    monkeypatch.setattr(
        "monGARS.core.operator_approvals._utcnow_isoformat",
        lambda: now["value"],
    )
    registry = OperatorApprovalRegistry(tmp_path / "lifecycle.json")

    expired = registry.submit(
        source="security.guardrail",
        payload={"user_id": "alice", "prompt_hash": "expired"},
        expires_in_seconds=60,
    )
    now["value"] = "2026-01-01T00:02:00+00:00"
    with pytest.raises(ValueError, match="expired"):
        registry.approve(expired.request_id, operator="ops")

    rejected_request = registry.submit(source="tools", payload={"id": "reject"})
    rejected = registry.reject(
        rejected_request.request_id, operator="ops", notes="unsafe"
    )
    assert rejected.status == "rejected"

    revoked_request = registry.submit(source="tools", payload={"id": "revoke"})
    registry.approve(revoked_request.request_id, operator="ops")
    revoked = registry.revoke(revoked_request.request_id, operator="ops")
    assert revoked.status == "revoked"


def test_security_approval_without_expiry_and_malformed_expiry_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-expiry.json"
    registry = OperatorApprovalRegistry(path)
    legacy = registry.submit(
        source="security.guardrail",
        payload={"user_id": "alice", "prompt_hash": "legacy"},
    )
    malformed = registry.submit(
        source="tools",
        payload={"id": "malformed-expiry"},
        expires_in_seconds=60,
    )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    for item in persisted["requests"]:
        if item["id"] == malformed.request_id:
            item["expires_at"] = "not-a-timestamp"
    path.write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(ValueError, match="expired"):
        registry.approve(legacy.request_id, operator="ops")
    with pytest.raises(ValueError, match="expired"):
        registry.approve(malformed.request_id, operator="ops")


def test_existing_request_auto_approval_is_persisted(tmp_path: Path) -> None:
    path = tmp_path / "auto-existing.json"
    registry = OperatorApprovalRegistry(path)
    payload = {"metrics": {"accuracy": 0.95}}
    pending = registry.submit(source="reinforcement.reasoning", payload=payload)

    approved = registry.submit(
        source="reinforcement.reasoning",
        payload=payload,
        policy=lambda _: True,
    )
    reloaded = OperatorApprovalRegistry(path).get(pending.request_id)

    assert approved.is_approved
    assert reloaded is not None
    assert reloaded.is_approved


def test_policy_cannot_mutate_persisted_approval_payload(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "policy-mutation.json")
    payload = {"metrics": {"accuracy": 0.95}}

    def mutating_policy(candidate: dict) -> bool:
        candidate["metrics"]["accuracy"] = 0.0
        candidate["injected"] = True
        return True

    approved = registry.submit(
        source="reinforcement.reasoning",
        payload=payload,
        policy=mutating_policy,
    )

    assert approved.is_approved
    assert approved.payload == {"metrics": {"accuracy": 0.95}}


def test_public_registry_records_are_detached_from_authorization_state(
    tmp_path: Path,
) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "detached.json")
    submitted = registry.submit(
        source="security.guardrail",
        payload={
            "user_id": "alice",
            "prompt_hash": "prompt-a",
            "required_action": "financial_operation",
        },
        expires_in_seconds=60,
    )

    submitted.status = "approved"
    submitted.payload["user_id"] = "mallory"
    fetched = registry.get(submitted.request_id)
    assert fetched is not None
    fetched.status = "approved"
    pending = next(iter(registry.pending()))
    pending.status = "approved"

    stored = registry.get(submitted.request_id)
    assert stored is not None
    assert stored.is_pending
    assert stored.payload["user_id"] == "alice"
    assert not registry.consume(
        submitted.request_id,
        user_id="alice",
        prompt_hash="prompt-a",
        required_action="financial_operation",
    )


def test_action_bound_approval_requires_the_exact_action(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "action-required.json")
    request = registry.submit(
        source="security.guardrail",
        payload={
            "user_id": "alice",
            "prompt_hash": "prompt-a",
            "required_action": "financial_operation",
        },
        expires_in_seconds=60,
    )
    registry.approve(request.request_id, operator="ops")

    assert not registry.consume(
        request.request_id,
        user_id="alice",
        prompt_hash="prompt-a",
    )
    assert not registry.consume(
        request.request_id,
        user_id="alice",
        prompt_hash="prompt-a",
        required_action="personal_data_access",
    )
    assert registry.consume(
        request.request_id,
        user_id="alice",
        prompt_hash="prompt-a",
        required_action="financial_operation",
    )


def test_cross_instance_consume_is_one_shot(tmp_path: Path) -> None:
    path = tmp_path / "cross-instance.json"
    first = OperatorApprovalRegistry(path)
    request = first.submit(
        source="security.guardrail",
        payload={
            "user_id": "alice",
            "prompt_hash": "prompt-a",
            "required_action": "financial_operation",
        },
        expires_in_seconds=60,
    )
    first.approve(request.request_id, operator="ops")
    second = OperatorApprovalRegistry(path)

    assert first.consume(
        request.request_id,
        user_id="alice",
        prompt_hash="prompt-a",
        required_action="financial_operation",
    )
    assert not second.consume(
        request.request_id,
        user_id="alice",
        prompt_hash="prompt-a",
        required_action="financial_operation",
    )


def test_persist_failure_rolls_back_in_memory_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "rollback.json")
    pending = registry.submit(source="tools", payload={"id": "sensitive-action"})

    def fail_persist() -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(registry, "_persist", fail_persist)

    with pytest.raises(OSError, match="disk unavailable"):
        registry.approve(pending.request_id, operator="ops")

    restored = registry.get(pending.request_id)
    assert restored is not None
    assert restored.is_pending
    assert not restored.is_approved


@pytest.mark.parametrize(
    "persisted",
    [
        [],
        {"requests": "not-a-list"},
        {"requests": [{"id": "bad-payload", "payload": []}]},
        {"requests": [{"id": "bad-status", "payload": {}, "status": "authorised"}]},
    ],
)
def test_malformed_persisted_registry_fails_closed(
    tmp_path: Path, persisted: object
) -> None:
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(persisted), encoding="utf-8")

    registry = OperatorApprovalRegistry(path)

    assert registry.get("bad-payload") is None
    assert registry.get("bad-status") is None
    assert list(registry.pending()) == []


def test_blocked_context_redacts_secret_key_variants(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "redaction.json")
    entity = PIIEntity(type="email", value="user@example.com", start=0, end=16)

    token_ref, _ = log_blocked_attempt(
        user_id="alice",
        prompt_hash="deadbeef",
        pii_entities=[entity],
        required_action="approval",
        context={
            "api-key": "secret-api-key",
            "nested": {"client_secret": "secret-client-value"},
        },
        registry=registry,
    )
    stored = registry.get(token_ref)

    assert stored is not None
    snapshot = stored.payload["context_snapshot"]
    assert snapshot["api-key"] == "[REDACTED]"
    assert snapshot["nested"]["client_secret"] == "[REDACTED]"


def test_guard_returns_only_reference_and_consumes_approval_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from monGARS.core import operator_approvals as approvals_module

    monkeypatch.setattr(
        approvals_module,
        "_DEFAULT_APPROVALS_PATH",
        tmp_path / "guard.json",
    )
    monkeypatch.setattr(approvals_module, "_GLOBAL_REGISTRY", None)
    prompt = "My credit card is 4111-1111-1111-1111"
    context = {"user_id": "alice", "allowed_actions": ["financial_operation"]}

    blocked = pre_generation_guard(prompt, context)
    assert blocked is not None
    assert "approval_token" not in blocked
    token_ref = blocked["token_ref"]

    registry = approvals_module.get_operator_approval_registry()
    registry.approve(token_ref, operator="ops")
    assert pre_generation_guard(prompt, {**context, "token_ref": token_ref}) is None

    replay = pre_generation_guard(prompt, {**context, "token_ref": token_ref})
    assert replay is not None
    assert replay["token_ref"] != token_ref


def test_guard_approval_is_bound_to_sensitive_action_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from monGARS.core import operator_approvals as approvals_module

    monkeypatch.setattr(
        approvals_module,
        "_DEFAULT_APPROVALS_PATH",
        tmp_path / "action-scope.json",
    )
    monkeypatch.setattr(approvals_module, "_GLOBAL_REGISTRY", None)
    prompt = "My credit card is 4111-1111-1111-1111"
    personal_context = {
        "user_id": "alice",
        "allowed_actions": ["personal_data_access"],
    }

    blocked = pre_generation_guard(prompt, personal_context)
    assert blocked is not None
    token_ref = blocked["token_ref"]
    registry = approvals_module.get_operator_approval_registry()
    registry.approve(token_ref, operator="ops")

    substituted = pre_generation_guard(
        prompt,
        {
            "user_id": "alice",
            "allowed_actions": ["financial_operation"],
            "token_ref": token_ref,
        },
    )
    assert substituted is not None
    assert substituted["token_ref"] != token_ref

    assert (
        pre_generation_guard(prompt, {**personal_context, "token_ref": token_ref})
        is None
    )


def test_verify_approval_token_rejects_invalid_token(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "invalid-token.json")
    entity = PIIEntity(type="email", value="user@example.com", start=0, end=16)
    token_ref, approval_token = log_blocked_attempt(
        user_id="dave",
        prompt_hash="cafebabe",
        pii_entities=[entity],
        required_action="approval",
        context={"allowed_actions": ["personal_data_access"], "user_id": "dave"},
        registry=registry,
    )
    registry.approve(token_ref, operator="ops")

    assert not verify_approval_token(
        user_id="dave",
        token_ref=token_ref,
        approval_token="0" * len(approval_token),
        prompt_hash="cafebabe",
        registry=registry,
    )


def test_verify_approval_token_missing_request(tmp_path: Path) -> None:
    registry = OperatorApprovalRegistry(tmp_path / "missing-request.json")

    assert not verify_approval_token(
        user_id="erin",
        token_ref="does-not-exist",
        approval_token="0" * 64,
        prompt_hash="unknown",
        registry=registry,
    )
