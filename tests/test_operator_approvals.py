from __future__ import annotations

from pathlib import Path

from monGARS.core.operator_approvals import OperatorApprovalRegistry


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
