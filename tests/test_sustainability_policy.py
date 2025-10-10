from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from modules.evolution_engine.sustainability import CarbonAwarePolicy


def _write_dashboard(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "sustainability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_policy_allows_when_no_telemetry(tmp_path: Path) -> None:
    policy = CarbonAwarePolicy(tmp_path / "missing.json")

    decision = policy.evaluate(scope="reinforcement")

    assert decision.should_proceed is True
    assert "policy clear" in decision.reason


def test_policy_blocks_high_carbon_intensity(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "energy_reports": [
            {
                "scope": "reinforcement.longhaul.cycle",
                "recorded_at": now.isoformat(),
                "energy_wh": 120.0,
                "carbon_intensity_g_co2_per_kwh": 720.0,
            }
        ]
    }
    path = _write_dashboard(tmp_path, payload)
    policy = CarbonAwarePolicy(
        path,
        carbon_pause_threshold=500.0,
        carbon_caution_threshold=300.0,
    )

    decision = policy.evaluate(scope="reinforcement")

    assert decision.should_proceed is False
    assert "exceeds pause threshold" in decision.reason
    assert decision.recommended_delay is not None


def test_policy_blocks_when_approvals_backlog(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "latest_reinforcement_summary": {
            "recorded_at": now.isoformat(),
            "approval_pending_final": 15,
            "incidents": [],
        }
    }
    path = _write_dashboard(tmp_path, payload)
    policy = CarbonAwarePolicy(path, approvals_threshold=10)

    decision = policy.evaluate(scope="reinforcement")

    assert decision.should_proceed is False
    assert "pending approvals" in decision.reason
