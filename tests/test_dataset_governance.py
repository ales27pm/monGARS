from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from models.datasets.governance import DatasetGovernance
from monGARS.config import Settings

UTC = getattr(datetime, "UTC", timezone.utc)


def _build_settings(**overrides: object) -> Settings:
    base = Settings(SECRET_KEY="unit-test-secret")
    return base.model_copy(update=overrides)


def test_dataset_governance_approves_clean_dataset(tmp_path: Path) -> None:
    settings = _build_settings(
        rag_curated_default_provenance="unit-test",
        rag_curated_reviewer="compliance-bot",
        rag_curated_default_tags=["rag", "unit-test"],
    )
    governance = DatasetGovernance(settings=settings)
    dataset_file = tmp_path / "curated.jsonl"
    dataset_file.write_text(
        json.dumps({"text": "sanitised", "text_preview": "sanitised"}) + "\n",
        encoding="utf-8",
    )

    evaluation = governance.evaluate_dataset(
        dataset_file,
        run_id="run-1",
        record_count=1,
        created_at=datetime.now(UTC) - timedelta(minutes=1),
    )

    assert evaluation.status == "approved"
    assert not evaluation.violations
    assert evaluation.metadata["provenance"] == "unit-test"
    assert evaluation.metadata["reviewed_by"] == "compliance-bot"


def test_dataset_governance_flags_pii(tmp_path: Path) -> None:
    settings = _build_settings(
        rag_curated_default_provenance="unit-test",
        rag_curated_reviewer="qa",
        rag_curated_retention_days=1,
        rag_curated_export_window_days=1,
    )
    governance = DatasetGovernance(settings=settings)
    dataset_file = tmp_path / "curated.jsonl"
    dataset_file.write_text(
        json.dumps({"text": "Contact me at test@example.com"}) + "\n",
        encoding="utf-8",
    )

    evaluation = governance.evaluate_dataset(
        dataset_file,
        run_id="run-2",
        record_count=1,
        created_at=datetime.now(UTC) - timedelta(days=2),
    )

    assert evaluation.status == "quarantined"
    assert any(v.code == "pii_detected" for v in evaluation.violations)
    assert any(v.code == "dataset_expired" for v in evaluation.violations)
    packed = evaluation.as_dict()
    assert packed["status"] == "quarantined"
    assert packed["metadata"]["run_id"] == "run-2"
