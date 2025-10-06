"""Governance helpers for curated Retrieval-Augmented Generation datasets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from monGARS.config import Settings, get_settings

from .sanitizer import detect_pii

log = logging.getLogger(__name__)


@dataclass(slots=True)
class GovernanceViolation:
    """Represents a policy violation detected during dataset evaluation."""

    code: str
    message: str
    details: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(slots=True)
class GovernanceEvaluation:
    """Result of evaluating a curated dataset against governance policies."""

    status: str
    metadata: Mapping[str, Any]
    checked_at: datetime
    violations: list[GovernanceViolation]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metadata": dict(self.metadata),
            "checked_at": self.checked_at.replace(tzinfo=UTC).isoformat(),
            "violations": [violation.as_dict() for violation in self.violations],
        }


class DatasetGovernance:
    """Evaluate curated datasets for retention, export, and sanitisation."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._retention_days = max(1, int(self._settings.rag_curated_retention_days))
        self._export_window_days = max(
            0, int(self._settings.rag_curated_export_window_days)
        )
        self._provenance = self._settings.rag_curated_default_provenance
        self._sensitivity = self._settings.rag_curated_default_sensitivity
        self._reviewer = self._settings.rag_curated_reviewer
        self._default_tags: tuple[str, ...] = tuple(
            self._settings.rag_curated_default_tags
        )

    def build_metadata(
        self,
        *,
        run_id: str,
        record_count: int,
        created_at: datetime,
    ) -> dict[str, Any]:
        expires_at = created_at + timedelta(days=self._retention_days)
        export_window_end = (
            created_at + timedelta(days=self._export_window_days)
            if self._export_window_days
            else None
        )
        metadata: dict[str, Any] = {
            "run_id": run_id,
            "record_count": record_count,
            "provenance": self._provenance,
            "sensitivity": self._sensitivity,
            "retention_days": self._retention_days,
            "reviewed_by": self._reviewer,
            "reviewed_at": created_at.replace(tzinfo=UTC).isoformat(),
            "expires_at": expires_at.replace(tzinfo=UTC).isoformat(),
            "export_window_days": self._export_window_days,
            "tags": list(self._default_tags),
        }
        if export_window_end is not None:
            metadata["export_window_ends_at"] = export_window_end.replace(
                tzinfo=UTC
            ).isoformat()
        return metadata

    def evaluate_dataset(
        self,
        dataset_file: Path,
        *,
        run_id: str,
        record_count: int,
        created_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GovernanceEvaluation:
        """Evaluate ``dataset_file`` and return governance metadata."""

        evaluation_time = datetime.now(UTC)
        created = created_at or evaluation_time
        base_metadata = (
            dict(metadata)
            if metadata is not None
            else self.build_metadata(
                run_id=run_id, record_count=record_count, created_at=created
            )
        )
        violations: list[GovernanceViolation] = []

        if not base_metadata.get("reviewed_by"):
            violations.append(
                GovernanceViolation(
                    code="missing_reviewer",
                    message="Curated dataset metadata is missing an assigned reviewer.",
                )
            )

        expires_at_raw = base_metadata.get("expires_at")
        if expires_at_raw:
            try:
                expires_at = datetime.fromisoformat(str(expires_at_raw))
                if expires_at <= evaluation_time:
                    violations.append(
                        GovernanceViolation(
                            code="dataset_expired",
                            message="Curated dataset has exceeded its retention window.",
                            details={"expired_at": expires_at_raw},
                        )
                    )
            except ValueError:
                violations.append(
                    GovernanceViolation(
                        code="invalid_expiry",
                        message="Curated dataset metadata contains an invalid expiry timestamp.",
                        details={"expires_at": expires_at_raw},
                    )
                )

        export_window_raw = base_metadata.get("export_window_ends_at")
        if export_window_raw:
            try:
                export_deadline = datetime.fromisoformat(str(export_window_raw))
                if export_deadline <= evaluation_time:
                    violations.append(
                        GovernanceViolation(
                            code="export_window_elapsed",
                            message=(
                                "Export window elapsed; dataset must be re-reviewed "
                                "before sharing externally."
                            ),
                            details={"export_window_ends_at": export_window_raw},
                        )
                    )
            except ValueError:
                violations.append(
                    GovernanceViolation(
                        code="invalid_export_window",
                        message="Curated dataset metadata contains an invalid export window timestamp.",
                        details={"export_window_ends_at": export_window_raw},
                    )
                )

        if record_count <= 0:
            violations.append(
                GovernanceViolation(
                    code="empty_dataset",
                    message="Curated dataset does not contain any records.",
                )
            )

        actual_count = 0
        if not dataset_file.exists():
            violations.append(
                GovernanceViolation(
                    code="dataset_missing",
                    message="Curated dataset file is missing.",
                    details={"path": str(dataset_file)},
                )
            )
        else:
            with dataset_file.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    actual_count += 1
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        violations.append(
                            GovernanceViolation(
                                code="invalid_record",
                                message="Dataset contains an invalid JSON record.",
                                details={
                                    "line": line_number,
                                    "error": str(exc),
                                },
                            )
                        )
                        continue
                    pii_hits = detect_pii(record)
                    if pii_hits:
                        violations.append(
                            GovernanceViolation(
                                code="pii_detected",
                                message="Dataset record still contains identifiable information.",
                                details={"line": line_number, "matches": pii_hits},
                            )
                        )
            if actual_count and actual_count != record_count:
                violations.append(
                    GovernanceViolation(
                        code="count_mismatch",
                        message="Catalog record count does not match dataset contents.",
                        details={"expected": record_count, "actual": actual_count},
                    )
                )

        status = "approved" if not violations else "quarantined"
        if status == "quarantined":
            log.warning(
                "curated.dataset.governance_failed",
                extra={
                    "run_id": run_id,
                    "violations": [violation.code for violation in violations],
                },
            )
        else:
            log.info(
                "curated.dataset.governance_passed",
                extra={"run_id": run_id, "record_count": record_count},
            )

        return GovernanceEvaluation(
            status=status,
            metadata=base_metadata,
            checked_at=evaluation_time,
            violations=violations,
        )


__all__ = [
    "DatasetGovernance",
    "GovernanceEvaluation",
    "GovernanceViolation",
]
