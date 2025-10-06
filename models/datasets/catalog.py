"""Version catalog management for curated fine-tuning datasets."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetVersion:
    """Metadata describing a curated dataset export."""

    version: int
    run_id: str
    created_at: datetime
    dataset_dir: Path
    dataset_file: Path
    record_count: int
    governance: Mapping[str, Any] | None = None
    compliance: Mapping[str, Any] | None = None
    quarantined: bool = False
    extra: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "run_id": self.run_id,
            "created_at": self.created_at.replace(tzinfo=UTC).isoformat(),
            "dataset_dir": str(self.dataset_dir),
            "dataset_file": str(self.dataset_file),
            "record_count": self.record_count,
        }
        if self.governance:
            payload["governance"] = dict(self.governance)
        if self.compliance:
            payload["compliance"] = dict(self.compliance)
        payload["quarantined"] = bool(self.quarantined)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


class DatasetCatalog:
    """Maintain a JSON index of available curated datasets."""

    def __init__(self, root: Path, *, catalog_name: str = "catalog.json") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / catalog_name
        self._cache: dict[str, Any] | None = None

    def register(
        self,
        *,
        run_id: str,
        dataset_dir: Path,
        dataset_file: Path,
        record_count: int,
        created_at: datetime | None = None,
        governance: Mapping[str, Any] | None = None,
        compliance: Mapping[str, Any] | None = None,
        quarantined: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        catalog = self._load()
        next_version = int(catalog.get("latest_version", 0)) + 1
        created = created_at or datetime.now(UTC)
        version = DatasetVersion(
            version=next_version,
            run_id=run_id,
            created_at=created,
            dataset_dir=dataset_dir,
            dataset_file=dataset_file,
            record_count=record_count,
            governance=governance,
            compliance=compliance,
            quarantined=bool(quarantined) if quarantined is not None else False,
            extra=extra,
        )
        catalog.setdefault("versions", []).append(version.as_dict())
        catalog["latest_version"] = version.version
        catalog.setdefault("index", {})[run_id] = version.version
        self._write_catalog(catalog)
        log.info(
            "dataset_version_registered",
            extra={
                "run_id": run_id,
                "version": version.version,
                "records": record_count,
                "quarantined": version.quarantined,
            },
        )
        return version

    def latest(self) -> DatasetVersion | None:
        catalog = self._load()
        versions = catalog.get("versions") or []
        if not versions:
            return None
        payload = versions[-1]
        return DatasetVersion(
            version=int(payload["version"]),
            run_id=str(payload["run_id"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            dataset_dir=Path(payload["dataset_dir"]),
            dataset_file=Path(payload["dataset_file"]),
            record_count=int(payload["record_count"]),
            governance=payload.get("governance"),
            compliance=payload.get("compliance"),
            quarantined=bool(payload.get("quarantined", False)),
            extra=payload.get("extra"),
        )

    def _load(self) -> dict[str, Any]:
        if self._cache is not None:
            return dict(self._cache)
        if not self.catalog_path.exists():
            self._cache = {"versions": [], "latest_version": 0, "index": {}}
            return dict(self._cache)
        try:
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse dataset catalog: %s", exc)
            data = {"versions": [], "latest_version": 0, "index": {}}
        self._cache = data
        return dict(data)

    def _write_catalog(self, catalog: Mapping[str, Any]) -> None:
        tmp_path = self.catalog_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp_path.replace(self.catalog_path)
        self._cache = dict(catalog)


__all__ = ["DatasetCatalog", "DatasetVersion"]
