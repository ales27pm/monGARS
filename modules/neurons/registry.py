"""Utilities for tracking LLM2Vec adapter artifacts produced by the evolution engine."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "adapter_manifest.json"
_HISTORY_LIMIT = 10


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _compute_checksum(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        logger.debug("Weights file not found for checksum", extra={"path": str(path)})
        return None
    except OSError as exc:  # pragma: no cover - unexpected IO failure
        logger.warning("Unable to read weights for checksum: %s", exc)
        return None
    return hashlib.sha256(data).hexdigest()


def _ensure_relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _normalise_path(value: str | os.PathLike[str] | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = path.resolve()
    return path


@dataclass(slots=True)
class AdapterRecord:
    """Entry describing a single trained adapter."""

    relative_adapter_path: str
    relative_weights_path: str | None
    status: str
    summary: dict[str, Any]
    created_at: str
    weights_checksum: str | None = None

    @property
    def version(self) -> str:
        """Return a stable identifier for the adapter version."""

        return self.weights_checksum or self.created_at

    def resolve_adapter_path(self, registry_path: Path) -> Path:
        path = Path(self.relative_adapter_path)
        if not path.is_absolute():
            path = registry_path / path
        return path.resolve(strict=False)

    def resolve_weights_path(self, registry_path: Path) -> Path | None:
        if not self.relative_weights_path:
            return None
        path = Path(self.relative_weights_path)
        if not path.is_absolute():
            path = registry_path / path
        return path.resolve(strict=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_adapter_path": self.relative_adapter_path,
            "relative_weights_path": self.relative_weights_path,
            "status": self.status,
            "summary": self.summary,
            "created_at": self.created_at,
            "weights_checksum": self.weights_checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterRecord":
        return cls(
            relative_adapter_path=str(data.get("relative_adapter_path")),
            relative_weights_path=data.get("relative_weights_path"),
            status=str(data.get("status", "")),
            summary=dict(data.get("summary", {})),
            created_at=str(data.get("created_at", "")),
            weights_checksum=data.get("weights_checksum"),
        )


@dataclass(slots=True)
class AdapterManifest:
    """Manifest describing available adapters and the active selection."""

    registry_path: Path
    path: Path
    current: AdapterRecord | None = None
    history: list[AdapterRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_dict() if self.current else None,
            "history": [record.to_dict() for record in self.history],
            "updated_at": _now_iso(),
        }

    def write(self) -> None:
        payload = self.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info(
            "adapter.manifest.updated",
            extra={
                "path": str(self.path),
                "active_version": self.current.version if self.current else None,
            },
        )

    def build_payload(self) -> dict[str, str]:
        """Return metadata required by remote inference services."""

        if not self.current:
            return {}
        adapter_path = self.current.resolve_adapter_path(self.registry_path)
        weights_path = self.current.resolve_weights_path(self.registry_path)
        payload = {
            "adapter_path": adapter_path.as_posix(),
            "version": self.current.version,
            "updated_at": self.current.created_at,
            "status": self.current.status,
        }
        if weights_path is not None:
            payload["weights_path"] = weights_path.as_posix()
        return payload

    @classmethod
    def from_dict(
        cls, registry_path: Path, manifest_path: Path, data: dict[str, Any]
    ) -> "AdapterManifest":
        current_data = data.get("current")
        history_data = data.get("history", [])
        history: Iterable[dict[str, Any]]
        if isinstance(history_data, list):
            history = history_data
        else:
            history = []
        current = AdapterRecord.from_dict(current_data) if current_data else None
        return cls(
            registry_path=registry_path,
            path=manifest_path,
            current=current,
            history=[AdapterRecord.from_dict(item) for item in history],
        )


def load_manifest(registry_path: str | Path) -> AdapterManifest | None:
    registry = Path(registry_path)
    manifest_path = registry / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        logger.error("Invalid adapter manifest JSON: %s", exc)
        raise
    return AdapterManifest.from_dict(registry, manifest_path, data)


def update_manifest(
    registry_path: str | Path,
    summary: dict[str, Any],
    *,
    history_limit: int = _HISTORY_LIMIT,
) -> AdapterManifest:
    """Persist a manifest entry for the latest adapter summary."""

    registry = Path(registry_path)
    registry.mkdir(parents=True, exist_ok=True)
    manifest_path = registry / MANIFEST_FILENAME

    artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    adapter_dir = _normalise_path(artifacts.get("adapter"))
    if adapter_dir is None:
        raise ValueError("Training summary missing 'artifacts.adapter' path")
    weights_value = artifacts.get("weights")
    weights_path: Path | None = None
    if weights_value:
        weights_candidate = Path(weights_value)
        if not weights_candidate.is_absolute():
            weights_candidate = Path(adapter_dir) / weights_candidate
        weights_path = _normalise_path(weights_candidate)

    relative_adapter_path = _ensure_relative(adapter_dir, registry)
    relative_weights_path = (
        _ensure_relative(weights_path, registry) if weights_path is not None else None
    )
    checksum = _compute_checksum(weights_path) if weights_path else None

    record = AdapterRecord(
        relative_adapter_path=relative_adapter_path,
        relative_weights_path=relative_weights_path,
        status=str(summary.get("status", "")),
        summary=dict(summary),
        created_at=_now_iso(),
        weights_checksum=checksum,
    )

    manifest = load_manifest(registry)
    if manifest is None:
        manifest = AdapterManifest(
            registry_path=registry, path=manifest_path, current=record, history=[]
        )
    else:
        if manifest.current:
            manifest.history.insert(0, manifest.current)
        manifest.history = manifest.history[:history_limit]
        manifest.current = record
        manifest.path = manifest_path

    manifest.write()
    _update_latest_symlink(registry, adapter_dir)
    return manifest


def _update_latest_symlink(registry: Path, adapter_dir: Path) -> None:
    link_path = registry / "latest"
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(adapter_dir, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform dependent
        logger.debug("Unable to refresh adapter symlink: %s", exc)


__all__ = [
    "AdapterManifest",
    "AdapterRecord",
    "MANIFEST_FILENAME",
    "load_manifest",
    "update_manifest",
]
