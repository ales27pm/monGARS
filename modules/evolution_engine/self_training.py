"""Utilities bridging curated self-training datasets with the evolution engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - optional dependency at runtime
    from datasets import Dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dataset library is optional
    Dataset = None  # type: ignore[assignment]

from models.datasets.catalog import DatasetCatalog

logger = logging.getLogger(__name__)

DEFAULT_CURATED_ROOT = Path("models/datasets/curated")


def _iter_curated_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed curated records from a JSONL dataset file."""

    if not path.exists():
        logger.warning(
            "curated.dataset.missing",
            extra={"dataset_file": str(path)},
        )
        return

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    logger.debug(
                        "curated.dataset.invalid_record",
                        extra={"dataset_file": str(path)},
                        exc_info=True,
                    )
    except OSError as exc:  # pragma: no cover - defensive IO guard
        logger.error(
            "curated.dataset.read_error",
            extra={"dataset_file": str(path)},
            exc_info=exc,
        )
    return


def _extract_text(record: dict[str, Any]) -> str | None:
    """Derive the textual training payload from a curated record."""

    candidates: Sequence[str | None] = (
        record.get("text"),
        record.get("response"),
        record.get("prompt"),
        record.get("text_preview"),
    )
    for value in candidates:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return None


def collect_curated_data(
    *,
    dataset_root: str | Path | None = None,
    limit: int | None = None,
) -> Sequence[dict[str, Any]] | Any:
    """Load the most recent curated dataset prepared by the self-training engine."""

    root = Path(dataset_root) if dataset_root is not None else DEFAULT_CURATED_ROOT
    catalog = DatasetCatalog(root)
    latest = catalog.latest()
    if latest is None:
        logger.info("curated.dataset.unavailable", extra={"root": str(root)})
        return []

    records: list[dict[str, Any]] = []
    for record in _iter_curated_records(latest.dataset_file):
        text = _extract_text(record)
        if not text:
            continue
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"embedding", "vector", "tokens"}
        }
        cleaned = {"text": text, "metadata": metadata}
        records.append(cleaned)
        if limit is not None and len(records) >= limit:
            break

    if not records:
        logger.info(
            "curated.dataset.empty",
            extra={"dataset_file": str(latest.dataset_file)},
        )
        return []

    if Dataset is not None:  # pragma: no cover - exercised when datasets installed
        try:
            return Dataset.from_list(records)
        except Exception:  # pragma: no cover - dataset creation failure
            logger.exception("curated.dataset.hf_conversion_failed")

    return records


__all__ = ["collect_curated_data"]
