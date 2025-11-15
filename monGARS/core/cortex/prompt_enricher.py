"""Utilities for shaping verified research context into prompt-ready payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from monGARS.core.iris import IrisDocument
from monGARS.core.search import VerifiedBundle
from monGARS.core.search.source_badge import source_badge


UTC = datetime.UTC if hasattr(datetime, "UTC") else timezone.utc


def _fmt_dt(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _safe_authors(doc: IrisDocument) -> list[str] | None:
    authors = getattr(doc, "authors", None)
    if not authors:
        return None
    cleaned: list[str] = []
    for author in authors:
        if isinstance(author, str) and (stripped := author.strip()):
            cleaned.append(stripped)
    return cleaned or None


def build_context_snippet(
    doc: IrisDocument | None,
    bundle: VerifiedBundle,
) -> Mapping[str, Any]:
    """Return a structured dictionary ready for downstream prompt injection."""

    provider = bundle.hits[0].provider if bundle.hits else None
    badge_name, trust_level = source_badge(bundle.primary_citation or "", provider)

    metadata: dict[str, Any] = {}
    if doc is not None:
        metadata = {
            "title": getattr(doc, "title", None),
            "published_at": _fmt_dt(getattr(doc, "published_at", None)),
            "modified_at": _fmt_dt(getattr(doc, "modified_at", None)),
            "event_start": _fmt_dt(getattr(doc, "event_start", None)),
            "event_end": _fmt_dt(getattr(doc, "event_end", None)),
            "authors": _safe_authors(doc),
            "publisher": getattr(doc, "publisher", None),
            "organization": getattr(doc, "organization", None),
            "location_name": getattr(doc, "location_name", None),
            "badge": badge_name,
            "trust_level": trust_level,
            "url": getattr(doc, "url", None),
        }

    return {
        "snippet": doc.summary if doc and doc.summary else "",
        "metadata": metadata,
        "citations": bundle.citations[:8],
        "confidence": bundle.confidence,
        "agreed_facts": bundle.agreed_facts,
        "disagreements": bundle.disagreements,
        "primary_citation": bundle.primary_citation,
    }


__all__ = ["build_context_snippet"]
