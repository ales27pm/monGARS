"""Utilities for removing personally identifiable information from datasets."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

log = logging.getLogger(__name__)

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(
    r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4})"
)
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_IP_PATTERN = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?!$)|$)){4}\b")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_EMAIL_PATTERN, "[REDACTED_EMAIL]"),
    (_PHONE_PATTERN, "[REDACTED_PHONE]"),
    (_CREDIT_CARD_PATTERN, "[REDACTED_PAYMENT]"),
    (_IP_PATTERN, "[REDACTED_IP]"),
    (_UUID_PATTERN, "[REDACTED_UUID]"),
)


def scrub_text(text: str) -> str:
    """Return ``text`` with known PII patterns redacted."""

    cleaned = text
    total_replacements = 0
    for pattern, replacement in _REPLACEMENTS:
        cleaned, replacements = pattern.subn(replacement, cleaned)
        total_replacements += replacements
    if total_replacements:
        log.debug(
            "pii_redacted",
            extra={"replacements": total_replacements, "original_length": len(text)},
        )
    return cleaned


def sanitize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively sanitize mappings so that nested strings are PII-free."""

    sanitized: dict[str, Any] = {}
    for key, value in record.items():
        sanitized[key] = _sanitize_value(value)
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, Mapping):
        return sanitize_record(value)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        sanitized_items = []
        for item in value:
            if isinstance(item, (str, Mapping)):
                sanitized_items.append(_sanitize_value(item))
            else:
                sanitized_items.append(item)
        if isinstance(value, tuple):
            return tuple(sanitized_items)
        return sanitized_items
    return value


def detect_pii(record: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return locations of values still matching PII patterns.

    The function mirrors :func:`sanitize_record` by walking nested mappings and
    iterables. Rather than mutating the payload it records the replacement
    tokens that would be applied if sanitisation were to run again. The result
    maps dotted key paths (or indices for lists) to the redaction markers that
    triggered.
    """

    findings: dict[str, set[str]] = {}

    def _walk(value: Any, path: list[str]) -> None:
        if isinstance(value, str):
            matches = [
                replacement
                for pattern, replacement in _REPLACEMENTS
                if pattern.search(value)
            ]
            if matches:
                key = ".".join(path) if path else "<root>"
                findings.setdefault(key, set()).update(matches)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                path.append(str(key))
                _walk(item, path)
                path.pop()
            return
        if isinstance(value, Iterable) and not isinstance(
            value, (bytes, bytearray, str)
        ):
            for index, item in enumerate(value):
                path.append(str(index))
                _walk(item, path)
                path.pop()

    _walk(record, [])
    return {key: sorted(labels) for key, labels in findings.items()}


__all__ = ["sanitize_record", "scrub_text", "detect_pii"]
