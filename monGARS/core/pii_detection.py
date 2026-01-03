"""Simple PII detection helpers used by security guardrails."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PIIEntity:
    """Detected PII snippet extracted from text."""

    type: str
    value: str
    start: int
    end: int


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    (
        "phone",
        re.compile(
            r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4})"
        ),
    ),
    ("payment", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    (
        "ip_address",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?!$)|$)){4}\b"),
    ),
    (
        "uuid",
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
    ),
)


def _scan_pattern(
    label: str, pattern: re.Pattern[str], text: str
) -> Iterable[PIIEntity]:
    for match in pattern.finditer(text):
        yield PIIEntity(
            type=label,
            value=match.group(0),
            start=match.start(),
            end=match.end(),
        )


def detect_pii(prompt: str) -> list[PIIEntity]:
    """Return detected PII entities from ``prompt``."""

    if not prompt:
        return []

    findings: list[PIIEntity] = []
    for label, pattern in _PATTERNS:
        findings.extend(_scan_pattern(label, pattern, prompt))

    if findings:
        logger.warning(
            "pii.detected",
            extra={
                "count": len(findings),
                "types": sorted({entity.type for entity in findings}),
            },
        )
    return findings


__all__: Sequence[str] = ("PIIEntity", "detect_pii")
