"""Derive human-friendly trust badges for search results."""

from __future__ import annotations

import re
from typing import Tuple
from urllib.parse import urlparse

_BADGE_RULES: tuple[tuple[re.Pattern[str], tuple[str, str]], ...] = (
    (re.compile(r"\.gov(\.|$)"), ("Government", "trust-high")),
    (re.compile(r"\.edu(\.|$)"), ("Academic", "trust-high")),
    (re.compile(r"\.gouv\."), ("Government", "trust-high")),
    (re.compile(r"wikipedia\.org$"), ("Reference", "trust-medium")),
    (
        re.compile(r"(reuters|bbc|apnews|nytimes|cnn|lemonde|guardian)\."),
        ("News", "trust-medium"),
    ),
    (
        re.compile(r"(nature|science|arxiv|crossref|pubmed)\."),
        ("Scientific", "trust-high"),
    ),
    (re.compile(r"(snopes|politifact)\."), ("Fact-Check", "trust-high")),
    (re.compile(r"(blogspot|medium|substack)\."), ("Blog", "trust-low")),
)

_PROVIDER_BADGES: dict[str, tuple[str, str]] = {
    "crossref": ("Scientific", "trust-high"),
    "pubmed": ("Scientific", "trust-high"),
    "arxiv": ("Scientific", "trust-high"),
    "gnews": ("News", "trust-medium"),
    "ddg": ("Search", "trust-medium"),
    "wikipedia": ("Reference", "trust-medium"),
}


def _normalise_host(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    return parsed.netloc.lower()


def source_badge(url: str, provider: str | None = None) -> Tuple[str, str]:
    """Return a tuple of ``(badge_name, trust_level)`` for *url*.

    Domain patterns take precedence, followed by provider-based fallbacks. If
    nothing matches, the result defaults to ``("Unclassified", "trust-low")``.
    """

    host = _normalise_host(url)
    for pattern, badge in _BADGE_RULES:
        if host and pattern.search(host):
            return badge

    provider_key = (provider or "").strip().lower()
    if provider_key:
        badge = _PROVIDER_BADGES.get(provider_key)
        if badge is not None:
            return badge

    return ("Unclassified", "trust-low")


__all__ = ["source_badge"]
