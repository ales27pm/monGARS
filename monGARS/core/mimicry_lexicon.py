"""Utilities for loading mimicry sentiment lexicons."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from monGARS.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "heureux",
        "heureuse",
        "ravi",
        "ravie",
        "content",
        "contente",
        "excellent",
        "excellente",
        "fantastique",
        "formidable",
        "super",
        "merci",
        "satisfait",
        "satisfaite",
        "positif",
        "positive",
        "agréable",
        "brillant",
        "génial",
    }
)
DEFAULT_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "triste",
        "furieux",
        "furieuse",
        "mauvais",
        "mauvaise",
        "terrible",
        "horrible",
        "déçu",
        "déçue",
        "problème",
        "problèmes",
        "mécontent",
        "mécontente",
        "négatif",
        "négative",
        "inquiet",
        "inquiète",
        "fâché",
        "fâchée",
    }
)


def _normalise_words(candidates: Iterable[object]) -> set[str]:
    """Convert an iterable of arbitrary objects into lowercase word tokens."""

    words: set[str] = set()
    for candidate in candidates:
        text = str(candidate).strip().lower()
        if text:
            words.add(text)
    return words


def _load_words_from_path(path: str | None) -> set[str]:
    """Load additional lexicon entries from the provided path."""

    if not path:
        return set()

    file_path = Path(path)
    if not file_path.exists():
        logger.warning("mimicry.lexicon.file_missing", extra={"path": str(file_path)})
        return set()

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem error path
        logger.error(
            "mimicry.lexicon.read_error",
            extra={"path": str(file_path), "error": str(exc)},
        )
        return set()

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error(
                "mimicry.lexicon.json_error",
                extra={"path": str(file_path), "error": str(exc)},
            )
            return set()
        if isinstance(parsed, dict):
            for key in ("words", "values", "lexicon", "terms"):
                if key in parsed and parsed[key]:
                    return _normalise_words(parsed[key])
            return _normalise_words(parsed.values())
        return _normalise_words(parsed if isinstance(parsed, list) else [parsed])

    return _normalise_words(line for line in content.splitlines())


@lru_cache(maxsize=1)
def get_sentiment_lexicon() -> tuple[frozenset[str], frozenset[str]]:
    """Return the configured positive and negative sentiment lexicons."""

    settings = get_settings()
    positive_words = set(DEFAULT_POSITIVE_WORDS)
    negative_words = set(DEFAULT_NEGATIVE_WORDS)

    positive_words.update(
        _load_words_from_path(getattr(settings, "mimicry_positive_lexicon_path", None))
    )
    negative_words.update(
        _load_words_from_path(getattr(settings, "mimicry_negative_lexicon_path", None))
    )

    overlap = positive_words & negative_words
    if overlap:
        logger.warning(
            "mimicry.lexicon.overlap",
            extra={"tokens": sorted(overlap)},
        )

    return frozenset(positive_words), frozenset(negative_words)
