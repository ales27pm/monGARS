"""Helpers for working with embedding backend configuration."""

from __future__ import annotations

import logging

from monGARS.core.constants import (
    DEFAULT_EMBEDDING_BACKEND,
    SUPPORTED_EMBEDDING_BACKENDS,
)

LOGGER = logging.getLogger("monGARS.core.embedding_backends")


def normalise_embedding_backend(
    candidate: str | None,
    *,
    default: str = DEFAULT_EMBEDDING_BACKEND,
    strict: bool = False,
    logger: logging.Logger | None = None,
    log_event: str = "llm2vec.embedding.backend.unsupported",
) -> str:
    """Normalise a backend identifier and optionally enforce strict validation."""

    if candidate is None:
        return default

    normalised = str(candidate).strip().lower()
    if normalised in SUPPORTED_EMBEDDING_BACKENDS:
        return normalised

    if strict:
        options = ", ".join(sorted(SUPPORTED_EMBEDDING_BACKENDS))
        raise ValueError(f"embedding_backend must be one of: {options}")

    active_logger = logger or LOGGER
    active_logger.warning(
        log_event,
        extra={
            "backend": candidate,
            "supported_backends": sorted(SUPPORTED_EMBEDDING_BACKENDS),
        },
    )
    return default


__all__ = [
    "DEFAULT_EMBEDDING_BACKEND",
    "SUPPORTED_EMBEDDING_BACKENDS",
    "normalise_embedding_backend",
]

