"""Shared constants for monGARS core components."""

from __future__ import annotations

SUPPORTED_EMBEDDING_BACKENDS: frozenset[str] = frozenset(
    {"huggingface", "ollama", "transformers"}
)
"""Backends available for LLM2Vec embedding generation."""


DEFAULT_EMBEDDING_BACKEND = "huggingface"
"""Fallback backend used when no supported backend is provided."""
