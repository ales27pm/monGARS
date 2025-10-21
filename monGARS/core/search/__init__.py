"""Search orchestration and verification primitives."""

from __future__ import annotations

from .contracts import NormalizedHit, VerifiedBundle
from .orchestrator import SearchOrchestrator
from .schema_org import SchemaOrgMetadata, parse_schema_org
from .source_badge import source_badge
from .verifier import Verifier, cross_check

__all__ = [
    "NormalizedHit",
    "VerifiedBundle",
    "SearchOrchestrator",
    "SchemaOrgMetadata",
    "parse_schema_org",
    "source_badge",
    "Verifier",
    "cross_check",
]
