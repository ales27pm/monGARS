"""Search orchestration and verification primitives."""

from __future__ import annotations

from .contracts import NormalizedHit, VerifiedBundle
from .orchestrator import SearchOrchestrator, domain_weight, recency_weight
from .schema_org import SchemaOrgMetadata, parse_schema_org
from .source_badge import source_badge
from .verifier import Verifier, agreement_score, cross_check, extract_claims

__all__ = [
    "NormalizedHit",
    "VerifiedBundle",
    "SearchOrchestrator",
    "domain_weight",
    "recency_weight",
    "SchemaOrgMetadata",
    "parse_schema_org",
    "source_badge",
    "Verifier",
    "extract_claims",
    "agreement_score",
    "cross_check",
]
