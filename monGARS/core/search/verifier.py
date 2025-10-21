"""Cross-source verification helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

from .contracts import NormalizedHit, VerifiedBundle

_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_DATE_PATTERN = re.compile(
    r"\b(20\d{2}|19\d{2})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b"
)
_NUMBER_PATTERN = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\b")
_FIELD_ALIASES = {"entities": "entity", "dates": "date", "numbers": "number"}


class Verifier:
    """Derive consensus facts from a set of search hits."""

    def __init__(self, *, minimum_agreement: int = 2) -> None:
        self._minimum_agreement = max(1, minimum_agreement)

    def cross_check(self, query: str, hits: Sequence[NormalizedHit]) -> VerifiedBundle:
        facts = self._extract_claims((hit.snippet or hit.title or "") for hit in hits)
        agreed = self._select_agreements(facts)
        disagreements = self._collect_disagreements(facts, agreed)
        confidence = self._calculate_confidence(hits, facts, agreed)
        seen: set[str] = set()
        citations: List[str] = []
        for hit in hits:
            if hit.url and hit.url not in seen:
                citations.append(hit.url)
                seen.add(hit.url)
        primary: Optional[str] = None
        for hit in hits:
            if hit.url and hit.is_trustworthy():
                primary = hit.url
                break
        if primary is None and hits:
            primary = hits[0].url or None
        return VerifiedBundle(
            query=query,
            hits=list(hits),
            agreed_facts=agreed,
            disagreements=disagreements,
            confidence=confidence,
            primary_citation=primary,
            citations=citations,
        )

    def _extract_claims(self, texts: Iterable[str]) -> Dict[str, Counter[str]]:
        buckets: Dict[str, Counter[str]] = {
            "entities": Counter(),
            "dates": Counter(),
            "numbers": Counter(),
        }
        for text in texts:
            if not text:
                continue
            for match in _ENTITY_PATTERN.finditer(text):
                buckets["entities"][match.group(1)] += 1
            for match in _DATE_PATTERN.finditer(text):
                buckets["dates"][match.group(0)] += 1
            for match in _NUMBER_PATTERN.finditer(text):
                buckets["numbers"][match.group(1)] += 1
        return buckets

    def _select_agreements(self, buckets: Dict[str, Counter[str]]) -> Dict[str, str]:
        agreed: Dict[str, str] = {}
        for key, counter in buckets.items():
            if not counter:
                continue
            value, count = counter.most_common(1)[0]
            if count >= self._minimum_agreement:
                alias = _FIELD_ALIASES.get(key, key)
                agreed[alias] = value
        return agreed

    def _collect_disagreements(
        self, buckets: Dict[str, Counter[str]], agreed: Dict[str, str]
    ) -> Dict[str, List[str]]:
        disagreements: Dict[str, List[str]] = {}
        for key, counter in buckets.items():
            if not counter:
                continue
            alias = _FIELD_ALIASES.get(key, key)
            agreed_value = agreed.get(alias)
            alternatives = [
                value for value, _ in counter.most_common(3) if value != agreed_value
            ]
            if not alternatives:
                continue
            existing = disagreements.get(alias, [])
            for value in alternatives:
                if value not in existing:
                    existing.append(value)
            disagreements[alias] = existing
        return disagreements

    def _calculate_confidence(
        self,
        hits: Sequence[NormalizedHit],
        buckets: Dict[str, Counter[str]],
        agreed: Dict[str, str],
    ) -> float:
        if not hits:
            return 0.0
        total_candidates = sum(sum(counter.values()) for counter in buckets.values())
        if total_candidates == 0:
            return 0.5
        if len(hits) == 1:
            confidence = 0.5 + (min(len(agreed), 1) / max(total_candidates, 3))
            return round(min(confidence, 1.0), 3)
        confidence = 0.5 + (len(agreed) / total_candidates)
        return round(min(confidence, 1.0), 3)
