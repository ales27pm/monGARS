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
_TRUST_WEIGHT_BONUS = 0.35


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
        primary: Optional[str] = None
        for hit in hits:
            if hit.url and hit.url not in seen:
                citations.append(hit.url)
                seen.add(hit.url)
            if primary is None and hit.url and hit.is_trustworthy():
                primary = hit.url
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
            field_values = self._extract_field_values(text)
            for field, values in field_values.items():
                for value in values:
                    buckets[field][value] += 1
        return buckets

    def _extract_field_values(self, text: str) -> Dict[str, List[str]]:
        values: Dict[str, List[str]] = {
            "entities": [],
            "dates": [],
            "numbers": [],
        }
        for match in _ENTITY_PATTERN.finditer(text):
            values["entities"].append(match.group(1))
        date_matches = list(_DATE_PATTERN.finditer(text))
        for match in date_matches:
            values["dates"].append(match.group(0))
        date_spans = [match.span(0) for match in date_matches]
        for match in _NUMBER_PATTERN.finditer(text):
            number_span = match.span(0)
            if any(
                span[0] <= number_span[0] < span[1]
                or span[0] < number_span[1] <= span[1]
                for span in date_spans
            ):
                continue
            values["numbers"].append(match.group(1))
        return values

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
            alternatives: List[str] = []
            for value, _ in counter.most_common():
                if value == agreed_value:
                    continue
                alternatives.append(value)
                if len(alternatives) == 3:
                    break
            if not alternatives:
                continue
            disagreements[alias] = alternatives
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
        weighted_total = 0.0
        weighted_agreement = 0.0
        seen_candidates: set[tuple[str, str, str]] = set()
        for hit in hits:
            text = hit.snippet or hit.title or ""
            if not text:
                continue
            field_values = self._extract_field_values(text)
            weight = 1.0 + (_TRUST_WEIGHT_BONUS if hit.is_trustworthy() else 0.0)
            source = hit.source_domain
            for field, values in field_values.items():
                if not values:
                    continue
                alias = _FIELD_ALIASES.get(field, field)
                for value in set(values):
                    key = (alias, source, value)
                    if key in seen_candidates:
                        continue
                    seen_candidates.add(key)
                    weighted_total += weight
                    if agreed.get(alias) == value:
                        weighted_agreement += weight
        if weighted_total == 0:
            return 0.5
        if len(hits) == 1:
            confidence = 0.5 + (weighted_agreement / max(weighted_total, 3.0))
            return round(min(confidence, 0.8), 3)
        confidence = 0.5 + (weighted_agreement / weighted_total)
        return round(min(confidence, 1.0), 3)
