from __future__ import annotations

import pytest

from monGARS.core.search import NormalizedHit, Verifier


def _make_hit(snippet: str, *, provider: str = "gnews") -> NormalizedHit:
    return NormalizedHit(
        provider=provider,
        title="Headline",
        url=f"https://example.com/{hash(snippet)}",
        snippet=snippet,
        published_at=None,
        event_date=None,
        source_domain="example.com",
        lang="en",
        raw={},
    )


def test_verifier_builds_agreements() -> None:
    verifier = Verifier()
    hits = [
        _make_hit("World Health Organization reports 120 cases on 2024-12-01"),
        _make_hit("WHO confirms 120 cases recorded 2024-12-01"),
        _make_hit("Reuters: 119 cases expected"),
    ]

    bundle = verifier.cross_check("test", hits)

    assert bundle.agreed_facts.get("number") == "120"
    assert bundle.agreed_facts.get("date") == "2024-12-01"
    assert bundle.confidence >= 0.65
    assert bundle.primary_citation == hits[0].url
    assert bundle.citations == [hit.url for hit in hits]


def test_verifier_collects_disagreements() -> None:
    verifier = Verifier(minimum_agreement=2)
    hits = [
        _make_hit("local office reports 10 incidents"),
        _make_hit("regional office reports 12 incidents"),
    ]

    bundle = verifier.cross_check("incidents", hits)

    assert "number" not in bundle.agreed_facts
    assert bundle.confidence == pytest.approx(0.5)
    assert "number" in bundle.disagreements
    assert set(bundle.disagreements["number"]) == {"10", "12"}


def test_verifier_handles_empty_hits() -> None:
    verifier = Verifier()

    bundle = verifier.cross_check("empty", [])

    assert bundle.agreed_facts == {}
    assert bundle.disagreements == {}
    assert bundle.confidence == 0.0
    assert bundle.primary_citation is None
    assert bundle.citations == []


def test_verifier_handles_single_hit() -> None:
    verifier = Verifier()
    hit = _make_hit("Single source confirms 42 units sold on 2024-11-01")

    bundle = verifier.cross_check("single", [hit])

    assert bundle.primary_citation == hit.url
    assert bundle.citations == [hit.url]
    assert bundle.confidence == pytest.approx(1.0)
    assert isinstance(bundle.agreed_facts, dict)
