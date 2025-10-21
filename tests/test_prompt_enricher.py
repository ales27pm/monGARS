from datetime import datetime, timezone

from monGARS.core.cortex.prompt_enricher import build_context_snippet
from monGARS.core.iris import IrisDocument
from monGARS.core.search import NormalizedHit, VerifiedBundle


def _make_hit(url: str, provider: str = "ddg") -> NormalizedHit:
    return NormalizedHit(
        provider=provider,
        title="",
        url=url,
        snippet="",
        published_at=None,
        event_date=None,
        source_domain="",
        lang="en",
        raw={},
    )


def _make_bundle(url: str) -> VerifiedBundle:
    hit = _make_hit(url)
    return VerifiedBundle(
        query="test",
        hits=[hit],
        agreed_facts={"entities": "Europa Clipper"},
        disagreements={},
        confidence=0.94,
        primary_citation=url,
        citations=[url, "https://secondary.example.com"],
    )


def test_build_context_snippet_renders_metadata():
    doc = IrisDocument(
        url="https://www.nasa.gov/europa-clipper",
        text="Europa Clipper text",
        title="Europa Clipper Mission Overview",
        summary="Europa Clipper will launch in 2025.",
    )
    doc.published_at = datetime(2025, 10, 19, 14, 5, tzinfo=timezone.utc)
    doc.modified_at = datetime(2025, 10, 20, 10, 0, tzinfo=timezone.utc)
    doc.event_start = datetime(2025, 10, 19, 14, 5, tzinfo=timezone.utc)
    doc.event_end = datetime(2025, 10, 20, 12, 0, tzinfo=timezone.utc)
    doc.authors = ["NASA Science", ""]
    doc.publisher = "NASA"
    doc.organization = "Jet Propulsion Laboratory"
    doc.location_name = "Cape Canaveral, FL"

    bundle = _make_bundle(doc.url)
    result = build_context_snippet(doc, bundle)

    assert result["snippet"] == doc.summary
    metadata = result["metadata"]
    assert metadata["badge"] == "Government"
    assert metadata["trust_level"] == "trust-high"
    assert metadata["authors"] == ["NASA Science"]
    assert metadata["published_at"] == "2025-10-19T14:05:00+00:00"
    assert metadata["modified_at"] == "2025-10-20T10:00:00+00:00"
    assert metadata["event_start"] == "2025-10-19T14:05:00+00:00"
    assert metadata["event_end"] == "2025-10-20T12:00:00+00:00"
    assert metadata["publisher"] == "NASA"
    assert metadata["organization"] == "Jet Propulsion Laboratory"
    assert metadata["location_name"] == "Cape Canaveral, FL"
    assert result["citations"] == bundle.citations[:8]
    assert result["confidence"] == bundle.confidence


def test_build_context_snippet_handles_missing_document():
    bundle = _make_bundle("https://www.nasa.gov/europa-clipper")
    result = build_context_snippet(None, bundle)
    assert result["metadata"] == {}
    assert result["snippet"] == ""
