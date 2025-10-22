from datetime import datetime, timezone

from monGARS.core.search.contracts import NormalizedHit
from monGARS.core.search.verifier import (
    agreement_score,
    cross_check,
    extract_claims,
)


def _hit(provider: str, title_snippet: str) -> NormalizedHit:
    return NormalizedHit(
        provider=provider,
        title=title_snippet,
        url=f"https://{provider}.example/item",
        snippet="",
        published_at=datetime(2025, 10, 19, tzinfo=timezone.utc),
        event_date=None,
        source_domain=f"{provider}.example",
        lang="en",
        raw={},
    )


def test_extract_and_agreement():
    texts = [
        "OpenAI DevDay on 2025-11-12 in San Francisco",
        "OpenAI DevDay scheduled 2025-11-12",
        "OpenAI conference 2025-11-13",
    ]
    counts = extract_claims(texts)
    agreed, conf = agreement_score(counts, k_min=2)
    entity_value = agreed.get("entities") or agreed.get("entity")
    assert entity_value in {"OpenAI", "OpenAI DevDay"}
    date_value = agreed.get("dates") or agreed.get("date")
    assert date_value in {"2025-11-12"}
    assert 0.5 <= conf <= 1.0


def test_cross_check_bundle():
    hits = [
        _hit("gnews", "OpenAI DevDay 2025-11-12"),
        _hit("wikipedia", "OpenAI DevDay 2025-11-12"),
        _hit("ddg", "OpenAI DevDay 2025-11-13"),
    ]
    bundle = cross_check("devday", hits)
    assert bundle.primary_citation.startswith("https://")
    assert bundle.citations and len(bundle.citations) == 3
    assert bundle.confidence >= 0.6
