from monGARS.core.search.source_badge import source_badge


def test_source_badge_recognises_government_domain():
    badge, trust = source_badge("https://www.nasa.gov/europa-clipper")
    assert badge == "Government"
    assert trust == "trust-high"


def test_source_badge_falls_back_to_provider():
    badge, trust = source_badge("", provider="crossref")
    assert badge == "Scientific"
    assert trust == "trust-high"


def test_source_badge_defaults_to_unclassified():
    badge, trust = source_badge("https://example.com/blog", provider="unknown")
    assert badge == "Unclassified"
    assert trust == "trust-low"
