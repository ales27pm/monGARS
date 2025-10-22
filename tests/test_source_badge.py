from monGARS.core.search.source_badge import source_badge


def test_gov_badge():
    assert source_badge("https://www.cdc.gov/page") == ("Government", "trust-high")


def test_science_badge_by_domain():
    assert source_badge("https://www.nature.com/articles/xyz") == (
        "Scientific",
        "trust-high",
    )


def test_science_badge_by_provider():
    assert source_badge("https://foo.example/item", provider="arxiv") == (
        "Scientific",
        "trust-high",
    )


def test_blog_badge():
    assert source_badge("https://myproject.medium.com/post") == ("Blog", "trust-low")


def test_reference_badge():
    assert source_badge("https://en.wikipedia.org/wiki/Test") == (
        "Reference",
        "trust-medium",
    )


def test_unclassified_fallback():
    assert source_badge("https://unknownsite.example/post") == (
        "Unclassified",
        "trust-low",
    )
