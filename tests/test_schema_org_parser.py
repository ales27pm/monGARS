from monGARS.core.search.schema_org import parse_schema_org

HTML_ARTICLE = """
<html><head>
<title>Sample News</title>
<meta property="article:published_time" content="2025-10-19T14:05:00Z"/>
<script type="application/ld+json">
{
  "@context":"https://schema.org",
  "@type":"NewsArticle",
  "headline":"Rocket lands successfully",
  "datePublished":"2025-10-19T14:05:00Z",
  "dateModified":"2025-10-19T15:10:00Z",
  "publisher":{"@type":"Organization","name":"Example Times"},
  "author":[{"@type":"Person","name":"Alex M."}],
  "mainEntityOfPage":"https://example.com/news/rocket-lands"
}
</script>
</head><body></body></html>
"""

HTML_EVENT = """
<html><head>
<script type="application/ld+json">
{
 "@context":"https://schema.org",
 "@type":"Event",
 "name":"OpenAI DevDay",
 "startDate":"2025-11-12T09:00:00Z",
 "endDate":"2025-11-12T17:00:00Z",
 "location":{"@type":"Place","name":"Moscone Center"}
}
</script>
</head><body></body></html>
"""


def test_article_parse():
    s = parse_schema_org(HTML_ARTICLE)
    assert s and s.type == "NewsArticle"
    assert s.headline == "Rocket lands successfully"
    assert s.publisher == "Example Times"
    assert s.date_published and s.date_modified
    assert s.model_dump()["date_published"].startswith("2025-10-19")


def test_event_parse():
    s = parse_schema_org(HTML_EVENT)
    assert s and s.type == "Event"
    assert s.location_name == "Moscone Center"
    assert s.event_start and s.event_end
