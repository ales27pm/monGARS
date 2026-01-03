import httpx
import pytest

from monGARS.core.iris import Iris

HTML = """
<html><head>
<title>Article</title>
<script type="application/ld+json">
{
 "@context":"https://schema.org",
 "@type":"NewsArticle",
 "headline":"Breaking: Systems upgraded",
 "datePublished":"2025-10-21T10:00:00Z",
 "author":{"@type":"Person","name":"Ada L."},
 "publisher":{"@type":"Organization","name":"Trusted News"}
}
</script>
</head><body><article>All systems upgraded successfully.</article></body></html>
"""


def _transport_ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/ok":
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_iris_fetch_document_enriches_schema():
    transport = httpx.MockTransport(_transport_ok)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        iris = Iris(client_factory=lambda **_: client)
        doc = await iris.fetch_document("https://example.com/ok")
        assert doc and "upgraded" in (doc.text or "").lower()
        assert doc.published_at and doc.publisher == "Trusted News"
        assert "Ada L." in (doc.authors or [])
