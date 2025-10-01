import json
import sys
import types

import httpx
import pytest
import trafilatura


def make_response(
    url: str,
    text: str,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", url)
    response_headers = {"Content-Type": "text/html"}
    if headers:
        response_headers.update(headers)
    return httpx.Response(
        status_code,
        request=request,
        content=text.encode("utf-8"),
        headers=response_headers,
    )


class ClientFactory:
    def __init__(
        self,
        *,
        responses: list[httpx.Response] | None = None,
        error_factory=None,
    ) -> None:
        self._responses = list(responses or [])
        self._error_factory = error_factory

    def __call__(self, *args, **kwargs):  # pragma: no cover - helper behaviour
        factory = self

        class _DummyAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def request(self, method, url, **request_kwargs):
                if factory._error_factory is not None:
                    raise factory._error_factory(method, url)
                if not factory._responses:
                    raise AssertionError("No response queued for request")
                if len(factory._responses) == 1:
                    return factory._responses[0]
                return factory._responses.pop(0)

        return _DummyAsyncClient()


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "spacy",
        types.SimpleNamespace(load=lambda name: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "sqlalchemy",
        types.SimpleNamespace(text=lambda q: q),
    )
    monkeypatch.setitem(
        sys.modules,
        "monGARS.init_db",
        types.SimpleNamespace(async_session_factory=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "monGARS.config",
        types.SimpleNamespace(
            get_settings=lambda: types.SimpleNamespace(
                DOC_RETRIEVAL_URL="",
                curiosity_similarity_threshold=0.5,
                curiosity_minimum_similar_history=0,
                curiosity_graph_gap_cutoff=1,
                curiosity_kg_cache_ttl=300,
                curiosity_kg_cache_max_entries=512,
                curiosity_research_cache_ttl=900,
                curiosity_research_cache_max_entries=256,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "monGARS.core.neurones",
        types.SimpleNamespace(EmbeddingSystem=lambda *a, **k: None),
    )
    yield


@pytest.mark.asyncio
async def test_fetch_text_success(monkeypatch):
    from monGARS.core.iris import Iris

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        ClientFactory(responses=[make_response("http://example.com", "<p>hello</p>")]),
    )
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps({"text": "hello world"}),
    )
    iris = Iris()
    result = await iris.fetch_text("http://example.com")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_fetch_text_http_error(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response("http://bad.com", "", status_code=500)
    monkeypatch.setattr(httpx, "AsyncClient", ClientFactory(responses=[response]))
    iris = Iris(max_retries=0)
    assert await iris.fetch_text("http://bad.com") is None


@pytest.mark.asyncio
async def test_fetch_text_timeout(monkeypatch):
    from monGARS.core.iris import Iris

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        ClientFactory(
            error_factory=lambda method, url: httpx.TimeoutException(
                "slow", request=httpx.Request(method, url)
            )
        ),
    )
    iris = Iris(max_retries=0)
    assert await iris.fetch_text("http://slow.com") is None


@pytest.mark.asyncio
async def test_fetch_text_invalid_url():
    from monGARS.core.iris import Iris

    iris = Iris()
    result = await iris.fetch_text("ftp://example.com")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_text_rejects_binary_payload(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response(
        "http://example.com/image",
        text="binary",
        headers={"Content-Type": "image/png"},
    )
    monkeypatch.setattr(httpx, "AsyncClient", ClientFactory(responses=[response]))
    iris = Iris()
    result = await iris.fetch_text("http://example.com/image")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_text_honours_max_content_length(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response("http://example.com/long", "<p>too long</p>")
    monkeypatch.setattr(httpx, "AsyncClient", ClientFactory(responses=[response]))
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps({"text": "too long"}),
    )
    iris = Iris(max_content_length=5)
    assert await iris.fetch_text("http://example.com/long") is None


@pytest.mark.asyncio
async def test_search_returns_snippet(monkeypatch):
    from monGARS.core.iris import Iris

    html = """
    <html>
      <body>
        <div class="results">
          <div class="result">
            <a class="result__a">Example</a>
            <div class="result__snippet">Example snippet</div>
          </div>
        </div>
      </body>
    </html>
    """
    encoded = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F"
    html = html.replace(
        '<a class="result__a">Example</a>',
        f'<a class="result__a" href="{encoded}">Example</a>',
    )
    search_response = make_response("https://duckduckgo.com/html/?q=test", html)
    document_response = make_response("https://example.com/", "<p>Doc</p>")
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        ClientFactory(responses=[search_response, document_response]),
    )
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps({"summary": "Doc summary", "text": "Doc text"}),
    )
    iris = Iris()
    result = await iris.search("test")
    assert result == "Doc summary"


@pytest.mark.asyncio
async def test_search_falls_back_to_snippet(monkeypatch):
    from monGARS.core.iris import Iris

    html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="https://example.com">Example</a>
          <div class="result__snippet">Example snippet</div>
        </div>
      </body>
    </html>
    """
    response = make_response("https://duckduckgo.com/html/?q=test", html)

    async def fake_fetch_document(url):
        return None

    monkeypatch.setattr(httpx, "AsyncClient", ClientFactory(responses=[response]))
    iris = Iris()
    monkeypatch.setattr(iris, "fetch_document", fake_fetch_document)
    result = await iris.search("test")
    assert result == "Example snippet"


@pytest.mark.asyncio
async def test_fetch_document_returns_structured_payload(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response("http://example.com", "<p>hello</p>")
    monkeypatch.setattr(httpx, "AsyncClient", ClientFactory(responses=[response]))
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps(
            {
                "text": "hello world",
                "summary": "short summary",
                "title": "Hello",
                "language": "en",
            }
        ),
    )
    iris = Iris()
    document = await iris.fetch_document("http://example.com")
    assert document is not None
    assert document.text == "hello world"
    assert document.summary == "short summary"
    assert document.title == "Hello"
    assert document.language == "en"


@pytest.mark.asyncio
async def test_fetch_document_fallbacks_to_html_text(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response(
        "http://example.com", "<p>hello <strong>world</strong></p>"
    )
    monkeypatch.setattr(httpx, "AsyncClient", ClientFactory(responses=[response]))

    def raise_extract(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trafilatura, "extract", raise_extract)
    iris = Iris()
    document = await iris.fetch_document("http://example.com")
    assert document is not None
    assert document.text == "hello world"


@pytest.mark.asyncio
async def test_curiosity_fallback_uses_iris(monkeypatch):
    from monGARS.core.cortex.curiosity_engine import CuriosityEngine
    from monGARS.core.iris import Iris

    async def fake_post(*args, **kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    iris = Iris()

    async def fake_search(query):
        return "web snippet"

    monkeypatch.setattr(iris, "search", fake_search)
    engine = CuriosityEngine(iris=iris)
    result = await engine._perform_research("test query")
    assert "web snippet" in result
