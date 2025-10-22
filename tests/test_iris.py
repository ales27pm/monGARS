import asyncio
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
        self.requests: list[tuple[str, str]] = []

    def __call__(self, *args, **kwargs):  # pragma: no cover - helper behaviour
        factory = self

        class _DummyAsyncClient:
            async def request(self, method, url, **request_kwargs):
                factory.requests.append((method, url))
                if factory._error_factory is not None:
                    raise factory._error_factory(method, url)
                if not factory._responses:
                    raise AssertionError("No response queued for request")
                if len(factory._responses) == 1:
                    return factory._responses[0]
                return factory._responses.pop(0)

            async def aclose(self):
                return None

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
                search_searx_enabled=False,
                search_searx_base_url=None,
                search_searx_api_key=None,
                search_searx_categories=[],
                search_searx_safesearch=None,
                search_searx_default_language="en",
                search_searx_result_cap=10,
                search_searx_timeout_seconds=6.0,
                search_searx_engines=[],
                search_searx_time_range=None,
                search_searx_sitelimit=None,
                search_searx_page_size=None,
                search_searx_max_pages=1,
                search_searx_language_strict=False,
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

    factory = ClientFactory(
        responses=[make_response("http://example.com", "<p>hello</p>")]
    )
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps({"text": "hello world"}),
    )
    iris = Iris(client_factory=factory)
    result = await iris.fetch_text("http://example.com")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_fetch_text_http_error(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response("http://bad.com", "", status_code=500)
    factory = ClientFactory(responses=[response])
    iris = Iris(max_retries=0, client_factory=factory)
    assert await iris.fetch_text("http://bad.com") is None


@pytest.mark.asyncio
async def test_fetch_text_timeout(monkeypatch):
    from monGARS.core.iris import Iris

    factory = ClientFactory(
        error_factory=lambda method, url: httpx.TimeoutException(
            "slow", request=httpx.Request(method, url)
        )
    )
    iris = Iris(max_retries=0, client_factory=factory)
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
    factory = ClientFactory(responses=[response])
    iris = Iris(client_factory=factory)
    result = await iris.fetch_text("http://example.com/image")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_text_honours_max_content_length(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response("http://example.com/long", "<p>too long</p>")
    factory = ClientFactory(responses=[response])
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps({"text": "too long"}),
    )
    iris = Iris(max_content_length=5, client_factory=factory)
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
    factory = ClientFactory(responses=[search_response, document_response])
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps({"summary": "Doc summary", "text": "Doc text"}),
    )
    iris = Iris(client_factory=factory)
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

    factory = ClientFactory(responses=[response])
    iris = Iris(client_factory=factory)
    monkeypatch.setattr(iris, "fetch_document", fake_fetch_document)
    result = await iris.search("test")
    assert result == "Example snippet"


@pytest.mark.asyncio
async def test_search_caches_snippets(monkeypatch):
    from monGARS.core.iris import Iris, IrisDocument

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
    search_response = make_response("https://duckduckgo.com/html/?q=test", html)

    factory = ClientFactory(responses=[search_response])

    iris = Iris(
        search_cache_ttl=60.0,
        search_cache_size=4,
        client_factory=factory,
    )

    fetch_calls = 0

    async def fake_fetch_document(url):
        nonlocal fetch_calls
        fetch_calls += 1
        return IrisDocument(url=url, text="Doc text", summary="Doc summary")

    monkeypatch.setattr(iris, "fetch_document", fake_fetch_document)

    result = await iris.search("Test Query")
    assert result == "Doc summary"
    assert fetch_calls == 1
    assert len(factory.requests) == 1

    result_cached = await iris.search("Test Query")
    assert result_cached == "Doc summary"
    assert fetch_calls == 1
    assert len(factory.requests) == 1


@pytest.mark.asyncio
async def test_search_cache_expires(monkeypatch):
    from monGARS.core.iris import Iris

    html_first = """
    <html>
      <body>
        <div class="result">
          <div class="result__snippet">Snippet 1</div>
        </div>
      </body>
    </html>
    """
    html_second = html_first.replace("Snippet 1", "Snippet 2")
    responses = [
        make_response("https://duckduckgo.com/html/?q=test", html_first),
        make_response("https://duckduckgo.com/html/?q=test", html_second),
    ]

    factory = ClientFactory(responses=responses)

    class FakeMonotonic:
        def __init__(self):
            self.value = 0.0

        def advance(self, amount: float) -> None:
            self.value += amount

        def __call__(self) -> float:
            return self.value

    fake_monotonic = FakeMonotonic()
    monkeypatch.setattr("monGARS.core.iris.monotonic", fake_monotonic)

    iris = Iris(
        search_cache_ttl=1.0,
        search_cache_size=2,
        client_factory=factory,
    )

    first = await iris.search("Cache Example")
    assert first == "Snippet 1"

    fake_monotonic.advance(2.0)

    second = await iris.search("Cache Example")
    assert second == "Snippet 2"
    assert len(factory.requests) == 2


@pytest.mark.asyncio
async def test_search_uses_orchestrator_when_available(monkeypatch):
    from monGARS.core.iris import Iris
    from monGARS.core.search import NormalizedHit

    class DummyOrchestrator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []

        async def search(self, query: str, *, lang: str, max_results: int):
            self.calls.append((query, lang, max_results))
            return [
                NormalizedHit(
                    provider="searxng",
                    title="Example",
                    url="https://example.com/article",
                    snippet="Example snippet from orchestrator.",
                    published_at=None,
                    event_date=None,
                    source_domain="example.com",
                    lang=lang,
                    raw={},
                )
            ]

    orchestrator = DummyOrchestrator()
    iris = Iris(client_factory=ClientFactory(responses=[]))
    iris.attach_search_orchestrator(orchestrator)

    async def fail_fallback(_query: str, _cache_key: str):
        pytest.fail("Fallback should not run when orchestrator returns a snippet")

    monkeypatch.setattr(iris, "_search_with_duckduckgo", fail_fallback)

    result = await iris.search("Example query")
    assert result == "Example snippet from orchestrator."
    assert orchestrator.calls == [("Example query", "en", 5)]

    cached = await iris.search("Example query")
    assert cached == result
    assert orchestrator.calls == [("Example query", "en", 5)]


@pytest.mark.asyncio
async def test_search_orchestrator_falls_back_to_document(monkeypatch):
    from monGARS.core.iris import Iris, IrisDocument
    from monGARS.core.search import NormalizedHit

    class DummyOrchestrator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int]] = []

        async def search(self, query: str, *, lang: str, max_results: int):
            self.calls.append((query, lang, max_results))
            return [
                NormalizedHit(
                    provider="searxng",
                    title="Example",
                    url="https://example.com/full",
                    snippet="",
                    published_at=None,
                    event_date=None,
                    source_domain="example.com",
                    lang=lang,
                    raw={},
                )
            ]

    orchestrator = DummyOrchestrator()
    iris = Iris(client_factory=ClientFactory(responses=[]))
    iris.attach_search_orchestrator(orchestrator)

    fetch_calls = 0

    async def fake_fetch_document(url: str) -> IrisDocument | None:
        nonlocal fetch_calls
        fetch_calls += 1
        return IrisDocument(url=url, summary="Document summary", text="Document text")

    monkeypatch.setattr(iris, "fetch_document", fake_fetch_document)

    async def fail_fallback(_query: str, _cache_key: str):
        pytest.fail("Fallback should not run when orchestrator resolves the snippet")

    monkeypatch.setattr(iris, "_search_with_duckduckgo", fail_fallback)

    result = await iris.search("Need context")
    assert result == "Document summary"
    assert fetch_calls == 1
    assert orchestrator.calls == [("Need context", "en", 5)]


@pytest.mark.asyncio
async def test_fetch_document_returns_structured_payload(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response("http://example.com", "<p>hello</p>")
    factory = ClientFactory(responses=[response])
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
    iris = Iris(client_factory=factory)
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
    factory = ClientFactory(responses=[response])

    def raise_extract(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trafilatura, "extract", raise_extract)
    iris = Iris(client_factory=factory)
    document = await iris.fetch_document("http://example.com")
    assert document is not None
    assert document.text == "hello world"


@pytest.mark.asyncio
async def test_fetch_document_caches_responses(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response("http://example.com", "<p>cached</p>")
    factory = ClientFactory(responses=[response])
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps({"text": "cached body"}),
    )
    iris = Iris(
        document_cache_ttl=60.0,
        document_cache_size=4,
        client_factory=factory,
    )

    first = await iris.fetch_document("http://example.com")
    second = await iris.fetch_document("http://example.com")

    assert first is not None
    assert first is second
    assert len(factory.requests) == 1


@pytest.mark.asyncio
async def test_fetch_document_coalesces_concurrent_requests(monkeypatch):
    from monGARS.core.iris import Iris

    response = make_response("http://example.com", "<p>coalesce</p>")
    release_event = asyncio.Event()
    request_started = asyncio.Event()

    class BlockingClient:
        def __init__(self):
            self.calls = 0

        async def request(self, method, url, **kwargs):
            self.calls += 1
            request_started.set()
            await release_event.wait()
            return response

        async def aclose(self):
            return None

    client = BlockingClient()
    monkeypatch.setattr(
        trafilatura,
        "extract",
        lambda html, **_: json.dumps({"text": "coalesced"}),
    )
    iris = Iris(client_factory=lambda **_: client)

    task_one = asyncio.create_task(iris.fetch_document("http://example.com"))
    task_two = asyncio.create_task(iris.fetch_document("http://example.com"))

    await asyncio.wait_for(request_started.wait(), timeout=1.0)
    assert client.calls == 1

    release_event.set()
    first, second = await asyncio.gather(task_one, task_two)

    assert first is second
    assert client.calls == 1


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
