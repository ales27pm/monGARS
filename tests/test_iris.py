import sys
import types

import httpx
import pytest
import trafilatura


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
            get_settings=lambda: types.SimpleNamespace(DOC_RETRIEVAL_URL="")
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
    async def fake_get(self, url, timeout=10):
        class Resp:
            text = "<html><body>hello world</body></html>"

            def raise_for_status(self):
                pass

        return Resp()

    from monGARS.core.iris import Iris

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(trafilatura, "extract", lambda html: "hello world")
    iris = Iris()
    result = await iris.fetch_text("http://example.com")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_fetch_text_http_error(monkeypatch):
    async def fake_get(self, url, timeout=10):
        raise httpx.HTTPStatusError("bad", request=None, response=None)

    from monGARS.core.iris import Iris

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    iris = Iris()
    assert await iris.fetch_text("http://bad.com") is None


@pytest.mark.asyncio
async def test_fetch_text_timeout(monkeypatch):
    async def fake_get(self, url, timeout=10):
        raise httpx.TimeoutException("slow")

    from monGARS.core.iris import Iris

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    iris = Iris()
    assert await iris.fetch_text("http://slow.com") is None


@pytest.mark.asyncio
async def test_fetch_text_invalid_url():
    from monGARS.core.iris import Iris

    iris = Iris()
    result = await iris.fetch_text("ftp://example.com")
    assert result is None


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
