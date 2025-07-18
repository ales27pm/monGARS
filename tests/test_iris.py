import sys
import types

import httpx
import pytest
import trafilatura

sys.modules.setdefault("spacy", types.ModuleType("spacy"))
sys.modules["spacy"].load = lambda name: object()
sqlalchemy_stub = types.ModuleType("sqlalchemy")
sqlalchemy_stub.text = lambda x: x
sys.modules.setdefault("sqlalchemy", sqlalchemy_stub)

init_db_stub = types.ModuleType("monGARS.init_db")
init_db_stub.async_session_factory = lambda: None
sys.modules.setdefault("monGARS.init_db", init_db_stub)
config_stub = types.ModuleType("monGARS.config")
config_stub.get_settings = lambda: types.SimpleNamespace(DOC_RETRIEVAL_URL="")
sys.modules.setdefault("monGARS.config", config_stub)
neuron_stub = types.ModuleType("monGARS.core.neurones")
neuron_stub.EmbeddingSystem = lambda *a, **k: None
sys.modules.setdefault("monGARS.core.neurones", neuron_stub)

from monGARS.core.cortex.curiosity_engine import CuriosityEngine
from monGARS.core.iris import Iris


@pytest.mark.asyncio
async def test_fetch_text_success(monkeypatch):
    async def fake_get(self, url, timeout=10):
        class Resp:
            text = "<html><body>hello world</body></html>"

            def raise_for_status(self):
                pass

        return Resp()

    def fake_extract(html):
        return "hello world"

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(trafilatura, "extract", fake_extract)
    iris = Iris()
    result = await iris.fetch_text("http://example.com")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_curiosity_fallback_uses_iris(monkeypatch):
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
