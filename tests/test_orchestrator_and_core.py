import asyncio
import json
from types import SimpleNamespace

import pytest

from monGARS.core.orchestrator import ACTION_PREFIX, Orchestrator, ReActOrchestrator, THOUGHT_PREFIX
from monGARS.core.peer import PeerCommunicator
from monGARS.core.ui_events import EventBus, BackendUnavailable, make_event


class FakeRuntime:
    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt: str) -> str:  # noqa: D401
        return f"{prompt} -> {self.text}"


class FakeAsync:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):  # noqa: ANN001
        self.calls.append((args, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.asyncio
async def test_orchestrator_process_query_happy_path(monkeypatch: pytest.MonkeyPatch):
    curiosity = FakeAsync({"status": "insufficient_knowledge", "additional_context": "more"})
    reasoner = FakeAsync({"result": "extra"})
    personality = FakeAsync({"tone": "warm"})
    mimicry_update = FakeAsync(None)
    mimicry_adapt = FakeAsync("adapted")

    class FakeMimicry:
        async def update_profile(self, *a, **k):  # noqa: ANN001
            return await mimicry_update(*a, **k)

        async def adapt_response_style(self, *a, **k):  # noqa: ANN001
            return await mimicry_adapt(*a, **k)

    monkeypatch.setattr(
        "monGARS.core.orchestrator.UnifiedLLMRuntime.instance",
        classmethod(lambda cls: SimpleNamespace(generate=lambda prompt: "llm")),
    )

    orch = Orchestrator(
        llm=SimpleNamespace(),
        reasoner=SimpleNamespace(reason=reasoner),
        personality=SimpleNamespace(),
        dynamic_response=SimpleNamespace(
            get_personality_traits=personality, generate_adaptive_response=lambda text, traits, user_id=None: f"{text}|{traits}"
        ),
        mimicry=FakeMimicry(),
        curiosity=SimpleNamespace(detect_gaps=curiosity),
        captioner=SimpleNamespace(generate_caption=FakeAsync("pic")),
    )

    result = await orch.process_query("user", "hello", image_data=b"img")
    assert "text" in result and result["confidence"] == 0.9
    assert result["text"]
    assert curiosity.calls and reasoner.calls
    assert personality.calls and mimicry_update.calls


def test_react_orchestrator_validates_and_executes(monkeypatch: pytest.MonkeyPatch):
    reactor = ReActOrchestrator()

    fake_response = f"{THOUGHT_PREFIX} Plan\n{ACTION_PREFIX} {{}}"
    monkeypatch.setattr(
        "monGARS.core.orchestrator.LLMIntegration.instance", classmethod(lambda cls: SimpleNamespace(generate=lambda prompt: fake_response))
    )

    async def fake_enrich(query, repositories=None, max_results=None):  # noqa: ANN001
        return SimpleNamespace(focus_areas=["x"], references=[])

    reactor._rag_enricher.enrich = fake_enrich  # type: ignore[attr-defined]
    result = reactor.execute_tool("rag_search", {"query": "hey"}, {"user_id": "u"})
    assert result.get("error") is None
    assert result.get("query") == "hey"


@pytest.mark.asyncio
async def test_peer_communicator_broadcasts_and_caches(monkeypatch: pytest.MonkeyPatch):
    posts = []

    class FakeResponse:
        def __init__(self, status_code=202, json_data=None):
            self.status_code = status_code
            self._json = json_data or {}

        async def aclose(self):
            return None

        def json(self):  # noqa: D401
            return self._json

    class FakeClient:
        def __init__(self):
            self.closed = False

        async def post(self, url, json=None, headers=None):  # noqa: ANN001
            posts.append((url, json, headers))
            return FakeResponse()

        async def get(self, url, headers=None):  # noqa: ANN001
            return FakeResponse(status_code=200, json_data={"load_factor": 0.5, "observed_at": "2024-01-01T00:00:00Z"})

    communicator = PeerCommunicator(peers=["https://peer.example"], client=FakeClient(), identity="node-1", bearer_token="tok")
    payload = {"load_factor": 0.1, "observed_at": "2024-01-01T00:00:00Z"}
    ok = await communicator.broadcast_telemetry(payload)
    assert ok is True
    assert communicator.get_peer_telemetry(include_self=True)

    loads = await communicator.fetch_peer_loads()
    assert loads
    assert posts  # broadcast attempted


@pytest.mark.asyncio
async def test_event_bus_falls_back_to_memory(monkeypatch: pytest.MonkeyPatch):
    calls = {"publish": 0}

    class FailingBackend:
        async def publish(self, ev):  # noqa: ANN001
            calls["publish"] += 1
            raise BackendUnavailable("fail")

        def subscribe(self):  # noqa: ANN001
            raise BackendUnavailable("fail")

    bus = EventBus()
    bus._backend = FailingBackend()  # type: ignore[assignment]

    ev = make_event("chat.message", "u1", {"text": "hi"})
    await bus.publish(ev)
    iterator = bus.subscribe()
    # ensure iterator is memory backed after fallback
    assert hasattr(iterator, "__anext__")
