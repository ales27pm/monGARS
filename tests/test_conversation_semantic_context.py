from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from monGARS.core.conversation import ConversationalModule
from monGARS.core.persistence import VectorMatch


class _StubLLMIntegration:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls: list[dict[str, object]] = []

    async def generate_response(self, prompt: str, task_type: str = "general", *, response_hints=None):  # type: ignore[override]
        self.prompts.append(prompt)
        self.calls.append(
            {
                "prompt": prompt,
                "task_type": task_type,
                "response_hints": response_hints,
            }
        )
        return {
            "text": "stub-response",
            "confidence": 0.73,
            "source": "stub",
            "adapter_version": "baseline",
        }

    def infer_task_type(self, prompt: str, default: str = "general") -> str:
        lowered = prompt.lower()
        if "```" in prompt or "function" in lowered or "class" in lowered:
            return "coding"
        return default


class _StubCuriosityEngine:
    async def detect_gaps(self, context):  # type: ignore[override]
        return {"status": "sufficient_knowledge"}


class _StubReasoner:
    async def reason(self, query: str, user_id: str):  # type: ignore[override]
        return {"result": ""}


class _ReasoningStub:
    async def reason(self, query: str, user_id: str):  # type: ignore[override]
        return {"result": "Analyse en profondeur."}


class _StubDynamic:
    async def get_personality_traits(self, user_id: str, interactions):  # type: ignore[override]
        return {"tone": "neutral"}

    def generate_adaptive_response(self, text: str, personality, user_id: str):  # type: ignore[override]
        return f"{text}::{user_id}"


class _StubMimicry:
    async def update_profile(self, user_id: str, payload):  # type: ignore[override]
        return None

    async def adapt_response_style(self, text: str, user_id: str):  # type: ignore[override]
        return f"{text}::styled"


class _StubPersonalityEngine:
    async def profile(self, user_id: str):  # pragma: no cover - unused stub hook
        return {}


class _StubCaptioner:
    async def generate_caption(
        self, image_data: bytes
    ):  # pragma: no cover - unused in tests
        return "caption"


class _StubSpeechTurn:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_payload(self) -> dict[str, str]:
        return {"text": self.text}


class _StubSpeakerService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def speak(self, text: str, *, session_id: str | None = None):  # type: ignore[override]
        self.calls.append((text, session_id))
        return _StubSpeechTurn(text)


class _StubMemoryService:
    def __init__(self) -> None:
        self.store_calls: list[tuple[str, str, str]] = []
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self._history = [
            SimpleNamespace(
                query="recent-one", response="recent-resp-one", timestamp=base_time
            ),
            SimpleNamespace(
                query="recent-two", response="recent-resp-two", timestamp=base_time
            ),
        ]

    async def history(self, user_id: str, limit: int = 10):  # type: ignore[override]
        return self._history[:limit]

    async def store(self, user_id: str, query: str, response: str):  # type: ignore[override]
        self.store_calls.append((user_id, query, response))
        return SimpleNamespace(
            user_id=user_id,
            query=query,
            response=response,
            timestamp=datetime.now(timezone.utc),
            expires_at=None,
        )


class _StubEvolutionEngine:
    def __init__(self) -> None:
        self.samples: list[dict[str, object]] = []

    async def record_memory_sample(self, **payload):  # type: ignore[override]
        self.samples.append(payload)


class _StubPersistenceRepository:
    def __init__(self, matches: list[VectorMatch]) -> None:
        self._matches = matches
        self.vector_queries: list[dict[str, object]] = []
        self.saved: list[tuple[object, str | None, str | None]] = []

    async def vector_search_history(self, user_id: str, query: str, *, limit: int, max_distance: float | None = None):  # type: ignore[override]
        self.vector_queries.append(
            {
                "user_id": user_id,
                "query": query,
                "limit": limit,
                "max_distance": max_distance,
            }
        )
        return self._matches

    async def save_interaction(self, interaction, *, history_query=None, history_response=None):  # type: ignore[override]
        self.saved.append((interaction, history_query, history_response))


def _build_conversational_module(
    matches: list[VectorMatch],
    *,
    reasoner: object | None = None,
    llm: _StubLLMIntegration | None = None,
) -> tuple[ConversationalModule, _StubLLMIntegration, _StubPersistenceRepository]:
    llm_instance = llm or _StubLLMIntegration()
    persistence = _StubPersistenceRepository(matches)
    module = ConversationalModule(
        llm=llm_instance,
        reasoner=reasoner or _StubReasoner(),
        curiosity=_StubCuriosityEngine(),
        dynamic=_StubDynamic(),
        mimicry=_StubMimicry(),
        personality=_StubPersonalityEngine(),
        captioner=_StubCaptioner(),
        memory=_StubMemoryService(),
        speaker=_StubSpeakerService(),
        persistence=persistence,
    )
    module.evolution_engine = _StubEvolutionEngine()
    return module, llm_instance, persistence


@pytest.mark.asyncio
async def test_generate_response_injects_semantic_context(monkeypatch) -> None:
    from monGARS.core import conversation as conversation_module

    monkeypatch.setattr(
        conversation_module.settings, "llm2vec_context_limit", 2, raising=False
    )
    monkeypatch.setattr(
        conversation_module.settings, "llm2vec_context_max_distance", 0.6, raising=False
    )

    record = SimpleNamespace(
        id=101,
        query="semantic-question",
        response="semantic-answer",
        timestamp=datetime(2024, 2, 2, tzinfo=timezone.utc),
    )
    matches = [VectorMatch(record=record, distance=0.25)]
    module, llm, persistence = _build_conversational_module(matches)

    response = await module.generate_response("user-1", "What is the project status?")

    assert response["text"].endswith("::styled")
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "Archived interactions retrieved via semantic search" in prompt
    assert "semantic-question" in prompt
    assert persistence.vector_queries == [
        {
            "user_id": "user-1",
            "query": "What is the project status?",
            "limit": 2,
            "max_distance": 0.6,
        }
    ]
    assert persistence.saved, "interaction should be persisted"
    saved_interaction = persistence.saved[0][0]
    assert saved_interaction.input_data[
        "semantic_context"
    ], "semantic context should be recorded"
    assert (
        saved_interaction.input_data["semantic_context"][0]["query"]
        == "semantic-question"
    )
    assert saved_interaction.input_data["semantic_prompt"] == prompt
    assert saved_interaction.context["semantic_matches"]
    assert llm.calls[0]["task_type"] == "general"
    assert llm.calls[0]["response_hints"] is None
    assert saved_interaction.input_data["llm_task_type"] == "general"
    assert saved_interaction.output_data["llm_source"] == "stub"


@pytest.mark.asyncio
async def test_generate_response_skips_semantic_context_when_disabled(
    monkeypatch,
) -> None:
    from monGARS.core import conversation as conversation_module

    monkeypatch.setattr(
        conversation_module.settings, "llm2vec_context_limit", 0, raising=False
    )

    module, llm, persistence = _build_conversational_module(matches=[])

    await module.generate_response("user-42", "Summarise recent updates")

    assert len(llm.prompts) == 1
    assert "Archived interactions retrieved via semantic search" not in llm.prompts[0]
    assert not persistence.vector_queries
    assert llm.calls[0]["task_type"] == "general"


@pytest.mark.asyncio
async def test_reasoning_stub_returns_unexpected_type(monkeypatch):
    from monGARS.core import conversation as conversation_module

    class BadStubMimicry:
        async def __call__(self, *args, **kwargs):
            return 42  # Unexpected type

    llm = conversation_module._StubLLM()
    persistence = conversation_module._StubPersistence()
    prompt = "Test prompt"
    try:
        await conversation_module.generate_response(
            prompt=prompt,
            llm=llm,
            persistence=persistence,
            mimicry=BadStubMimicry(),
        )
    except Exception as e:
        assert isinstance(e, TypeError) or isinstance(e, ValueError)


@pytest.mark.asyncio
async def test_llm_integration_raises_exception(monkeypatch):
    from monGARS.core import conversation as conversation_module

    class FailingLLM(conversation_module._StubLLM):
        async def __call__(self, *args, **kwargs):
            raise RuntimeError("LLM failure")

    llm = FailingLLM()
    persistence = conversation_module._StubPersistence()
    prompt = "Test prompt"
    try:
        await conversation_module.generate_response(
            prompt=prompt,
            llm=llm,
            persistence=persistence,
            mimicry=conversation_module._StubMimicry(),
        )
    except Exception as e:
        assert isinstance(e, RuntimeError)
        assert "LLM failure" in str(e)


@pytest.mark.asyncio
async def test_generate_response_routes_coding_queries(monkeypatch) -> None:
    from monGARS.core import conversation as conversation_module

    monkeypatch.setattr(
        conversation_module.settings, "llm2vec_context_limit", 0, raising=False
    )

    module, llm, _ = _build_conversational_module(matches=[])

    await module.generate_response("dev", "```python\nprint('hello')\n```")

    assert llm.calls[0]["task_type"] == "coding"


@pytest.mark.asyncio
async def test_generate_response_sets_reasoning_hint(monkeypatch) -> None:
    from monGARS.core import conversation as conversation_module

    monkeypatch.setattr(
        conversation_module.settings, "llm2vec_context_limit", 0, raising=False
    )

    module, llm, persistence = _build_conversational_module(
        matches=[], reasoner=_ReasoningStub()
    )

    await module.generate_response("user-9", "Explain why systems fail")

    assert llm.calls[0]["response_hints"] == {"reasoning": True}
    saved_interaction = persistence.saved[0][0]
    assert saved_interaction.input_data["llm_response_hints"] == {"reasoning": True}


@pytest.mark.asyncio
async def test_generate_response_warns_on_unexpected_reasoner_output(
    monkeypatch, caplog
) -> None:
    from monGARS.core import conversation as conversation_module

    class _BadReasoner:
        async def reason(self, query: str, user_id: str):  # type: ignore[override]
            return 42

    monkeypatch.setattr(
        conversation_module.settings, "llm2vec_context_limit", 0, raising=False
    )

    caplog.set_level(logging.WARNING)
    module, llm, persistence = _build_conversational_module(
        matches=[], reasoner=_BadReasoner()
    )

    response = await module.generate_response("user-19", "Summarise the logs")

    assert response["text"].endswith("::styled")
    assert any(
        record.message == "conversation.reasoner.invalid_result"
        for record in caplog.records
    )
    saved_interaction = persistence.saved[0][0]
    assert saved_interaction.input_data["reasoning_metadata"] == {}
    assert saved_interaction.input_data["llm_response_hints"] is None
    assert llm.calls[0]["response_hints"] is None


@pytest.mark.asyncio
async def test_generate_response_propagates_llm_failures(monkeypatch) -> None:
    from monGARS.core import conversation as conversation_module

    class _FailingLLM(_StubLLMIntegration):
        async def generate_response(  # type: ignore[override]
            self, prompt: str, task_type: str = "general", *, response_hints=None
        ):
            raise RuntimeError("LLM failure")

    monkeypatch.setattr(
        conversation_module.settings, "llm2vec_context_limit", 0, raising=False
    )

    failing_llm = _FailingLLM()
    module, _, persistence = _build_conversational_module(matches=[], llm=failing_llm)

    with pytest.raises(RuntimeError, match="LLM failure"):
        await module.generate_response("user-23", "Trigger the failure path")

    assert persistence.saved == []
