from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from monGARS.core import orchestrator as orchestrator_module
from monGARS.core.orchestrator import (
    InvalidReasoningFormatError,
    ReActOrchestrator,
)


class DummyLLM:
    def __init__(self) -> None:
        self.response = ""

    def generate(self, prompt: str) -> str:
        return self.response


class DummyRagEnricher:
    async def enrich(self, query: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(focus_areas=["context"], references=[])


class MonitorStub:
    def __init__(self) -> None:
        self.traces: list[tuple[str, dict]] = []

    def start_trace(
        self, trace_id: str, metadata: dict
    ) -> None:  # pragma: no cover - noop
        _ = (trace_id, metadata)

    def log_trace(self, trace_id: str, payload: dict) -> None:
        self.traces.append((trace_id, payload))

    def log_error(
        self, trace_id: str, error_code: str, payload: dict
    ) -> None:  # pragma: no cover - noop
        _ = (trace_id, error_code, payload)

    def complete_trace(
        self, trace_id: str, result: dict
    ) -> None:  # pragma: no cover - noop
        _ = (trace_id, result)


@pytest.fixture()
def dummy_llm(monkeypatch: pytest.MonkeyPatch) -> DummyLLM:
    client = DummyLLM()
    monkeypatch.setattr(
        orchestrator_module.LLMIntegration,
        "instance",
        classmethod(lambda cls: client),
    )
    return client


@pytest.fixture()
def monitor_stub(monkeypatch: pytest.MonkeyPatch) -> MonitorStub:
    stub = MonitorStub()
    monkeypatch.setattr(orchestrator_module, "monitor", stub)
    return stub


def test_react_pattern_enforcement(
    dummy_llm: DummyLLM, monitor_stub: MonitorStub
) -> None:
    orchestrator = ReActOrchestrator(rag_enricher=DummyRagEnricher())
    dummy_llm.response = (
        '[THOUGHT] Evaluate rag search\n[ACTION] {"tool": "rag_search"}'
    )

    result = orchestrator.execute_tool(
        "rag_search", {"query": "latest"}, {"user_id": "alice"}
    )

    assert result["tool"] == "rag_search"
    assert monitor_stub.traces, "trace payload must be logged"
    trace_id, payload = monitor_stub.traces[-1]
    assert payload["event"] == "tool_execution"
    assert payload["tool"] == "rag_search"
    assert payload["arguments"] == {"query": "latest"}
    assert payload["reasoning"] == "Evaluate rag search"
    datetime.fromisoformat(
        payload["timestamp"]
    )  # raises ValueError on malformed timestamp
    assert trace_id

    dummy_llm.response = "No react prefix"
    with pytest.raises(InvalidReasoningFormatError) as exc:
        orchestrator.execute_tool(
            "rag_search", {"query": "latest"}, {"user_id": "alice"}
        )
    assert str(exc.value) == "LLM response missing [THOUGHT] reasoning prefix"
