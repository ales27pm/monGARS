from __future__ import annotations

import json
from typing import Any

import pytest

from monGARS.core.aui import LLMActionSuggester


def test_llm_action_suggester_fallback_orders_by_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLM:
        def generate(
            self, prompt: str, max_new_tokens: int, **_: Any
        ) -> str:  # noqa: ARG002
            return "not valid json"

    monkeypatch.setattr("monGARS.core.aui.LLMIntegration.instance", lambda: FakeLLM())

    suggester = LLMActionSuggester()
    prompt = "code code code summarize explain"
    actions = ["code", "summarize", "explain"]
    order = suggester.suggest(prompt, actions, {})
    assert order[0] == "code"

    summary_prompt = "summarize summarize summarize code"
    order2 = suggester.suggest(summary_prompt, actions, {})
    assert order2[0] == "summarize"

    explain_prompt = "explain explain explain summarize"
    order3 = suggester.suggest(explain_prompt, actions, {})
    assert order3[0] == "explain"


def test_llm_action_suggester_deduplicates_ranked_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLM:
        def generate(
            self, prompt: str, max_new_tokens: int, **_: Any
        ) -> str:  # noqa: ARG002
            return '["code", "code", "summarize"]'

    monkeypatch.setattr("monGARS.core.aui.LLMIntegration.instance", lambda: FakeLLM())

    suggester = LLMActionSuggester()
    actions = ["code", "summarize", "explain"]
    order = suggester.suggest("anything", actions, {})

    assert order == ["code", "summarize"]


def test_extract_json_handles_multiple_response_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("monGARS.core.aui.LLMIntegration.instance", lambda: None)
    suggester = LLMActionSuggester()

    responses = [
        '["code", "summarize"]',
        """```json\n[\"summarize\", \"explain\"]\n```""",
        'Here you go: ["explain", "code"]\nThanks!',
        'Result:\n```\n["summarize"]\n``` Extra notes.',
    ]

    for response in responses:
        parsed = json.loads(suggester._extract_json(response))
        assert isinstance(parsed, list)
        assert parsed


def test_extract_json_handles_malformed_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("monGARS.core.aui.LLMIntegration.instance", lambda: None)
    suggester = LLMActionSuggester()

    malformed_responses = [
        "No brackets here",
        "```json\nnot an array\n```",
        "[missing_closer",
    ]

    for response in malformed_responses:
        parsed = json.loads(suggester._extract_json(response))
        assert isinstance(parsed, list)
        assert parsed == []


def test_llm_action_suggester_forwards_context_to_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def encode(
            self, text: str, add_special_tokens: bool = False
        ) -> list[int]:  # noqa: FBT002
            return list(range(len(text)))

    class FakeLLM:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()
            self.received_context: dict | None = None

        def generate(
            self,
            prompt: str,  # noqa: ARG002
            max_new_tokens: int = 100,  # noqa: ARG002
            *,
            context: dict | None = None,
        ) -> str:
            self.received_context = context
            return '["code"]'

    fake_llm = FakeLLM()
    monkeypatch.setattr("monGARS.core.aui.LLMIntegration.instance", lambda: fake_llm)

    suggester = LLMActionSuggester()
    prompt = "code code code"
    actions = ["code", "summarize"]
    provided_context = {"allowed_actions": ["code"], "user_id": "alice"}

    order = suggester.suggest(prompt, actions, provided_context)

    assert order == ["code"]
    assert fake_llm.received_context == provided_context
