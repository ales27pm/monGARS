from __future__ import annotations

import pytest

from monGARS.core.aui import LLMActionSuggester


def test_llm_action_suggester_fallback_orders_by_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLM:
        def generate(self, prompt: str, max_new_tokens: int) -> str:  # noqa: ARG002
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
        def generate(self, prompt: str, max_new_tokens: int) -> str:  # noqa: ARG002
            return '["code", "code", "summarize"]'

    monkeypatch.setattr("monGARS.core.aui.LLMIntegration.instance", lambda: FakeLLM())

    suggester = LLMActionSuggester()
    actions = ["code", "summarize", "explain"]
    order = suggester.suggest("anything", actions, {})

    assert order == ["code", "summarize"]
