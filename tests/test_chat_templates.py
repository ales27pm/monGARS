"""Focused tests for model-native chat-template handling."""

from __future__ import annotations

from monGARS.mlops.chat_templates import (
    DOLPHIN_CHAT_TEMPLATE,
    ensure_dolphin_chat_template,
)


class _Tokenizer:
    def __init__(self, chat_template: str | None) -> None:
        self.chat_template = chat_template


def test_native_chat_template_is_preserved() -> None:
    tokenizer = _Tokenizer("{{ native_model_template }}")

    result = ensure_dolphin_chat_template(tokenizer)

    assert result is tokenizer
    assert tokenizer.chat_template == "{{ native_model_template }}"


def test_missing_chat_template_receives_role_aware_fallback() -> None:
    tokenizer = _Tokenizer(None)

    ensure_dolphin_chat_template(tokenizer)

    assert tokenizer.chat_template == DOLPHIN_CHAT_TEMPLATE
    assert "{% for message in messages %}" in tokenizer.chat_template
    assert "message['role']" in tokenizer.chat_template
    assert "advanced coding agent" not in tokenizer.chat_template.lower()
