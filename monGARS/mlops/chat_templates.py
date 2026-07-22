"""Shared chat-template helpers for Dolphin-aligned tokenizers."""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Compatibility fallback for exported tokenizers that genuinely do not ship a
# chat template.  Runtime tokenizers normally provide their own model-specific
# Jinja template; ``ensure_dolphin_chat_template`` deliberately preserves it.
DOLPHIN_CHAT_TEMPLATE = """{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""


def ensure_dolphin_chat_template(
    tokenizer: PreTrainedTokenizerBase | None,
    template: str = DOLPHIN_CHAT_TEMPLATE,
) -> PreTrainedTokenizerBase | None:
    """Keep a tokenizer's native template, filling only a missing template.

    Older monGARS builds replaced every tokenizer template with a static system
    prompt.  That bypassed the model's role and special-token contract.  The
    fallback here is used only for legacy/exported tokenizers whose
    ``chat_template`` is absent or blank.
    """

    if tokenizer is None or not hasattr(tokenizer, "chat_template"):
        return tokenizer

    current_template = getattr(tokenizer, "chat_template", None)
    if isinstance(current_template, str) and current_template.strip():
        return tokenizer

    try:
        tokenizer.chat_template = template
        logger.debug(
            "Applied fallback Dolphin chat template to tokenizer",
            extra={"tokenizer": type(tokenizer).__name__},
        )
    except Exception:  # pragma: no cover - some tokenizers expose read-only state
        logger.debug(
            "Tokenizer does not allow assigning a fallback chat template",
            exc_info=True,
        )
    return tokenizer


def load_tokenizer_with_dolphin_chat_template(
    model_id: str,
    /,
    *,
    use_fast: bool = True,
    ensure_padding: bool = True,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer while preserving its model-native chat template."""

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast, **kwargs)
    if (
        ensure_padding
        and getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    ensure_dolphin_chat_template(tokenizer)
    return tokenizer
