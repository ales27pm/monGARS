"""Shared helpers for text preprocessing across chat and embedding flows."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

CHATML_BEGIN_OF_TEXT = "<|begin_of_text|>"
CHATML_START_HEADER = "<|start_header_id|>"
CHATML_END_HEADER = "<|end_header_id|>"
CHATML_END_OF_TURN = "<|eot_id|>"


@dataclass(slots=True)
class ChatPrompt:
    """Container bundling human readable and ChatML-formatted prompts."""

    text: str
    chatml: str


def _normalise_text(value: str) -> str:
    return value.strip() if value else ""


def _render_chatml_segment(role: str, content: str, *, terminate: bool = True) -> str:
    normalized_role = role.strip().lower() or "user"
    normalized_content = _normalise_text(content)
    segment = (
        f"{CHATML_START_HEADER}{normalized_role}{CHATML_END_HEADER}\n\n"
        f"{normalized_content}"
    )
    if terminate:
        segment += CHATML_END_OF_TURN
    return segment


def render_chat_prompt_from_text(
    user_text: str,
    *,
    system_prompt: str | None = None,
    include_assistant_stub: bool = True,
) -> ChatPrompt:
    """Return a :class:`ChatPrompt` wrapping ``user_text`` in ChatML markers."""

    segments: list[str] = [CHATML_BEGIN_OF_TEXT]
    if system_prompt and system_prompt.strip():
        segments.append(_render_chatml_segment("system", system_prompt))
    segments.append(_render_chatml_segment("user", user_text))
    if include_assistant_stub:
        segments.append(
            _render_chatml_segment("assistant", "", terminate=False)
        )
    chatml = "".join(segments)
    return ChatPrompt(text=user_text, chatml=chatml)



def _move_to_device(value: Any, device: Any | None) -> Any:
    """Move ``value`` to ``device`` when supported."""

    if device is None:
        return value

    to_callable = getattr(value, "to", None)
    if callable(to_callable):
        try:
            return to_callable(device)
        except TypeError:
            # Some tokenizers expose tensors that expect keyword arguments.
            return to_callable(device=device)
    return value


def prepare_tokenizer_inputs(
    tokenizer: Any,
    texts: Sequence[str] | str,
    *,
    max_length: int | None = None,
    device: Any | None = None,
    padding: bool | str = True,
    truncation: bool = True,
    return_tensors: str = "pt",
) -> tuple[dict[str, Any], bool]:
    """Normalise ``texts`` and run the tokenizer with consistent settings."""

    if isinstance(texts, (str, bytes)):
        payload: list[str] = [str(texts)]
        batched = False
    else:
        payload = [str(text) for text in texts]
        batched = True

    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": return_tensors,
        "padding": padding,
        "truncation": truncation,
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length

    tokenized = tokenizer(payload, **tokenizer_kwargs)

    if hasattr(tokenized, "to") and device is not None:
        tokenized = tokenized.to(device)

    if isinstance(tokenized, Mapping):
        iterator = tokenized.items()
    else:
        try:
            iterator = dict(tokenized).items()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise TypeError(
                "tokenizer must return a mapping-compatible object"
            ) from exc

    prepared: dict[str, Any] = {}
    for name, tensor in iterator:
        prepared[name] = _move_to_device(tensor, device)

    return prepared, batched


def build_context_prompt(
    refined_prompt: str,
    *,
    history_pairs: Sequence[tuple[str, str]] | None = None,
    semantic_context: Sequence[Mapping[str, object]] | None = None,
    instruction_template: str | None = None,
) -> str:
    """Render a prompt combining chat history, semantic recall, and instructions."""

    sections: list[str] = []
    pairs = history_pairs or ()
    archive = semantic_context or ()

    if pairs:
        history_lines: list[str] = []
        for idx, (query_text, response_text) in enumerate(pairs, start=1):
            user_line = (query_text or "").strip()
            assistant_line = (response_text or "").strip()
            history_lines.append(
                f"[{idx}] User: {user_line}\n    Assistant: {assistant_line}"
            )
        sections.append(
            "Recent conversation turns (most recent first):\n"
            + "\n".join(history_lines)
        )

    if archive:
        semantic_lines: list[str] = []
        for idx, entry in enumerate(archive, start=1):
            similarity = entry.get("similarity")
            similarity_text = (
                f" (similarity {similarity:.3f})"
                if isinstance(similarity, float)
                else ""
            )
            query_text = (entry.get("query") or "").strip()
            response_text = (entry.get("response") or "").strip()
            semantic_lines.append(
                f"[{idx}]{similarity_text} User: {query_text}\n    Assistant: {response_text}"
            )
        sections.append(
            "Archived interactions retrieved via semantic search:\n"
            + "\n".join(semantic_lines)
        )

    template = instruction_template or (
        "Leverage the provided context to craft an accurate and concise reply. "
        "If the context is unrelated, continue with your best effort response. "
        "Current user request:\n{prompt}"
    )
    sections.append(template.format(prompt=refined_prompt))

    return "\n\n".join(section for section in sections if section.strip())


def build_converged_chat_prompt(
    refined_prompt: str,
    *,
    history_pairs: Sequence[tuple[str, str]] | None = None,
    semantic_context: Sequence[Mapping[str, object]] | None = None,
    instruction_template: str | None = None,
    system_prompt: str | None = None,
    include_assistant_stub: bool = True,
) -> ChatPrompt:
    """Return a :class:`ChatPrompt` that mirrors chat preprocessing for embeddings."""

    context_text = build_context_prompt(
        refined_prompt,
        history_pairs=history_pairs,
        semantic_context=semantic_context,
        instruction_template=instruction_template,
    )
    return render_chat_prompt_from_text(
        context_text,
        system_prompt=system_prompt,
        include_assistant_stub=include_assistant_stub,
    )
