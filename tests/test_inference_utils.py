from monGARS.core.inference_utils import (
    CHATML_BEGIN_OF_TEXT,
    CHATML_END_HEADER,
    CHATML_END_OF_TURN,
    CHATML_START_HEADER,
    build_context_prompt,
    build_converged_chat_prompt,
    prepare_tokenizer_inputs,
    render_chat_prompt_from_text,
)


class _TensorStub:
    def __init__(self, name: str) -> None:
        self.name = name
        self.device = "cpu"

    def to(self, device: str | None = None, **_: object) -> "_TensorStub":
        if device is not None:
            self.device = device
        return self


class _MappingStub(dict):
    def to(self, device: str | None = None) -> "_MappingStub":
        moved = _MappingStub({key: value.to(device) for key, value in self.items()})
        return moved


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, payload, **kwargs):  # noqa: ANN001 - signature matches tokenizer
        self.calls.append({"payload": payload, "kwargs": kwargs})
        return _MappingStub(
            {
                "input_ids": _TensorStub("input_ids"),
                "attention_mask": _TensorStub("attention_mask"),
            }
        )


def test_prepare_tokenizer_inputs_batches_sequences_and_moves_to_device() -> None:
    tokenizer = _RecordingTokenizer()
    inputs, batched = prepare_tokenizer_inputs(
        tokenizer,
        ["hello", "world"],
        max_length=32,
        device="cuda:0",
    )

    assert batched is True
    assert tokenizer.calls[0]["payload"] == ["hello", "world"]
    assert tokenizer.calls[0]["kwargs"]["max_length"] == 32
    assert inputs["input_ids"].device == "cuda:0"
    assert inputs["attention_mask"].device == "cuda:0"


def test_prepare_tokenizer_inputs_handles_single_prompt_without_padding() -> None:
    tokenizer = _RecordingTokenizer()
    inputs, batched = prepare_tokenizer_inputs(
        tokenizer,
        "single",
        padding=False,
        truncation=False,
    )

    assert batched is False
    assert tokenizer.calls[0]["payload"] == ["single"]
    assert tokenizer.calls[0]["kwargs"]["padding"] is False
    assert tokenizer.calls[0]["kwargs"]["truncation"] is False
    assert inputs["input_ids"].device == "cpu"


def test_build_context_prompt_includes_sections() -> None:
    prompt = build_context_prompt(
        "Explain the latest update.",
        history_pairs=[("Hi", "Hello there")],
        semantic_context=[
            {"query": "Previous", "response": "Result", "similarity": 0.75}
        ],
    )

    assert "Recent conversation turns" in prompt
    assert "Archived interactions" in prompt
    assert "Current user request" in prompt


def test_render_chat_prompt_from_text_wraps_chatml_tokens() -> None:
    prompt = render_chat_prompt_from_text(
        "Summarise the deployment status.",
        system_prompt="You are Dolphin.",
        include_assistant_stub=False,
    )

    assert "System: You are Dolphin." in prompt.text
    assert "User: Summarise the deployment status." in prompt.text
    assert "Assistant:" not in prompt.text
    assert prompt.chatml.startswith(CHATML_BEGIN_OF_TEXT)
    assert prompt.chatml.endswith(CHATML_END_OF_TURN)
    assert "You are Dolphin." in prompt.chatml


def test_render_chat_prompt_from_text_handles_empty_system_prompt() -> None:
    prompt = render_chat_prompt_from_text(
        "Explain the failure modes.",
        system_prompt="   ",
        include_assistant_stub=False,
    )

    assert prompt.text.startswith("User:")
    assert "Assistant:" not in prompt.text
    assert prompt.chatml.startswith(CHATML_BEGIN_OF_TEXT)
    assert prompt.chatml.endswith(CHATML_END_OF_TURN)
    assert f"{CHATML_START_HEADER}system{CHATML_END_HEADER}" not in prompt.chatml


def test_render_chat_prompt_from_text_handles_empty_user_text() -> None:
    prompt = render_chat_prompt_from_text(
        "",
        system_prompt="You are Dolphin.",
        include_assistant_stub=False,
    )

    assert prompt.text.splitlines()[0] == "System: You are Dolphin."
    assert prompt.text.splitlines()[-1].startswith("User:")
    assert prompt.chatml.startswith(CHATML_BEGIN_OF_TEXT)
    assert prompt.chatml.endswith(CHATML_END_OF_TURN)
    assert f"{CHATML_START_HEADER}user{CHATML_END_HEADER}" in prompt.chatml


def test_render_chat_prompt_from_text_appends_assistant_stub() -> None:
    prompt = render_chat_prompt_from_text(
        "Provide the metrics report.",
        system_prompt="You are Dolphin.",
        include_assistant_stub=True,
    )

    assert prompt.chatml.startswith(CHATML_BEGIN_OF_TEXT)
    assert prompt.chatml.endswith(
        f"{CHATML_START_HEADER}assistant{CHATML_END_HEADER}\n\n"
    )
    assert prompt.chatml.count(CHATML_END_OF_TURN) == 2
    assert prompt.text.endswith("Assistant:")


def test_build_converged_chat_prompt_formats_role_segments() -> None:
    prompt = build_converged_chat_prompt(
        "Summarise the latest release.",
        history_pairs=[("Hello", "Hi there!")],
        semantic_context=[{"query": "Previous", "response": "Answer"}],
        system_prompt="You are Dolphin.",
    )

    lines = prompt.text.splitlines()
    assert lines[0] == "System: You are Dolphin."
    assert any(line.startswith("User:") for line in lines)
    assert prompt.text.strip().endswith("Assistant:")
    assert prompt.chatml.count(CHATML_END_OF_TURN) == 2
