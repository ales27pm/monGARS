from monGARS.core.inference_utils import (
    CHATML_BEGIN_OF_TEXT,
    CHATML_END_OF_TURN,
    build_context_prompt,
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

    assert prompt.text == "Summarise the deployment status."
    assert prompt.chatml.startswith(CHATML_BEGIN_OF_TEXT)
    assert prompt.chatml.endswith(CHATML_END_OF_TURN)
    assert "You are Dolphin." in prompt.chatml
