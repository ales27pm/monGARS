"""Tests for the LLM2Vec embedder utilities."""

import math

import pytest

from monGARS.config import Settings
from monGARS.core.embeddings import (
    Dolphin3Embedder,
    EmbeddingBackendError,
    LLM2VecEmbedder,
)
from monGARS.core.inference_utils import (
    CHATML_BEGIN_OF_TEXT,
    CHATML_END_HEADER,
    CHATML_END_OF_TURN,
    CHATML_START_HEADER,
    render_chat_prompt_from_text,
)


class _RecordingManager:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        self.calls.append(list(texts))
        base_vector = [float(len(self.calls)), 42.0, 84.0, 168.0]
        return [base_vector for _ in texts]


class _FailingManager:
    def __init__(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        raise RuntimeError("embedding backend unavailable")


class _PartialManager:
    def __init__(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float] | None]:
        return [[], None]


class _NotReadyManager:
    def __init__(self) -> None:
        self._ready = False
        self.calls: int = 0

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 2.0, 3.0] for _ in texts]


class _NonFiniteManager:
    def __init__(self) -> None:
        self._ready = True
        self.return_value = float("nan")

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        return [[self.return_value, 1.0, 2.0] for _ in texts]


@pytest.mark.asyncio
async def test_encode_batch_chunks_requests_and_normalises_dimensions() -> None:
    settings = Settings(
        llm2vec_max_batch_size=2,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _RecordingManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payloads = [f"text-{idx}" for idx in range(5)]
    result = await embedder.encode_batch(payloads)

    assert len(result.vectors) == len(payloads)
    assert all(len(vector) == 3 for vector in result.vectors)
    expected_batches = [payloads[:2], payloads[2:4], payloads[4:]]
    assert len(manager.calls) == len(expected_batches)
    for recorded_batch, expected_texts in zip(
        manager.calls, expected_batches, strict=True
    ):
        assert len(recorded_batch) == len(expected_texts)
        for rendered, original in zip(recorded_batch, expected_texts, strict=True):
            assert rendered.startswith(CHATML_BEGIN_OF_TEXT)
            assert rendered.endswith(CHATML_END_OF_TURN)
            assert f"{CHATML_START_HEADER}user{CHATML_END_HEADER}" in rendered
            assert settings.llm2vec_instruction in rendered
            assert original in rendered
    assert result.used_fallback is False


@pytest.mark.asyncio
async def test_embed_text_raises_on_backend_failure() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=_FailingManager
    )

    with pytest.raises(EmbeddingBackendError):
        await embedder.embed_text("hello world")


@pytest.mark.asyncio
async def test_encode_batch_generates_deterministic_fallback_vectors() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=5,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _PartialManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payloads = ["alpha", "beta"]
    first = await embedder.encode_batch(payloads)
    second = await embedder.encode_batch(payloads)

    assert first.used_fallback is True
    assert len(first.vectors) == len(payloads)
    assert first.vectors == second.vectors
    assert {len(vector) for vector in first.vectors} == {5}
    for vector in first.vectors:
        magnitude = math.sqrt(sum(component * component for component in vector))
        assert magnitude == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_encode_batch_skips_backend_for_blank_inputs() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=4,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _RecordingManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payloads = ["", "   "]
    batch = await embedder.encode_batch(payloads)

    assert batch.used_fallback is True
    assert manager.calls == []
    assert all(len(vector) == 4 for vector in batch.vectors)


@pytest.mark.asyncio
async def test_encode_batch_uses_fallback_when_manager_not_ready() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _NotReadyManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    batch = await embedder.encode_batch(["alpha", "beta"])

    assert batch.used_fallback is True
    assert manager.calls == 0
    assert all(len(vector) == 3 for vector in batch.vectors)


@pytest.mark.asyncio
async def test_encode_batch_records_chatml_for_fallback_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        llm2vec_max_batch_size=2,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        llm2vec_instruction="Embed with care.",
        debug=True,
    )
    manager = _NotReadyManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    captured: list[dict[str, object]] = []
    original = render_chat_prompt_from_text

    def _recording_render_chat_prompt_from_text(
        user_text: str,
        *,
        system_prompt: str | None = None,
        include_assistant_stub: bool = True,
    ):
        prompt = original(
            user_text,
            system_prompt=system_prompt,
            include_assistant_stub=include_assistant_stub,
        )
        captured.append(
            {
                "text": prompt.text,
                "chatml": prompt.chatml,
                "system_prompt": system_prompt,
                "include_assistant_stub": include_assistant_stub,
            }
        )
        return prompt

    monkeypatch.setattr(
        "monGARS.core.embeddings.render_chat_prompt_from_text",
        _recording_render_chat_prompt_from_text,
    )

    payloads = ["first payload", "second payload"]
    batch = await embedder.encode_batch(payloads)

    assert batch.used_fallback is True
    assert manager.calls == 0
    assert len(captured) % len(payloads) == 0
    block_count = len(captured) // len(payloads)
    blocks = [
        captured[index * len(payloads) : (index + 1) * len(payloads)]
        for index in range(block_count)
    ]

    for idx, original_text in enumerate(payloads):
        previous_chatml: str | None = None
        for block in blocks:
            entry = block[idx]
            assert entry["text"] == original_text
            assert entry["system_prompt"] == settings.llm2vec_instruction
            assert entry["include_assistant_stub"] is False
            chatml = entry["chatml"]
            if previous_chatml is None:
                previous_chatml = chatml
            else:
                assert chatml == previous_chatml
            assert chatml.startswith(CHATML_BEGIN_OF_TEXT)
            assert chatml.endswith(CHATML_END_OF_TURN)
            assert f"{CHATML_START_HEADER}user{CHATML_END_HEADER}" in chatml
            assert f"{CHATML_START_HEADER}system{CHATML_END_HEADER}" in chatml
            assert settings.llm2vec_instruction in chatml
            assert original_text in chatml


@pytest.mark.asyncio
async def test_encode_batch_fallback_depends_on_instruction_context() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=5,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _PartialManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payload = ["shared-text"]
    without_instruction = await embedder.encode_batch(payload)
    with_instruction = await embedder.encode_batch(
        payload, instruction="represent for troubleshooting"
    )

    assert without_instruction.used_fallback is True
    assert with_instruction.used_fallback is True
    assert without_instruction.vectors[0] != with_instruction.vectors[0]


@pytest.mark.asyncio
async def test_encode_batch_fallback_triggers_on_non_finite_values() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _NonFiniteManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    # Test fallback for NaN values
    batch = await embedder.encode_batch(["alpha"])

    assert batch.used_fallback is True
    assert len(batch.vectors) == 1
    assert len(batch.vectors[0]) == 3
    magnitude = math.sqrt(sum(component * component for component in batch.vectors[0]))
    assert magnitude == pytest.approx(1.0)

    # Test fallback for positive infinity values
    manager.return_value = float("inf")
    batch_inf = await embedder.encode_batch(["alpha"])

    assert batch_inf.used_fallback is True
    assert len(batch_inf.vectors) == 1
    assert len(batch_inf.vectors[0]) == 3
    magnitude_inf = math.sqrt(
        sum(component * component for component in batch_inf.vectors[0])
    )
    assert magnitude_inf == pytest.approx(1.0)

    # Test fallback for negative infinity values
    manager.return_value = float("-inf")
    batch_ninf = await embedder.encode_batch(["alpha"])

    assert batch_ninf.used_fallback is True
    assert len(batch_ninf.vectors) == 1
    assert len(batch_ninf.vectors[0]) == 3
    magnitude_ninf = math.sqrt(
        sum(component * component for component in batch_ninf.vectors[0])
    )
    assert magnitude_ninf == pytest.approx(1.0)


@pytest.fixture(scope="session")
def dolphin3_tiny_embedder() -> Dolphin3Embedder:
    """Return a Dolphin 3 embedder backed by a tiny reference checkpoint."""

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    embedder = Dolphin3Embedder(
        settings=Settings(),
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        device="cpu",
        batch_size=2,
        max_length=64,
        target_dimension=3072,
        torch_dtype="float32",
    )

    try:
        embedder.encode(["warmup sentence for dolphin 3 embeddings"])
    except EmbeddingBackendError as exc:  # pragma: no cover - dependency missing
        pytest.skip(f"Unable to load Dolphin 3 embedding model: {exc}")
    except OSError as exc:  # pragma: no cover - HF download/IO failure
        pytest.skip(f"Dolphin 3 embedding model unavailable: {exc}")

    return embedder


def test_dolphin3_embedder_respects_configured_dimension(
    dolphin3_tiny_embedder: Dolphin3Embedder,
) -> None:
    vectors = dolphin3_tiny_embedder.encode(["alpha", "beta"])

    assert len(vectors) == 2
    assert {len(vector) for vector in vectors} == {
        dolphin3_tiny_embedder.vector_dimension
    }

    torch_module, model, _ = dolphin3_tiny_embedder._ensure_model_components()
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if (
        isinstance(hidden_size, int)
        and hidden_size < dolphin3_tiny_embedder.vector_dimension
    ):
        tail_index = hidden_size
        for vector in vectors:
            assert all(abs(component) < 1e-6 for component in vector[tail_index:])


def test_dolphin3_embedder_matches_manual_mean_pool(
    dolphin3_tiny_embedder: Dolphin3Embedder,
) -> None:
    torch_module, model, tokenizer = dolphin3_tiny_embedder._ensure_model_components()
    text = "verifying dolphin 3 pooling"

    reference_vectors = dolphin3_tiny_embedder.encode([text])
    assert len(reference_vectors) == 1
    reference_vector = reference_vectors[0]

    system_prompt = getattr(
        dolphin3_tiny_embedder._settings, "llm2vec_instruction", None
    )
    formatted = render_chat_prompt_from_text(
        text,
        system_prompt=system_prompt,
        include_assistant_stub=False,
    ).chatml

    tokenized = tokenizer(
        [formatted],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=dolphin3_tiny_embedder.max_length,
    )
    prepared = {
        name: (
            tensor.to(dolphin3_tiny_embedder.device)
            if hasattr(tensor, "to")
            else tensor
        )
        for name, tensor in tokenized.items()
    }
    with torch_module.inference_mode():
        outputs = model(**prepared, output_hidden_states=True)

    final_hidden = outputs.hidden_states[-1]
    mask = prepared.get("attention_mask")
    if mask is None:
        mask_tensor = torch_module.ones(
            final_hidden.shape[:2],
            dtype=final_hidden.dtype,
            device=final_hidden.device,
        )
    else:
        mask_tensor = mask.to(final_hidden.dtype)
    mask_tensor = mask_tensor.unsqueeze(-1)

    pooled = torch_module.nan_to_num(
        (final_hidden * mask_tensor).sum(dim=1) / mask_tensor.sum(dim=1).clamp_min(1.0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )[0]
    manual_vector = pooled.detach().to(torch_module.float32)
    if manual_vector.device.type != "cpu":
        manual_vector = manual_vector.cpu()
    if manual_vector.shape[-1] > dolphin3_tiny_embedder.vector_dimension:
        manual_vector = manual_vector[: dolphin3_tiny_embedder.vector_dimension]
    elif manual_vector.shape[-1] < dolphin3_tiny_embedder.vector_dimension:
        pad = torch_module.zeros(
            dolphin3_tiny_embedder.vector_dimension - manual_vector.shape[-1],
            dtype=manual_vector.dtype,
            device=manual_vector.device,
        )
        manual_vector = torch_module.cat((manual_vector, pad), dim=0)

    assert reference_vector == pytest.approx(manual_vector.tolist(), abs=1e-5)
