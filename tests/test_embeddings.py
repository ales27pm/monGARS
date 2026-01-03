"""Tests for the LLM2Vec embedder utilities."""

import math
import sys

import pytest

from monGARS.config import Settings
from monGARS.core.embeddings import (
    DolphinX1Embedder,
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


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(component * component for component in values))


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


class _DeterministicManager:
    """Manager double that returns a preset set of vectors for assertions."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._ready = True
        self._vectors = vectors
        self.calls: list[tuple[list[str], str]] = []

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        self.calls.append((list(texts), prompt))
        return self._vectors


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


class _FakeHTTPError(Exception):
    """Substitute for httpx.HTTPError in tests."""


class _FakeTimeout:
    """Lightweight timeout stub mirroring httpx.Timeout initialisation."""

    def __init__(self, total: float, *, connect: float) -> None:
        self.total = total
        self.connect = connect


class _FakeResponse:
    def __init__(
        self,
        data: dict[str, object] | list[object],
        *,
        status_code: int = 200,
        error_cls: type[Exception] = _FakeHTTPError,
    ) -> None:
        self._data = data
        self.status_code = status_code
        self._error_cls = error_cls

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise self._error_cls(f"HTTP {self.status_code}")

    def json(self) -> dict[str, object] | list[object]:
        return self._data


class _FakeAsyncClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: _FakeTimeout,
        headers: dict[str, str] | None,
        module: "_FakeHTTPXModule",
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers
        self._module = module
        self.post_calls: list[tuple[str, dict[str, object]]] = []
        self.get_calls: list[str] = []

    async def get(self, path: str) -> _FakeResponse:
        self.get_calls.append(path)
        if self._module.health_queue:
            payload, status = self._module.health_queue.pop(0)
        else:
            payload, status = ({"status": "ok"}, 200)
        return _FakeResponse(
            payload, status_code=status, error_cls=self._module.HTTPError
        )

    async def post(self, path: str, json: dict[str, object]) -> _FakeResponse:
        self.post_calls.append((path, json))
        if self._module.post_queue:
            payload, status = self._module.post_queue.pop(0)
        else:
            payload, status = ({"embeddings": [], "dimension": 0}, 200)
        return _FakeResponse(
            payload, status_code=status, error_cls=self._module.HTTPError
        )

    async def aclose(self) -> None:  # pragma: no cover - compatibility hook
        self._module.closed_clients.append(self)


class _FakeHTTPXModule:
    """Minimal shim emulating the subset of httpx used by the embedder."""

    def __init__(self) -> None:
        self.HTTPError = _FakeHTTPError
        self.Timeout = _FakeTimeout
        self.post_queue: list[tuple[dict[str, object], int]] = []
        self.health_queue: list[tuple[dict[str, object], int]] = []
        self.created_clients: list[_FakeAsyncClient] = []
        self.closed_clients: list[_FakeAsyncClient] = []

    def AsyncClient(
        self,
        *,
        base_url: str,
        timeout: _FakeTimeout,
        headers: dict[str, str] | None = None,
    ) -> _FakeAsyncClient:
        client = _FakeAsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            module=self,
        )
        self.created_clients.append(client)
        return client


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
async def test_encode_batch_returns_cached_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _RecordingManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payloads = ["cached", "batch"]
    first = await embedder.encode_batch(payloads)

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("encode should not be invoked on a cache hit")

    monkeypatch.setattr(manager, "encode", _should_not_run)

    second = await embedder.encode_batch(payloads)

    assert second.vectors == first.vectors
    assert second.used_fallback is first.used_fallback


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
        magnitude = _vector_norm(vector)
        assert magnitude == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_dolphin_service_backend_requests_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeHTTPXModule()
    module.health_queue.append(({"status": "ok", "model": "stub", "dimension": 3}, 200))
    module.post_queue.append(
        (
            {
                "embeddings": [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                "dimension": 3,
            },
            200,
        )
    )
    monkeypatch.setitem(sys.modules, "httpx", module)

    settings = Settings(
        embedding_backend="dolphin-x1-llm2vec",
        llm2vec_max_batch_size=8,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
        dolphin_x1_llm2vec_service_url="http://localhost:9090",
    )
    embedder = LLM2VecEmbedder(settings=settings)

    payloads = ["alpha", "beta"]
    batch = await embedder.encode_batch(payloads)

    assert batch.used_fallback is False
    assert batch.vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    assert module.created_clients  # client instantiated lazily
    client = module.created_clients[0]
    assert client.get_calls == ["/health"]
    assert client.post_calls == [("/embed", {"texts": payloads})]


@pytest.mark.asyncio
async def test_dolphin_service_backend_falls_back_on_invalid_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeHTTPXModule()
    module.health_queue.append(({"status": "ok", "model": "stub", "dimension": 3}, 200))
    module.post_queue.append(
        ({"embeddings": [["not", "numbers"]], "dimension": 3}, 200)
    )
    monkeypatch.setitem(sys.modules, "httpx", module)

    settings = Settings(
        embedding_backend="dolphin-x1-llm2vec",
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    embedder = LLM2VecEmbedder(settings=settings)

    batch = await embedder.encode_batch(["needs fallback"])
    assert batch.used_fallback is True
    assert len(batch.vectors) == 1
    assert len(batch.vectors[0]) == 3

    # Subsequent calls return the cached fallback without additional HTTP calls.
    cached = await embedder.encode_batch(["needs fallback"])
    assert cached.vectors == batch.vectors
    assert module.created_clients  # ensure the same client is reused
    client = module.created_clients[0]
    assert client.post_calls == [("/embed", {"texts": ["needs fallback"]})]


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
async def test_encode_batch_returns_vectors_from_ready_manager() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=4,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    expected_vectors = [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]]
    manager = _DeterministicManager(expected_vectors)
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    batch = await embedder.encode_batch(["first", "second"], instruction="Prompt")

    assert batch.used_fallback is False
    assert batch.vectors == expected_vectors
    assert len(manager.calls) == 1
    payloads, recorded_prompt = manager.calls[0]
    assert recorded_prompt == "Prompt"
    assert len(payloads) == 2
    for payload, original_text in zip(payloads, ["first", "second"], strict=True):
        assert payload.startswith(CHATML_BEGIN_OF_TEXT)
        assert f"{CHATML_START_HEADER}system{CHATML_END_HEADER}" in payload
        assert f"{CHATML_START_HEADER}user{CHATML_END_HEADER}" in payload
        assert original_text in payload


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
        captured[slice(index * len(payloads), (index + 1) * len(payloads))]
        for index in range(block_count)
    ]

    for idx, original_text in enumerate(payloads):
        previous_chatml: str | None = None
        for block in blocks:
            entry = block[idx]
            text_lines = [line for line in entry["text"].splitlines() if line]
            assert text_lines[0].startswith("System:")
            assert text_lines[-1].startswith("User:")
            assert text_lines[-1].endswith(original_text)
            assert "Assistant:" not in entry["text"]
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
    magnitude = _vector_norm(batch.vectors[0])
    assert magnitude == pytest.approx(1.0)

    # Test fallback for positive infinity values
    manager.return_value = float("inf")
    batch_inf = await embedder.encode_batch(["alpha"])

    assert batch_inf.used_fallback is True
    assert len(batch_inf.vectors) == 1
    assert len(batch_inf.vectors[0]) == 3
    magnitude_inf = _vector_norm(batch_inf.vectors[0])
    assert magnitude_inf == pytest.approx(1.0)

    # Test fallback for negative infinity values
    manager.return_value = float("-inf")
    batch_ninf = await embedder.encode_batch(["alpha"])

    assert batch_ninf.used_fallback is True
    assert len(batch_ninf.vectors) == 1
    assert len(batch_ninf.vectors[0]) == 3
    magnitude_ninf = _vector_norm(batch_ninf.vectors[0])
    assert magnitude_ninf == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_encode_batch_fallback_matches_internal_generator() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=5,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=_NotReadyManager
    )
    prompt = "Fallback instruction"
    payload = "payload"

    batch = await embedder.encode_batch([payload], instruction=prompt)

    assert batch.used_fallback is True
    assert len(batch.vectors) == 1
    expected_vector = embedder._fallback_vector(prompt, payload)
    assert batch.vectors[0] == expected_vector
    assert len(expected_vector) == settings.llm2vec_vector_dimensions


@pytest.mark.asyncio
async def test_transformers_backend_matches_reference_model() -> None:
    sentence_transformers = pytest.importorskip("sentence_transformers")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    reference_model = sentence_transformers.SentenceTransformer(model_name)

    dimension = int(reference_model.get_sentence_embedding_dimension())
    text = "test sentence"

    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=dimension,
        transformers_embedding_model=model_name,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )

    embedder = LLM2VecEmbedder(settings=settings, backend="transformers")

    batch = await embedder.encode_batch([text])

    assert embedder.backend == "transformers"
    assert batch.used_fallback is False
    assert len(batch.vectors) == 1

    reference_vector = reference_model.encode(
        [text], convert_to_numpy=True, normalize_embeddings=False
    )[0].tolist()

    assert batch.vectors[0] == pytest.approx(reference_vector, rel=1e-6, abs=1e-6)


@pytest.fixture(scope="session")
def dolphin_x1_tiny_embedder() -> DolphinX1Embedder:
    """Return a Dolphin-X1 embedder backed by a tiny reference checkpoint."""

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    embedder = DolphinX1Embedder(
        settings=Settings(),
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        device="cpu",
        batch_size=2,
        max_length=64,
        target_dimension=3072,
        torch_dtype="float32",
    )

    try:
        embedder.encode(["warmup sentence for dolphin-x1 embeddings"])
    except EmbeddingBackendError as exc:  # pragma: no cover - dependency missing
        pytest.skip(f"Unable to load Dolphin-X1 embedding model: {exc}")
    except OSError as exc:  # pragma: no cover - HF download/IO failure
        pytest.skip(f"Dolphin-X1 embedding model unavailable: {exc}")

    return embedder


def test_dolphin_x1_embedder_respects_configured_dimension(
    dolphin_x1_tiny_embedder: DolphinX1Embedder,
) -> None:
    vectors = dolphin_x1_tiny_embedder.encode(["alpha", "beta"])

    assert len(vectors) == 2
    assert {len(vector) for vector in vectors} == {
        dolphin_x1_tiny_embedder.vector_dimension
    }

    torch_module, model, _ = dolphin_x1_tiny_embedder._ensure_model_components()
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if (
        isinstance(hidden_size, int)
        and hidden_size < dolphin_x1_tiny_embedder.vector_dimension
    ):
        tail_index = hidden_size
        for vector in vectors:
            assert all(abs(component) < 1e-6 for component in vector[tail_index:])


def test_dolphin_x1_embedder_matches_manual_mean_pool(
    dolphin_x1_tiny_embedder: DolphinX1Embedder,
) -> None:
    torch_module, model, tokenizer = dolphin_x1_tiny_embedder._ensure_model_components()
    text = "verifying dolphin-x1 pooling"

    reference_vectors = dolphin_x1_tiny_embedder.encode([text])
    assert len(reference_vectors) == 1
    reference_vector = reference_vectors[0]

    system_prompt = getattr(
        dolphin_x1_tiny_embedder._settings, "llm2vec_instruction", None
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
        max_length=dolphin_x1_tiny_embedder.max_length,
    )
    prepared = {
        name: (
            tensor.to(dolphin_x1_tiny_embedder.device)
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
    if manual_vector.shape[-1] > dolphin_x1_tiny_embedder.vector_dimension:
        manual_vector = manual_vector[: dolphin_x1_tiny_embedder.vector_dimension]
    elif manual_vector.shape[-1] < dolphin_x1_tiny_embedder.vector_dimension:
        pad = torch_module.zeros(
            dolphin_x1_tiny_embedder.vector_dimension - manual_vector.shape[-1],
            dtype=manual_vector.dtype,
            device=manual_vector.device,
        )
        manual_vector = torch_module.cat((manual_vector, pad), dim=0)

    assert reference_vector == pytest.approx(manual_vector.tolist(), abs=1e-5)


def test_dolphin_x1_embedder_embeddings_are_deterministic(
    dolphin_x1_tiny_embedder: DolphinX1Embedder,
) -> None:
    text = "determinism check for dolphin embeddings"

    first = dolphin_x1_tiny_embedder.encode([text])[0]
    second = dolphin_x1_tiny_embedder.encode([text])[0]

    assert len(first) == dolphin_x1_tiny_embedder.vector_dimension
    assert len(second) == dolphin_x1_tiny_embedder.vector_dimension
    assert first == pytest.approx(second, rel=1e-6, abs=1e-6)
    assert _vector_norm(first) == pytest.approx(
        _vector_norm(second), rel=1e-6, abs=1e-6
    )
    assert all(math.isfinite(component) for component in first)


def test_dolphin_x1_embedder_batch_determinism(
    dolphin_x1_tiny_embedder: DolphinX1Embedder,
) -> None:
    texts = [
        "batch determinism check 1",
        "batch determinism check 2",
        "batch determinism check 3",
        "batch determinism check 4",
        "batch determinism check 5",
    ]

    for batch_size in (1, 2, len(texts)):
        payload = texts[:batch_size]
        first_batch = dolphin_x1_tiny_embedder.encode(payload)
        second_batch = dolphin_x1_tiny_embedder.encode(payload)

        assert len(first_batch) == len(payload)
        assert len(second_batch) == len(payload)

        for first_vector, second_vector in zip(first_batch, second_batch, strict=True):
            assert len(first_vector) == dolphin_x1_tiny_embedder.vector_dimension
            assert len(second_vector) == dolphin_x1_tiny_embedder.vector_dimension
            assert first_vector == pytest.approx(second_vector, rel=1e-6, abs=1e-6)
            assert _vector_norm(first_vector) == pytest.approx(
                _vector_norm(second_vector), rel=1e-6, abs=1e-6
            )
            assert all(math.isfinite(component) for component in first_vector)


def test_dolphin_x1_embedder_normalises_empty_inputs(
    dolphin_x1_tiny_embedder: DolphinX1Embedder,
) -> None:
    payloads = [
        "",  # empty string
        " ",  # single space
        "   ",  # multiple spaces
        "\t",  # tab
        "\n",  # newline
        "\r\n",  # carriage return + newline
        "\u2003",  # em space (unicode whitespace)
        "\u2009",  # thin space (unicode whitespace)
        "\u202f",  # narrow no-break space (unicode whitespace)
        " \t\n\u2003\u2009\u202f",  # combination of whitespace
    ]

    vectors = dolphin_x1_tiny_embedder.encode(payloads)

    assert len(vectors) == len(payloads)
    for vector in vectors:
        assert len(vector) == dolphin_x1_tiny_embedder.vector_dimension
        assert all(math.isfinite(component) for component in vector)
        norm = _vector_norm(vector)
        assert norm >= 0.0
        # Ensure the prompt normalisation keeps vectors non-zero for downstream cosine similarity.
        assert norm > 0.0
