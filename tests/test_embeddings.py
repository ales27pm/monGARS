"""Tests for the LLM2Vec embedder utilities."""

import math
import sys
from types import SimpleNamespace

import pytest

from monGARS.config import Settings
from monGARS.core import embeddings as embeddings_module
from monGARS.core.embeddings import (
    DolphinX1Embedder,
    EmbeddingBackendError,
    EmbeddingIdentity,
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
        base_vector = [float(len(self.calls)), 42.0, 84.0]
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
async def test_encode_batch_chunks_requests_with_exact_dimensions() -> None:
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
    assert result.identity == EmbeddingIdentity(
        backend="huggingface",
        model=settings.llm2vec_base_model,
        revision=settings.embedding_model_revision,
        dimension=3,
    )
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
    assert second.identity == first.identity


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
async def test_encode_batch_rejects_invalid_vectors_without_fallback() -> None:
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
    with pytest.raises(EmbeddingBackendError, match="dimension"):
        await embedder.encode_batch(payloads)


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
                "model": "stub",
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
    assert batch.identity == EmbeddingIdentity(
        backend="dolphin-x1-llm2vec",
        model="stub",
        revision="unversioned",
        dimension=3,
    )

    assert module.created_clients  # client instantiated lazily
    client = module.created_clients[0]
    assert client.get_calls == ["/health"]
    assert client.post_calls == [("/embed", {"texts": payloads})]


@pytest.mark.asyncio
async def test_dolphin_service_backend_rejects_invalid_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeHTTPXModule()
    module.health_queue.append(({"status": "ok", "model": "stub", "dimension": 3}, 200))
    module.post_queue.append(
        (
            {
                "embeddings": [["not", "numbers"]],
                "model": "stub",
                "dimension": 3,
            },
            200,
        )
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

    with pytest.raises(EmbeddingBackendError, match="non-numeric"):
        await embedder.encode_batch(["invalid vector"])

    assert module.created_clients
    client = module.created_clients[0]
    assert client.post_calls == [("/embed", {"texts": ["invalid vector"]})]


@pytest.mark.asyncio
async def test_dolphin_service_rejects_health_dimension_mismatch_before_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeHTTPXModule()
    module.health_queue.append(({"status": "ok", "model": "stub", "dimension": 4}, 200))
    monkeypatch.setitem(sys.modules, "httpx", module)
    settings = Settings(
        embedding_backend="dolphin-x1-llm2vec",
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = LLM2VecEmbedder(settings=settings)

    with pytest.raises(EmbeddingBackendError, match="4 != 3"):
        await embedder.encode_batch(["dimension mismatch"])

    client = module.created_clients[0]
    assert client.post_calls == []


@pytest.mark.asyncio
async def test_dolphin_service_does_not_reuse_cache_across_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeHTTPXModule()
    module.health_queue.append(
        (
            {
                "status": "ok",
                "model": "model-a",
                "revision": "rev-a",
                "dimension": 3,
            },
            200,
        )
    )
    module.post_queue.extend(
        [
            (
                {
                    "embeddings": [[1.0, 2.0, 3.0]],
                    "model": "model-a",
                    "revision": "rev-a",
                    "dimension": 3,
                },
                200,
            ),
            (
                {
                    "embeddings": [[4.0, 5.0, 6.0]],
                    "model": "model-b",
                    "revision": "rev-b",
                    "dimension": 3,
                },
                200,
            ),
        ]
    )
    monkeypatch.setitem(sys.modules, "httpx", module)
    settings = Settings(
        embedding_backend="dolphin-x1-llm2vec",
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = LLM2VecEmbedder(settings=settings)

    first = await embedder.encode_batch(["same payload"])
    second = await embedder.encode_batch(["same payload"])

    assert first.identity is not None
    assert second.identity is not None
    assert first.identity.cache_key != second.identity.cache_key
    assert first.vectors != second.vectors
    client = module.created_clients[0]
    assert client.post_calls == [
        ("/embed", {"texts": ["same payload"]}),
        ("/embed", {"texts": ["same payload"]}),
    ]


@pytest.mark.asyncio
async def test_encode_batch_rejects_blank_inputs_without_vectors() -> None:
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
    with pytest.raises(EmbeddingBackendError, match="blank"):
        await embedder.encode_batch(payloads)
    assert manager.calls == []


@pytest.mark.asyncio
async def test_encode_batch_fails_when_manager_not_ready() -> None:
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

    with pytest.raises(EmbeddingBackendError, match="not ready"):
        await embedder.encode_batch(["alpha", "beta"])
    assert manager.calls == 0


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
async def test_encode_batch_renders_chatml_before_fail_closed_not_ready(
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
    with pytest.raises(EmbeddingBackendError, match="not ready"):
        await embedder.encode_batch(payloads)

    assert manager.calls == 0
    assert len(captured) == len(payloads)
    blocks = [captured]

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
async def test_encode_batch_cache_is_partitioned_by_model_revision() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        embedding_model_revision="rev-a",
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _RecordingManager()
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    payload = ["shared-text"]
    first = await embedder.encode_batch(payload)
    settings.embedding_model_revision = "rev-b"
    second = await embedder.encode_batch(payload)

    assert first.identity is not None
    assert second.identity is not None
    assert first.identity.revision == "rev-a"
    assert second.identity.revision == "rev-b"
    assert len(manager.calls) == 2


def test_sentence_transformer_load_is_pinned_and_revision_partitioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, str | None]] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, *, revision: str | None = None) -> None:
            created.append((model_name, revision))

        def encode(self, *_args, **_kwargs):
            return [[1.0, 2.0, 3.0]]

    fake_module = SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setattr(
        embeddings_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "sentence_transformers" else None,
    )
    monkeypatch.setattr(
        embeddings_module.importlib,
        "import_module",
        lambda name: (
            fake_module
            if name == "sentence_transformers"
            else pytest.fail(f"unexpected import: {name}")
        ),
    )
    embeddings_module._SENTENCE_TRANSFORMER_CACHE.clear()
    first = Settings(
        transformers_embedding_model="example/model",
        embedding_model_revision="commit-a",
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    second = first.model_copy(update={"embedding_model_revision": "commit-b"})

    embeddings_module._encode_with_sentence_transformers(["one"], None, first)
    embeddings_module._encode_with_sentence_transformers(["one"], None, first)
    embeddings_module._encode_with_sentence_transformers(["one"], None, second)

    assert created == [
        ("example/model", "commit-a"),
        ("example/model", "commit-b"),
    ]


@pytest.mark.asyncio
async def test_encode_batch_cache_is_partitioned_by_active_encoder() -> None:
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    manager = _RecordingManager()
    manager.base_model_path = "base-model"
    manager.encoder_path = "encoder-a"
    embedder = LLM2VecEmbedder(
        settings=settings,
        neuron_manager_factory=lambda: manager,
    )

    first = await embedder.encode_batch(["shared-text"])
    manager.encoder_path = "encoder-b"
    second = await embedder.encode_batch(["shared-text"])

    assert first.identity is not None
    assert second.identity is not None
    assert first.identity.model.endswith("encoder-a")
    assert second.identity.model.endswith("encoder-b")
    assert len(manager.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_encode_batch_rejects_non_finite_values(value: float) -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _NonFiniteManager()
    manager.return_value = value
    embedder = LLM2VecEmbedder(
        settings=settings, neuron_manager_factory=lambda: manager
    )

    with pytest.raises(EmbeddingBackendError, match="non-finite"):
        await embedder.encode_batch(["alpha"])


@pytest.mark.asyncio
async def test_encode_batch_rejects_dimension_mismatch_without_resize() -> None:
    settings = Settings(
        llm2vec_max_batch_size=4,
        llm2vec_max_concurrency=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
        debug=True,
    )
    manager = _DeterministicManager([[1.0, 2.0, 3.0, 4.0]])
    embedder = LLM2VecEmbedder(
        settings=settings,
        neuron_manager_factory=lambda: manager,
    )

    with pytest.raises(EmbeddingBackendError, match="4 != 3"):
        await embedder.encode_batch(["payload"])


@pytest.mark.asyncio
async def test_encode_batch_rejects_vector_count_mismatch() -> None:
    settings = Settings(
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    manager = _DeterministicManager([[1.0, 2.0, 3.0]])
    embedder = LLM2VecEmbedder(
        settings=settings,
        neuron_manager_factory=lambda: manager,
    )

    with pytest.raises(EmbeddingBackendError, match="invalid vector count"):
        await embedder.encode_batch(["first", "second"])


@pytest.mark.asyncio
async def test_encode_batch_rejects_identity_change_between_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        llm2vec_max_batch_size=1,
        llm2vec_vector_dimensions=3,
        SECRET_KEY="test",  # noqa: S106 - test configuration only
    )
    embedder = LLM2VecEmbedder(
        settings=settings,
        neuron_manager_factory=_RecordingManager,
    )
    identities = iter(
        [
            EmbeddingIdentity("huggingface", "model", "rev-a", 3),
            EmbeddingIdentity("huggingface", "model", "rev-b", 3),
        ]
    )

    async def _dispatch(_chunk, _prompt):
        return [[1.0, 2.0, 3.0]], next(identities)

    monkeypatch.setattr(embedder, "_dispatch_backend", _dispatch)

    with pytest.raises(EmbeddingBackendError, match="identity changed"):
        await embedder.encode_batch(["first", "second"])


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
        target_dimension=16,
        torch_dtype="float32",
    )

    try:
        embedder.encode(["warmup sentence for dolphin-x1 embeddings"])
    except EmbeddingBackendError as exc:  # pragma: no cover - dependency missing
        pytest.skip(f"Unable to load Dolphin-X1 embedding model: {exc}")
    except OSError as exc:  # pragma: no cover - HF download/IO failure
        pytest.skip(f"Dolphin-X1 embedding model unavailable: {exc}")

    return embedder


def test_dolphin_x1_embedder_preserves_native_configured_dimension(
    dolphin_x1_tiny_embedder: DolphinX1Embedder,
) -> None:
    vectors = dolphin_x1_tiny_embedder.encode(["alpha", "beta"])

    assert len(vectors) == 2
    assert {len(vector) for vector in vectors} == {
        dolphin_x1_tiny_embedder.vector_dimension
    }

    _, model, _ = dolphin_x1_tiny_embedder._ensure_model_components()
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    assert hidden_size == dolphin_x1_tiny_embedder.vector_dimension


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

    pooled = (
        (final_hidden * mask_tensor).sum(dim=1) / mask_tensor.sum(dim=1).clamp_min(1.0)
    )[0]
    manual_vector = pooled.detach().to(torch_module.float32)
    if manual_vector.device.type != "cpu":
        manual_vector = manual_vector.cpu()
    assert manual_vector.shape[-1] == dolphin_x1_tiny_embedder.vector_dimension

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


def test_dolphin_x1_embedder_rejects_blank_inputs(
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

    with pytest.raises(EmbeddingBackendError, match="blank"):
        dolphin_x1_tiny_embedder.encode(payloads)


def test_dolphin_x1_embedder_rejects_dimension_mismatch() -> None:
    torch_module = pytest.importorskip("torch")
    embedder = DolphinX1Embedder(settings=Settings(), target_dimension=3)

    with pytest.raises(EmbeddingBackendError, match="2 != 3"):
        embedder._prepare_output_vector(
            torch_module.tensor([1.0, 2.0]),
            torch_module,
        )


def test_dolphin_x1_embedder_rejects_non_finite_output() -> None:
    torch_module = pytest.importorskip("torch")
    embedder = DolphinX1Embedder(settings=Settings(), target_dimension=3)

    with pytest.raises(EmbeddingBackendError, match="non-finite"):
        embedder._prepare_output_vector(
            torch_module.tensor([1.0, float("nan"), 3.0]),
            torch_module,
        )
