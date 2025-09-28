import copy
import math

import pytest

from monGARS.core import neurones


@pytest.fixture(autouse=True)
def stub_tiered_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTieredCache:
        def __init__(self) -> None:
            self.time = 0.0
            self.store: dict[str, tuple[float | None, object]] = {}
            self.caches = [self]

        async def get(self, key: str) -> object | None:
            entry = self.store.get(key)
            if entry is None:
                return None
            expires_at, payload = entry
            if expires_at is not None and expires_at <= self.time:
                self.store.pop(key, None)
                return None
            return copy.deepcopy(payload)

        async def set(self, key: str, value: object, ttl: int | None = None) -> None:
            expires_at = self.time + ttl if ttl else None
            self.store[key] = (expires_at, copy.deepcopy(value))

        async def delete(self, key: str) -> None:
            self.store.pop(key, None)

    monkeypatch.setattr(neurones, "TieredCache", DummyTieredCache)


@pytest.mark.asyncio
async def test_encode_uses_fallback_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(neurones, "SentenceTransformer", None, raising=False)

    calls = []

    def fake_fallback(self: neurones.EmbeddingSystem, text: str) -> list[float]:
        calls.append(text)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(neurones.EmbeddingSystem, "_fallback_embedding", fake_fallback)

    system = neurones.EmbeddingSystem(cache_ttl=60)

    first_vector, first_fallback = await system.encode("bonjour monde")
    second_vector, second_fallback = await system.encode("bonjour monde")

    assert first_vector == [0.1, 0.2, 0.3]
    assert second_vector == first_vector
    assert first_fallback is True
    assert second_fallback is True
    assert calls == ["bonjour monde"]


@pytest.mark.asyncio
async def test_encode_empty_and_whitespace_returns_normalized_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = neurones.EmbeddingSystem()

    empty_vector, empty_fallback = await system.encode("")
    whitespace_vector, whitespace_fallback = await system.encode("   ")

    assert empty_vector == whitespace_vector
    assert len(empty_vector) == system._fallback_dimensions  # type: ignore[attr-defined]
    magnitude = math.sqrt(sum(value * value for value in empty_vector))
    assert math.isclose(magnitude, 1.0, rel_tol=1e-6)
    assert empty_fallback is False
    assert whitespace_fallback is False


@pytest.mark.asyncio
async def test_encode_with_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyModel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls: list[tuple[str, bool]] = []

        def encode(self, text: str, normalize_embeddings: bool = True) -> list[float]:
            self.calls.append((text, normalize_embeddings))
            return [0.4, -0.6]

        def get_sentence_embedding_dimension(self) -> int:
            return 2

    dummy_model = DummyModel("dummy")

    async def fake_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    def fake_sentence_transformer(name: str) -> DummyModel:
        return dummy_model

    def fail_fallback(self: neurones.EmbeddingSystem, text: str) -> list[float]:
        raise AssertionError("Fallback should not be used when model is available")

    monkeypatch.setattr(neurones, "SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(neurones.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(neurones.EmbeddingSystem, "_fallback_embedding", fail_fallback)

    system = neurones.EmbeddingSystem()
    vector, used_fallback = await system.encode("Salut")

    assert vector == [0.4, -0.6]
    assert used_fallback is False
    assert dummy_model.calls == [("Salut", True)]


@pytest.mark.asyncio
async def test_embedding_cache_expiry_and_recaching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyModel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls: list[tuple[str, bool]] = []

        def encode(self, text: str, normalize_embeddings: bool = True) -> list[float]:
            self.calls.append((text, normalize_embeddings))
            return [0.4, -0.6]

        def get_sentence_embedding_dimension(self) -> int:
            return 2

    dummy_model = DummyModel("dummy")

    async def fake_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    def fake_sentence_transformer(name: str) -> DummyModel:
        return dummy_model

    def fail_fallback(self: neurones.EmbeddingSystem, text: str) -> list[float]:
        raise AssertionError("Fallback should not be used when model is available")

    monkeypatch.setattr(neurones, "SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(neurones.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(neurones.EmbeddingSystem, "_fallback_embedding", fail_fallback)

    system = neurones.EmbeddingSystem(cache_ttl=1)

    dummy_model.calls.clear()
    vector1, fallback1 = await system.encode("Bonjour")
    assert vector1 == [0.4, -0.6]
    assert fallback1 is False
    assert dummy_model.calls == [("Bonjour", True)]

    dummy_model.calls.clear()
    vector2, fallback2 = await system.encode("Bonjour")
    assert vector2 == [0.4, -0.6]
    assert fallback2 is False
    assert dummy_model.calls == []

    # Advance the stub cache time to simulate expiry
    system._cache.time += 2  # type: ignore[attr-defined]

    dummy_model.calls.clear()
    vector3, fallback3 = await system.encode("Bonjour")
    assert vector3 == [0.4, -0.6]
    assert fallback3 is False
    assert dummy_model.calls == [("Bonjour", True)]


@pytest.mark.asyncio
async def test_noop_driver_returns_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(neurones, "AsyncGraphDatabase", None, raising=False)
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    system = neurones.EmbeddingSystem()

    async with system.driver.session() as session:
        result = await session.run("RETURN 1 AS exists")
        record = await result.single()

    assert record == {"exists": False}
