import pytest

from monGARS.core import neurones


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

    first = await system.encode("bonjour monde")
    second = await system.encode("bonjour monde")

    assert first == [0.1, 0.2, 0.3]
    assert second == first
    assert calls == ["bonjour monde"]


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
    vector = await system.encode("Salut")

    assert vector == [0.4, -0.6]
    assert dummy_model.calls == [("Salut", True)]


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
