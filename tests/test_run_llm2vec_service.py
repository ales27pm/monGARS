from __future__ import annotations

from fastapi.testclient import TestClient

from scripts import run_llm2vec_service


class DummyEmbeddingModel:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.embedding_dim = 3
        self.calls: list[tuple[list[str], int]] = []

    def embed_texts(self, texts, batch_size: int):
        payload = [str(text) for text in texts]
        self.calls.append((payload, batch_size))
        return [[float(index)] * self.embedding_dim for index, _ in enumerate(payload)]


def test_service_exposes_health_and_embeddings(monkeypatch) -> None:
    dummy = DummyEmbeddingModel()

    def factory(*args, **kwargs):
        return dummy

    monkeypatch.setattr(run_llm2vec_service, "EmbeddingModel", factory)

    cfg = run_llm2vec_service.ServiceConfig(
        model_dir="/tmp/model",
        host="127.0.0.1",
        port=8080,
        device="cpu",
        max_length=128,
        batch_size=4,
        normalize=True,
        pooling="mean",
        log_level="INFO",
    )

    app = run_llm2vec_service.create_app(cfg)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": "/tmp/model",
        "dimension": 3,
    }

    response = client.post("/embed", json={"texts": ["hello", "world"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dimension"] == 3
    assert payload["model"] == "/tmp/model"
    assert payload["embeddings"] == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert dummy.calls == [(["hello", "world"], 4)]


def test_service_rejects_empty_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        run_llm2vec_service, "EmbeddingModel", lambda *a, **kw: DummyEmbeddingModel()
    )

    cfg = run_llm2vec_service.ServiceConfig(
        model_dir="/tmp/model",
        host="127.0.0.1",
        port=8080,
        device="cpu",
        max_length=128,
        batch_size=1,
        normalize=True,
        pooling="mean",
        log_level="INFO",
    )

    app = run_llm2vec_service.create_app(cfg)
    client = TestClient(app)

    response = client.post("/embed", json={"texts": []})
    assert response.status_code == 400
    assert response.json()["detail"] == "Field 'texts' must be a non-empty list"
