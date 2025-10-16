from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from monGARS.config import get_settings
from monGARS.core.embeddings import EmbeddingBatch, LLM2VecEmbedder
from scripts.run_llm2vec_service import EmbeddingService, create_app


class DummyWrapper:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool | None]] = []

    def embed(self, texts: list[str], normalise: bool | None = None) -> torch.Tensor:
        self.calls.append((texts, normalise))
        return torch.ones(len(texts), 3)


def test_embedding_service_exposes_health_and_embed(tmp_path: Path) -> None:
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()

    config = {
        "base_model_id": "sample/dolphin",
        "embedding_backend": "huggingface",
        "embedding_options": {"max_length": 16, "normalise": False},
    }
    (tmp_path / "wrapper_config.json").write_text(json.dumps(config))
    (wrapper_dir / "config.json").write_text(json.dumps(config))

    dummy = DummyWrapper()
    service = EmbeddingService(
        tmp_path,
        prefer_merged=False,
        device="cpu",
        load_in_4bit=None,
        wrapper_factory=lambda: dummy,
    )
    app = create_app(service)
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["model"] == "sample/dolphin"

    response = client.post(
        "/embed",
        json={"inputs": ["hello", "world"], "normalise": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["dims"] == 3
    assert payload["normalised"] is True
    assert dummy.calls[0][0] == ["hello", "world"]
    assert dummy.calls[0][1] is True


def test_embedding_service_ollama_backend(monkeypatch) -> None:
    async def fake_encode_batch(_self, texts, *, instruction=None):
        del instruction
        return EmbeddingBatch(
            vectors=[[float(index), float(index + 1)] for index, _ in enumerate(texts)],
            used_fallback=False,
        )

    monkeypatch.setattr(
        LLM2VecEmbedder, "encode_batch", fake_encode_batch, raising=True
    )

    service = EmbeddingService(
        None,
        backend="ollama",
        settings=get_settings(),
    )

    assert service.backend == "ollama"

    app = create_app(service)
    client = TestClient(app)

    response = client.post("/embed", json={"inputs": ["alpha", "beta"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "ollama"
    assert payload["count"] == 2
    assert payload["dims"] == 2
    assert payload["normalised"] is False
    assert payload["vectors"] == [[0.0, 1.0], [1.0, 2.0]]


def test_embedding_service_unsupported_backend_warns(tmp_path: Path, caplog) -> None:
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()
    config = {
        "base_model_id": "fallback/model",
        "embedding_backend": "huggingface",
        "embedding_options": {"normalise": False},
    }
    (tmp_path / "wrapper_config.json").write_text(json.dumps(config))
    (wrapper_dir / "config.json").write_text(json.dumps(config))

    caplog.set_level("WARNING")

    service = EmbeddingService(
        tmp_path,
        backend="unsupported-backend",
        settings=get_settings(),
    )

    assert service.backend == "huggingface"
    assert any(
        record.getMessage() == "llm2vec.embedding.backend.unsupported"
        for record in caplog.records
    )
