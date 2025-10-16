from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient

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
