"""Tests for Ollama client and module caching in :mod:`monGARS.core.embeddings`."""

from __future__ import annotations

import asyncio
import importlib
import logging
import types

import pytest

from monGARS.core.embeddings import LLM2VecEmbedder


class _DummySettings:
    llm2vec_max_concurrency = 1
    llm2vec_instruction = "test"
    llm2vec_max_batch_size = 1
    llm2vec_vector_dimensions = 4
    SECRET_KEY = "dummy"
    llm2vec_base_model = "base"
    llm2vec_encoder = "encoder"
    llm2vec_device_map = None
    llm2vec_torch_dtype = None
    llm2vec_pooling_strategy = None
    llm2vec_tokenizer_name = "tokenizer"
    llm2vec_tokenizer_revision = None
    llm2vec_revision = None
    llm2vec_loader = None
    llm2vec_trust_remote_code = False
    llm2vec_use_safetensors = True
    ollama_host = "http://localhost:11434"
    ollama_embedding_model = "test-model"
    ollama_embedding_dimensions = 3


@pytest.mark.asyncio
async def test_ollama_client_initialised_once(caplog: pytest.LogCaptureFixture) -> None:
    embedder = LLM2VecEmbedder(settings=_DummySettings(), backend="ollama")
    caplog.set_level(logging.DEBUG)

    creation_hosts: list[str | None] = []

    class _FakeClient:
        def __init__(self, host: str | None = None) -> None:
            creation_hosts.append(host)
            self.host = host

        def embed(
            self, **_: object
        ) -> dict[str, list[list[float]]]:  # pragma: no cover - helper
            return {"embeddings": [[0.0]]}

    module = types.SimpleNamespace(Client=_FakeClient)

    client_one, client_two = await asyncio.gather(
        embedder._ensure_ollama_client(module),
        embedder._ensure_ollama_client(module),
    )

    assert client_one is client_two
    assert creation_hosts == ["http://localhost:11434"]

    initialising_logs = [
        record
        for record in caplog.records
        if record.msg == "llm2vec.ollama.client.initialising"
    ]
    initialised_logs = [
        record
        for record in caplog.records
        if record.msg == "llm2vec.ollama.client.initialised"
    ]
    assert len(initialising_logs) == 1
    assert len(initialised_logs) == 1


def test_ollama_module_import_cached(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    embedder = LLM2VecEmbedder(settings=_DummySettings(), backend="ollama")
    caplog.set_level(logging.DEBUG)

    imported_modules: list[str] = []
    located_modules: list[str] = []

    module = types.SimpleNamespace(Client=lambda **_: None)

    def _fake_find_spec(name: str):
        located_modules.append(name)
        return object()

    def _fake_import_module(name: str):
        imported_modules.append(name)
        return module

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    resolved_one = embedder._ensure_ollama_module()
    resolved_two = embedder._ensure_ollama_module()

    assert resolved_one is module
    assert resolved_two is module
    assert located_modules == ["ollama"]
    assert imported_modules == ["ollama"]

    start_logs = [
        record
        for record in caplog.records
        if record.msg == "llm2vec.ollama.module_import.start"
    ]
    success_logs = [
        record
        for record in caplog.records
        if record.msg == "llm2vec.ollama.module_import.success"
    ]
    reuse_logs = [
        record
        for record in caplog.records
        if record.msg == "llm2vec.ollama.module_import.reuse"
    ]

    assert len(start_logs) == 1
    assert len(success_logs) == 1
    assert len(reuse_logs) == 1
