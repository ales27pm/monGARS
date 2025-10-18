"""Tests for GGUF export utilities."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_export_to_gguf_invokes_exporter(monkeypatch, tmp_path):
    from monGARS.mlops import exporters

    calls: dict[str, object] = {}

    class DummyModel:
        def to(self, *_args, **_kwargs):  # pragma: no cover - exercised implicitly
            calls["moved_to_cpu"] = True

    class DummyTokenizer:
        def __init__(self, name: str):
            self.name_or_path = name

    class DummyExporter:
        def __init__(self, *, model, tokenizer):
            calls["model"] = model
            calls["tokenizer"] = tokenizer

        def export(self, destination: str) -> None:
            calls["destination"] = destination
            Path(destination).write_bytes(b"GGUF")

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda name, **_: DummyModel(),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name, **_: DummyTokenizer(name),
    )
    monkeypatch.setattr("transformers.GgufExporter", DummyExporter, raising=False)

    output_file = exporters.export_to_gguf("dummy/model", tmp_path / "gguf")

    assert output_file.exists()
    assert output_file.read_bytes() == b"GGUF"
    assert calls["model"].__class__ is DummyModel
    assert isinstance(calls["tokenizer"], DummyTokenizer)
    assert calls["destination"].endswith("dummy_model.gguf")


def test_export_to_gguf_missing_exporter(monkeypatch):
    from monGARS.mlops import exporters

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *_: None
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_: None)

    monkeypatch.setattr(
        exporters, "_GGUF_EXPORTER_CANDIDATES", ("nonexistent.module.Exporter",)
    )

    with pytest.raises(RuntimeError):
        exporters.export_to_gguf("dummy/model", "out.gguf")
