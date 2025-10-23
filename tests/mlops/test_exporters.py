"""Tests for GGUF export utilities."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_export_to_gguf_invokes_exporter(monkeypatch, tmp_path):
    from monGARS.mlops import exporters

    exporters._load_gguf_exporter_cached.cache_clear()

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

        def export(
            self, destination: str, *, quantization_method: str | None = None
        ) -> None:
            calls["destination"] = destination
            calls["quantization_method"] = quantization_method
            calls["method"] = "export"
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

    result = exporters.export_to_gguf(
        "dummy/model",
        tmp_path / "gguf",
        quantization_method="q4_k_m",
    )

    assert result.path.exists()
    assert result.path.read_bytes() == b"GGUF"
    assert calls["model"].__class__ is DummyModel
    assert isinstance(calls["tokenizer"], DummyTokenizer)
    assert calls["destination"].endswith("dummy_model.gguf")
    assert calls["quantization_method"] == "q4_k_m"
    assert calls["method"] == "export"
    assert result.method == "export"
    assert result.exporter == "transformers.GgufExporter"
    assert result.quantization_method == "q4_k_m"
    assert str(result) == str(result.path)


def test_export_to_gguf_ignores_quantization_when_not_supported(monkeypatch, tmp_path):
    from monGARS.mlops import exporters

    exporters._load_gguf_exporter_cached.cache_clear()

    calls: dict[str, object] = {}

    class DummyExporter:
        def __init__(self, *, model, tokenizer):  # pragma: no cover - set via kwargs
            calls["model"] = model
            calls["tokenizer"] = tokenizer

        def export(self, destination: str) -> None:
            calls["method"] = "export"
            Path(destination).write_bytes(b"GGUF")

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda name, **_: object(),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name, **_: object(),
    )
    monkeypatch.setattr("transformers.GgufExporter", DummyExporter, raising=False)

    result = exporters.export_to_gguf(
        "dummy/model",
        tmp_path / "gguf",
        quantization_method="q5_1",
    )

    assert result.path.exists()
    assert result.path.read_bytes() == b"GGUF"
    assert calls["method"] == "export"
    assert result.method == "export"
    assert result.quantization_method == "q5_1"


def test_export_to_gguf_falls_back_to_export_model(monkeypatch, tmp_path):
    from monGARS.mlops import exporters

    exporters._load_gguf_exporter_cached.cache_clear()

    calls: dict[str, object] = {}

    class DummyExporter:
        def __init__(self, *, model, tokenizer):  # pragma: no cover - kwargs only
            calls["model"] = model
            calls["tokenizer"] = tokenizer

        def export_model(
            self, destination: str, *, quantization_method: str | None = None
        ) -> None:
            calls["destination"] = destination
            calls["quantization_method"] = quantization_method
            calls["method"] = "export_model"
            Path(destination).write_bytes(b"GGUF")

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda name, **_: object(),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name, **_: object(),
    )
    monkeypatch.setattr("transformers.GgufExporter", DummyExporter, raising=False)

    result = exporters.export_to_gguf(
        "dummy/model",
        tmp_path / "gguf",
        quantization_method="q4_k_s",
    )

    assert result.path.exists()
    assert result.path.read_bytes() == b"GGUF"
    assert calls["method"] == "export_model"
    assert result.method == "export_model"
    assert result.exporter == "transformers.GgufExporter"


def test_export_to_gguf_missing_exporter(monkeypatch):
    from monGARS.mlops import exporters

    exporters._load_gguf_exporter_cached.cache_clear()

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", lambda *_: None
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_: None)

    monkeypatch.setattr(
        exporters, "_GGUF_EXPORTER_CANDIDATES", ("nonexistent.module.Exporter",)
    )

    with pytest.raises(RuntimeError):
        exporters.export_to_gguf("dummy/model", "out.gguf")


def test_load_gguf_exporter_returns_metadata(monkeypatch):
    from monGARS.mlops import exporters

    exporters._load_gguf_exporter_cached.cache_clear()

    class DummyExporter:
        pass

    def fake_locate(path: str):
        return DummyExporter if path == "pkg.Dummy" else None

    monkeypatch.setattr(exporters, "_locate_symbol", fake_locate)

    info = exporters._load_gguf_exporter(("pkg.Dummy", "pkg.Other"))

    assert isinstance(info, exporters.GGUFExporterInfo)
    assert info is not None
    assert info.qualified_name == "pkg.Dummy"
    assert info.factory is DummyExporter

    exporters._load_gguf_exporter_cached.cache_clear()
