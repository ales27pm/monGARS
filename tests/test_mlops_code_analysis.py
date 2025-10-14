from pathlib import Path

from monGARS.mlops.code_analysis import (
    LLMUsage,
    ModuleInteraction,
    build_strategy_recommendation,
    render_usage_report,
    scan_llm_usage,
    scan_module_interactions,
)


def test_scan_llm_usage_detects_transformers(tmp_path: Path) -> None:
    source = """from transformers import AutoModel\n\n\ndef build_model():\n    return AutoModel.from_pretrained('hf/test')\n"""
    module_path = tmp_path / "module.py"
    module_path.write_text(source)

    usages = scan_llm_usage(tmp_path)
    assert len(usages) == 1
    usage = usages[0]
    assert usage.framework == "transformers"
    assert usage.call == "AutoModel.from_pretrained"
    assert "0004" in usage.snippet  # snippet should include line numbers

    report = render_usage_report(usages)
    assert "transformers" in report
    assert "AutoModel.from_pretrained" in report


def test_build_strategy_recommendation_varies_by_framework(tmp_path: Path) -> None:
    usage = LLMUsage(
        file_path=Path("foo.py"),
        line=42,
        symbol="Trainer.build",
        framework="llm2vec",
        call="LLM2Vec",
        snippet="0040: pass",
    )
    text = build_strategy_recommendation(usage)
    assert "LLM2Vec" in text
    assert "pooled embeddings" in text


def test_scan_module_interactions_handles_relative_imports(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "b.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (package / "c.py").write_text("VALUE = 2\n", encoding="utf-8")
    (package / "a.py").write_text(
        "from .b import helper as local_helper\nimport pkg.c as config\n",
        encoding="utf-8",
    )

    interactions = scan_module_interactions(tmp_path, packages=("pkg",))
    assert any(
        isinstance(item, ModuleInteraction)
        and item.source_module.endswith("pkg.a")
        and item.target_module == "pkg.b"
        and "local_helper" in item.import_names
        for item in interactions
    )
    assert any(
        item.target_module == "pkg.c" and item.kind == "import" for item in interactions
    )
