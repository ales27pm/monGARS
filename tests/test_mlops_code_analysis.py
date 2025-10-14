from pathlib import Path

from monGARS.mlops.code_analysis import (
    LLMUsage,
    build_strategy_recommendation,
    render_usage_report,
    scan_llm_usage,
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
