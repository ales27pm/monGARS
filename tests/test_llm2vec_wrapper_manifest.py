from __future__ import annotations

import json
from pathlib import Path

from scripts.export_llm2vec_wrapper import load_wrapper_config, write_wrapper


def test_load_wrapper_config_merges_training_metadata(tmp_path: Path) -> None:
    raw_config = {
        "base_model_id": "custom/dolphin",
        "embedding_options": {"max_length": 256, "normalise": True},
        "artifacts": {"adapter_subdir": "adapters"},
    }
    (tmp_path / "wrapper_config.json").write_text(json.dumps(raw_config))

    config = load_wrapper_config(tmp_path)

    assert config["base_model_id"] == "custom/dolphin"
    assert config["embedding_options"]["max_length"] == 256
    assert config["embedding_options"]["normalise"] is True
    assert config["artifacts"]["adapter_subdir"] == "adapters"
    assert config["artifacts"]["tokenizer_dir"] == str(tmp_path / "tokenizer")
    assert config["wrapper_metadata"]["export_root"] == str(tmp_path)


def test_write_wrapper_generates_manifest_and_python_module(tmp_path: Path) -> None:
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "lora_adapter").mkdir()

    wrap_dir = write_wrapper(tmp_path, base_model="custom/dolphin")

    config_path = wrap_dir / "config.json"
    module_path = wrap_dir / "llm2vec_wrapper.py"
    readme_path = wrap_dir / "README.md"

    assert config_path.exists()
    assert module_path.exists()
    assert readme_path.exists()

    config = json.loads(config_path.read_text())
    assert config["base_model_id"] == "custom/dolphin"
    assert (
        config["wrapper_metadata"]["generated_by"]
        == "scripts/export_llm2vec_wrapper.py"
    )
    assert "generated_at" in config["wrapper_metadata"]
