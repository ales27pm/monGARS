"""Tests for :mod:`monGARS.mlops.artifacts`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monGARS.mlops.artifacts import (
    WrapperConfig,
    render_output_bundle_readme,
    render_project_wrapper,
    write_wrapper_bundle,
)


@pytest.fixture()
def sample_config(tmp_path: Path) -> WrapperConfig:
    return WrapperConfig(
        base_model_id="base/model",
        lora_dir=(tmp_path / "lora").resolve(),
        max_seq_len=1024,
        vram_budget_mb=7300,
        offload_dir=(tmp_path / "offload").resolve(),
    )


def test_render_project_wrapper_includes_expected_sections(
    sample_config: WrapperConfig,
) -> None:
    rendered = render_project_wrapper(sample_config)
    assert "class ChatAndEmbed" in rendered
    assert f"BASE_MODEL_ID = '{sample_config.base_model_id}'" in rendered
    assert "def _bnb4()" in rendered
    assert "os.makedirs" in rendered
    assert "prompt_length =" in rendered
    assert "def embed(self, texts: Iterable[str])" in rendered


def test_write_wrapper_bundle_writes_all_files(
    tmp_path: Path, sample_config: WrapperConfig
) -> None:
    paths = write_wrapper_bundle(sample_config, tmp_path)
    module_text = paths["module"].read_text(encoding="utf-8")
    config_json = json.loads(paths["config"].read_text(encoding="utf-8"))
    readme_text = paths["readme"].read_text(encoding="utf-8")

    assert "ChatAndEmbed" in module_text
    assert config_json["base_model_id"] == sample_config.base_model_id
    assert config_json["vram_budget_mb"] == sample_config.vram_budget_mb
    assert "Wrapper Integration" in readme_text


@pytest.mark.parametrize(
    ("merged", "gguf"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_render_output_bundle_readme_flags(
    sample_config: WrapperConfig, merged: bool, gguf: bool
) -> None:
    readme = render_output_bundle_readme(
        sample_config,
        merged_fp16=merged,
        gguf_enabled=gguf,
        gguf_method="q4_k_m",
    )
    if merged:
        assert "merged_fp16" in readme
    else:
        assert "merged_fp16" not in readme
    if gguf:
        assert "gguf/" in readme
    else:
        assert "gguf/" not in readme
