"""Tests for artefact generation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from monGARS.mlops import artifacts


def test_build_adapter_summary_collects_metadata(tmp_path):
    adapter_dir = tmp_path / "adapter"
    weights_path = tmp_path / "weights.bin"
    wrapper_dir = tmp_path / "wrapper"

    labels = {"quality": "gold"}
    metrics = {"loss": 0.42}
    training = {"epochs": 3}

    summary = artifacts.build_adapter_summary(
        adapter_dir=adapter_dir,
        weights_path=weights_path,
        wrapper_dir=wrapper_dir,
        status="complete",
        labels=labels,
        metrics=metrics,
        training=training,
    )

    assert summary["status"] == "complete"
    assert summary["artifacts"] == {
        "adapter": str(adapter_dir),
        "weights": str(weights_path),
        "wrapper": str(wrapper_dir),
    }
    assert summary["labels"] == labels
    assert summary["metrics"] == metrics
    assert summary["training"] == training
    assert summary["labels"] is not labels
    assert summary["metrics"] is not metrics


def test_write_wrapper_bundle_generates_expected_files(tmp_path):
    config = artifacts.WrapperConfig(
        base_model_id="dummy/model",
        lora_dir=Path("/models/dummy"),
        max_seq_len=1024,
        vram_budget_mb=6144,
        offload_dir=Path("/tmp/offload"),
    )

    outputs = artifacts.write_wrapper_bundle(config, tmp_path)

    wrapper_dir = tmp_path / "wrapper"
    assert outputs == {
        "module": wrapper_dir / "project_wrapper.py",
        "config": wrapper_dir / "config.json",
        "readme": wrapper_dir / "README_integration.md",
    }

    module_source = outputs["module"].read_text(encoding="utf-8")
    assert "ChatAndEmbed" in module_source
    assert str(config.lora_dir) in module_source

    config_data = json.loads(outputs["config"].read_text(encoding="utf-8"))
    assert config_data["base_model_id"] == "dummy/model"
    assert config_data["lora_dir"] == str(config.lora_dir)
    assert config_data["vram_budget_mb"] == config.vram_budget_mb

    readme_contents = outputs["readme"].read_text(encoding="utf-8")
    assert "Wrapper Integration" in readme_contents
    assert str(config.lora_dir) in readme_contents
